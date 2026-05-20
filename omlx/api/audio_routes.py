# SPDX-License-Identifier: Apache-2.0
"""
Audio API routes for oMLX.

This module provides OpenAI-compatible audio endpoints:
- POST /v1/audio/transcriptions  - Speech-to-Text (streaming SSE + batch)
- POST /v1/audio/speech          - Text-to-Speech (format conversion, speed, streaming PCM)
- POST /v1/audio/process         - Speech-to-Speech / audio processing
- GET  /v1/audio/speakers        - List available TTS speakers
- GET  /v1/audio/languages       - List supported ASR languages
"""

import asyncio
import base64
import json
import logging
import math
import os
import re
import tempfile
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from ..engine.audio_utils import wav_bytes_to_pcm_frames, wav_header
from .audio_models import (
    AudioSpeechRequest,
    AudioTranscriptionResponse,
    LanguagesResponse,
    SpeakersResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum upload size for audio files (100 MB).
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024

# Maximum base64-encoded ref_audio size (~15 MB raw audio, enough for ~60s).
MAX_REF_AUDIO_BASE64_BYTES = 20 * 1024 * 1024

# Default native TTS chunk cadence. Keep this below the mlx-audio default to
# improve TTFT while still letting the model process the full input at once.
DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.2
MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.01

# Video container extensions that should be routed through ffmpeg decoding.
# mlx-audio only recognises audio-specific extensions (m4a, aac, ogg, opus),
# so we remap video containers to .m4a before handing off. ffmpeg detects the
# actual format from file content, not the extension.
_VIDEO_CONTAINERS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"}


# ---------------------------------------------------------------------------
# Auth wiring lives in server.py at router-include time via
# ``app.include_router(audio_router, dependencies=[Depends(verify_api_key)])``.
# Tests that mount the bare router get an unauthenticated router — install
# auth via ``app.dependency_overrides`` or by re-including with a deps list.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Engine pool accessor — patched in tests via omlx.api.audio_routes._get_engine_pool
# ---------------------------------------------------------------------------

def _get_engine_pool():
    """Return the active EnginePool from server state.

    Imported lazily to avoid a circular import at module load time.
    Can be replaced in tests via patch('omlx.api.audio_routes._get_engine_pool').
    """
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve a model alias to its real model ID.

    Delegates to the same resolve_model_id used by LLM/chat endpoints,
    ensuring audio endpoints handle aliases consistently.
    """
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


# ---------------------------------------------------------------------------
# Settings manager accessor — wired by server.py via set_settings_manager_getter()
# ---------------------------------------------------------------------------

_settings_manager_getter = None


def set_settings_manager_getter(getter):
    """Set the callback function to get the settings manager."""
    global _settings_manager_getter
    _settings_manager_getter = getter


def _get_settings_manager():
    """Return the current SettingsManager, or None."""
    if _settings_manager_getter is None:
        return None
    return _settings_manager_getter()


# ---------------------------------------------------------------------------
# Local use_engine context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _use_engine(model_id: str):
    """Get an engine with eviction protection for the duration of the block.

    Converts ModelNotFoundError to HTTP 404, EngineEvictedError (raised by
    pool.ensure_engine_alive on the post-get-engine race check) to HTTP 503,
    and ImportError (raised by audio engine load when an optional dependency
    like mlx-audio isn't installed) to HTTP 501 with the engine's actionable
    install message — so callers don't need to handle them individually.
    """
    from omlx.exceptions import EngineEvictedError, ModelNotFoundError

    pool = _get_engine_pool()
    sm = _get_settings_manager()
    resolved = pool.resolve_model_id(model_id, sm) if sm else model_id
    try:
        engine = await pool.get_engine(resolved)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {avail}",
        ) from exc
    except ImportError as exc:
        # Engine load needed an optional dep that isn't installed (e.g.
        # mlx-audio for TTS/STT/STS).  Surface the engine's actionable
        # message ("Install it with: pip install 'omlx[audio]'") via 501,
        # not 503 — this isn't a transient capacity problem, the server
        # genuinely lacks the functionality until ops installs the dep.
        # Using 501 also keeps clients with 503-retry policies from
        # uselessly pounding the endpoint.
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    pool.acquire_engine(resolved)
    try:
        # Close the race window between pool.get_engine's last yield
        # and the caller touching `engine`: if the process memory
        # enforcer unloaded the model in between, fail fast with 503
        # instead of invoking methods on a stale reference.
        try:
            pool.ensure_engine_alive(resolved, engine)
        except EngineEvictedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        yield engine
    finally:
        pool.release_engine(resolved)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_audio_request(model_id: str) -> None:
    """Record an audio-API request in server metrics, with zero token counts.

    Audio endpoints (transcribe / speech / process) don't produce token
    deltas, but they still count as work the server did and should
    appear in request totals. Mirrors main's recording for completion-
    style endpoints (see ``server.get_server_metrics``).
    """
    try:
        from ..server_metrics import get_server_metrics
        get_server_metrics().record_request_complete(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model_id=model_id,
        )
    except Exception as exc:
        logger.warning("Failed to record audio metrics for %s: %s", model_id, exc)


async def _read_upload(file: UploadFile) -> bytes:
    """Read an uploaded file in chunks, bailing early if it exceeds the limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio file exceeds maximum allowed size "
                    f"({MAX_AUDIO_UPLOAD_BYTES} bytes)"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _format_transcription_response(output, response_format: str):
    """Build the appropriate response object for a completed transcription."""
    if response_format == "text":
        return PlainTextResponse(output.text)

    if response_format == "verbose_json":
        segments = []
        if output.segments:
            for i, seg in enumerate(output.segments):
                if isinstance(seg, dict):
                    segments.append(TranscriptionSegment(
                        id=seg.get("id", i),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", ""),
                    ))
        return VerboseTranscriptionResponse(
            text=output.text,
            language=output.language,
            duration=output.duration,
            segments=segments,
        )

    return AudioTranscriptionResponse(
        text=output.text,
        language=output.language,
        duration=output.duration,
    )


async def _stream_transcription(
    req_id: str,
    model_str: str,
    tmp_path: str,
    language: str,
    prompt: str | None,
    response_format: str,
):
    """SSE generator that yields progress events during transcription.

    Each chunk completion sends a progress event so the HTTP connection
    stays alive — the client's read timeout resets on every received event,
    allowing transcription of arbitrarily long audio files.

    Events:
        data: {"type":"progress","chunk":1,"total_chunks":10,"chunk_text":"..."}
        data: {"type":"transcription","text":"...","language":"en","duration":3600.0}
        data: [DONE]
    """
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def _on_progress(*, chunk: int, total_chunks: int, chunk_text: str):
        await progress_queue.put({
            "type": "progress",
            "chunk": chunk,
            "total_chunks": total_chunks,
            "chunk_text": chunk_text,
        })

    try:
        start_time = time.perf_counter()

        # Run transcription in a background task so we can yield SSE events
        # from the progress queue concurrently.
        async def _run_transcription():
            async with _use_engine(model_str) as engine:
                from omlx.engine.stt import STTEngine
                if not isinstance(engine, STTEngine):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{model_str}' is not a speech-to-text model",
                    )
                return await engine.transcribe(
                    tmp_path,
                    language=language,
                    prompt=str(prompt) if prompt else None,
                    on_progress=_on_progress,
                )

        task = asyncio.create_task(_run_transcription())

        # Yield progress events as they arrive. When the task completes,
        # it stops producing events and we break out.
        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                continue

        # Drain any remaining progress events
        while not progress_queue.empty():
            event = progress_queue.get_nowait()
            yield f"data: {json.dumps(event)}\n\n"

        output = await task  # propagate exceptions
        elapsed = time.perf_counter() - start_time

        logger.info(
            f"Transcription [{req_id}]: {elapsed:.3f}s, "
            f"language={output.language}, "
            f"duration={output.duration}s"
        )

        # Final result event
        result = {
            "type": "transcription",
            "text": output.text,
            "language": output.language,
            "duration": output.duration,
        }
        if response_format == "verbose_json" and output.segments:
            result["segments"] = [
                {"id": i, "start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": s.get("text", "")}
                for i, s in enumerate(output.segments) if isinstance(s, dict)
            ]
        _record_audio_request(model_str)
        yield f"data: {json.dumps(result)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        os.unlink(tmp_path)


def _read_speakers_from_config(model_path: str) -> list[str]:
    """Read TTS speaker list from model config.json without loading the engine."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return []
    try:
        with open(config_path) as f:
            config = json.load(f)
        # VoiceDesign and Base models don't have preset speakers
        if config.get("tts_model_type") in ("voice_design", "base"):
            return []
        talker_config = config.get("talker_config", {})
        spk_id = talker_config.get("spk_id", {})
        return list(spk_id.keys())
    except Exception:
        return []


def _decode_ref_audio_base64(request: AudioSpeechRequest) -> Optional[bytes]:
    """Validate and decode optional base64 ref_audio from a TTS request."""
    if request.ref_audio is None:
        return None

    if not request.ref_text:
        raise HTTPException(
            status_code=400,
            detail="'ref_text' is required when 'ref_audio' is provided "
            "(must be the transcript of the reference audio)",
        )
    if len(request.ref_audio) > MAX_REF_AUDIO_BASE64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"ref_audio exceeds maximum allowed size "
                f"({MAX_REF_AUDIO_BASE64_BYTES} bytes base64, "
                f"~60 seconds of audio)"
            ),
        )
    try:
        return base64.b64decode(request.ref_audio, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 encoding in 'ref_audio' field",
        )


def _write_ref_audio_tempfile(audio_bytes: Optional[bytes]) -> Optional[str]:
    """Persist decoded ref audio to a temp file if present."""
    if audio_bytes is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(audio_bytes)
        return tmp.name
    finally:
        tmp.close()


def _cleanup_tempfile(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _resolve_tts_streaming_interval(request: AudioSpeechRequest) -> float:
    """Return a native TTS streaming interval that is safe for mlx-audio."""
    if request.streaming_interval is None:
        return DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS

    interval = request.streaming_interval
    if (
        not math.isfinite(interval)
        or interval < MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "'streaming_interval' must be at least "
                f"{MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS} seconds"
            ),
        )
    return interval


def _split_tts_text(text: str, max_chars: int = 300) -> list[str]:
    """Split TTS input into conservative sentence-like chunks."""
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush_current()
            parts = re.split(r"(?<=[,;:，；：])\s*", sentence)
            parts = [p.strip() for p in parts if p and p.strip()]
            buffer = ""
            for part in parts or [sentence]:
                while len(part) > max_chars:
                    if buffer:
                        chunks.append(buffer.strip())
                        buffer = ""
                    chunks.append(part[:max_chars].strip())
                    part = part[max_chars:].strip()
                if not part:
                    continue
                candidate = f"{buffer} {part}".strip() if buffer else part
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = part
            if buffer:
                chunks.append(buffer.strip())
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            flush_current()
            current = sentence
        else:
            current = candidate

    flush_current()
    return chunks or [text]


async def _stream_speech_response(
    engine,
    request: AudioSpeechRequest,
    ref_audio_path: Optional[str],
    streaming_interval: float,
) -> AsyncIterator[bytes]:
    """Stream sentence-level TTS as a single WAV header plus PCM chunks."""
    try:
        if (
            hasattr(engine, "supports_native_tts_streaming")
            and engine.supports_native_tts_streaming()
            and hasattr(engine, "stream_synthesize_pcm")
        ):
            logger.info(
                "TTS native streaming start: model=%s, text_len=%d, voice=%s",
                request.model, len(request.input), request.voice,
            )
            stream_format: Optional[tuple[int, int, int]] = None
            try:
                async for sample_rate, channels, sample_width, pcm_bytes in engine.stream_synthesize_pcm(
                    request.input,
                    voice=request.voice,
                    speed=request.speed,
                    instructions=request.instructions,
                    ref_audio=ref_audio_path,
                    ref_text=request.ref_text,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    max_tokens=request.max_tokens,
                    streaming_interval=streaming_interval,
                ):
                    fmt = (sample_rate, channels, sample_width)
                    if stream_format is None:
                        stream_format = fmt
                        yield wav_header(
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_width=sample_width,
                        )
                    elif fmt != stream_format:
                        raise RuntimeError(
                            "Inconsistent native streaming PCM format: "
                            f"expected {stream_format}, got {fmt}"
                        )
                    if pcm_bytes:
                        yield pcm_bytes
            except NotImplementedError:
                if stream_format is not None:
                    raise
                logger.info(
                    "TTS native streaming unavailable at runtime; falling back "
                    "to segmented synthesis: model=%s",
                    request.model,
                )
            else:
                return

        segments = _split_tts_text(request.input)
        logger.info(
            "TTS streaming start: model=%s, text_len=%d, segments=%d, voice=%s",
            request.model, len(request.input), len(segments), request.voice,
        )

        stream_format: Optional[tuple[int, int, int]] = None
        for idx, segment in enumerate(segments, start=1):
            output = await engine.synthesize(
                segment,
                voice=request.voice,
                speed=request.speed,
                instructions=request.instructions,
                ref_audio=ref_audio_path,
                ref_text=request.ref_text,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                max_tokens=request.max_tokens,
            )
            # engine.synthesize returns SpeechOutput; pull the WAV bytes off
            # so wav_bytes_to_pcm_frames can split header from frames. Some
            # tests pass raw bytes through AsyncMock; accept either shape.
            wav_bytes = output.audio_bytes if hasattr(output, "audio_bytes") else output
            sample_rate, channels, sample_width, pcm_bytes = wav_bytes_to_pcm_frames(wav_bytes)
            fmt = (sample_rate, channels, sample_width)
            if stream_format is None:
                stream_format = fmt
                yield wav_header(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
            elif fmt != stream_format:
                raise RuntimeError(
                    "Inconsistent WAV format across TTS segments: "
                    f"expected {stream_format}, got {fmt}"
                )
            logger.debug(
                "TTS streaming segment %d/%d: text_len=%d, pcm_bytes=%d",
                idx, len(segments), len(segment), len(pcm_bytes),
            )
            if pcm_bytes:
                yield pcm_bytes
    finally:
        _cleanup_tempfile(ref_audio_path)


async def _stream_with_prefetched_chunk(
    first_chunk: bytes,
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Yield a chunk fetched before response headers, then the rest of the stream."""
    try:
        yield first_chunk
        async for chunk in stream:
            yield chunk
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    http_request: Request,
):
    """
    Transcribe audio to text.

    OpenAI-compatible endpoint for audio transcription (ASR).

    Accepts multipart/form-data with:
    - file: audio file (wav, mp3, m4a, etc.)
    - model: model ID (e.g., "Qwen3-ASR-1.7B-bf16")
    - language: ISO language code or "auto" (default)
    - prompt: optional text to guide transcription (Whisper models)
    - response_format: "json" (default), "verbose_json", or "text"
    - stream: "true" to get SSE progress events per chunk (keeps connection
      alive for long audio). The final event contains the full result.
    """
    from omlx.engine.stt import STTEngine
    from omlx.exceptions import AudioError, InvalidAudioFormatError

    req_id = str(uuid.uuid4())
    form = await http_request.form()
    audio_file = form.get("file")
    model = form.get("model")
    language = form.get("language")
    prompt = form.get("prompt")
    response_format = str(form.get("response_format", "json"))
    stream = str(form.get("stream", "false")).lower() == "true"
    # oMLX extension exposing mlx-audio's native word-level alignment for
    # Whisper models. When True, each segment in the response includes a
    # ``words`` array of ``{word, start, end, probability}`` objects.
    word_timestamps = str(form.get("word_timestamps", "false")).lower() == "true"
    # oMLX extension raising the underlying model's output cap. Useful for
    # long audio with models like VibeVoice-ASR whose mlx-audio default
    # (8192) truncates ~24 min files. Precedence: request form > per-model
    # ModelSettings.max_tokens > model's own generate() default.
    _max_tokens_raw = form.get("max_tokens")
    req_max_tokens: int | None
    try:
        req_max_tokens = int(_max_tokens_raw) if _max_tokens_raw not in (None, "") else None
    except (TypeError, ValueError):
        req_max_tokens = None

    if audio_file is None:
        raise HTTPException(status_code=400, detail="Audio file is required")
    if model is None:
        raise HTTPException(status_code=400, detail="Model is required")

    logger.info(f"Transcription [{req_id}] received: model={model}")

    model = _resolve_model(str(model))

    model_str = str(model)

    # Apply model settings defaults
    if language is None:
        sm = _get_settings_manager()
        if sm:
            ms = sm.get_settings(model_str)
            language = ms.default_language or "auto"
        else:
            language = "auto"
    language = str(language)

    # Determine temp file suffix from original filename.
    # Video container extensions (mp4, mkv, etc.) are remapped to .m4a so
    # mlx-audio routes them through ffmpeg instead of miniaudio (which can't
    # decode containers).
    filename = getattr(audio_file, "filename", None) or "audio.wav"
    suffix = os.path.splitext(filename)[1] or ".wav"
    if suffix.lower() in _VIDEO_CONTAINERS:
        suffix = ".m4a"

    # Write uploaded file to temp path for mlx_audio
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    if stream:
        return StreamingResponse(
            _stream_transcription(
                req_id, model_str, tmp_path, language, prompt, response_format,
            ),
            media_type="text/event-stream",
        )

    try:
        start_time = time.perf_counter()
        async with _use_engine(model_str) as engine:
            if not isinstance(engine, STTEngine):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_str}' is not a speech-to-text model",
                )
            try:
                # Effective max_tokens precedence: request > per-model
                # ModelSettings.max_tokens > model's own generate() default.
                effective_max_tokens = req_max_tokens
                if effective_max_tokens is None:
                    sm = _get_settings_manager()
                    if sm is not None:
                        try:
                            ms = sm.get_settings(model_str)
                            if ms is not None and getattr(ms, "max_tokens", None) is not None:
                                effective_max_tokens = ms.max_tokens
                        except Exception:
                            pass

                transcribe_kwargs: dict[str, object] = {
                    "language": language,
                    "prompt": str(prompt) if prompt else None,
                }
                if word_timestamps:
                    transcribe_kwargs["word_timestamps"] = True
                if effective_max_tokens is not None:
                    transcribe_kwargs["max_tokens"] = effective_max_tokens
                output = await engine.transcribe(tmp_path, **transcribe_kwargs)
            except InvalidAudioFormatError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except AudioError as e:
                raise HTTPException(status_code=422, detail=str(e))
        elapsed = time.perf_counter() - start_time

        logger.info(
            f"Transcription [{req_id}]: {elapsed:.3f}s, "
            f"language={output.language}, "
            f"duration={output.duration}s"
        )

        _record_audio_request(model_str)
        return _format_transcription_response(output, response_format)
    finally:
        os.unlink(tmp_path)


@router.post("/v1/audio/speech")
async def create_speech(
    request: AudioSpeechRequest,
):
    """
    Generate speech from text.

    OpenAI-compatible endpoint for text-to-speech (TTS).

    For CustomVoice models:
    - voice: preset speaker name (e.g., "ryan", "vivian")
    - instructions: emotional/tonal control (e.g., "Speak warmly")

    For VoiceDesign models:
    - voice: ignored
    - instructions: voice description (e.g., "A warm female narrator")

    When ``stream=true`` is set on the request body, the response is
    a streaming WAV (header + PCM frames) so clients can begin playback
    before synthesis completes. Otherwise a single binary WAV (24 kHz)
    is returned.
    """
    from omlx.engine.tts import (
        TTSEngine,
        _SUPPORTED_FORMATS,
        _FORMAT_MEDIA_TYPES,
        _FORMAT_EXTENSIONS,
        _convert_wav,
    )
    from omlx.exceptions import AudioError, ModelNotFoundError, VoiceCloningError

    req_id = str(uuid.uuid4())
    logger.info(
        f"Speech [{req_id}] received: model={request.model}, "
        f"format={request.response_format}"
    )

    fmt = request.response_format
    if fmt not in _SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {fmt}. Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}",
        )

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="'input' field must not be empty")
    streaming_interval = DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    if request.stream:
        if request.response_format not in (None, "wav"):
            raise HTTPException(
                status_code=400,
                detail="Streaming TTS currently only supports response_format='wav'",
            )
        streaming_interval = _resolve_tts_streaming_interval(request)

    # --- Validate and decode ref_audio (voice clone) ---
    # We decode to bytes here (cheap) and defer tempfile creation to each
    # branch (streaming / non-streaming) right before the engine call so
    # that a 4xx raised by sync validation below cannot leak a tempfile.
    ref_audio_bytes = _decode_ref_audio_base64(request)

    pool = _get_engine_pool()
    resolved_model = _resolve_model(request.model)

    # Apply model settings defaults
    speaker = request.voice if request.voice != "default" else None
    instruct = request.instructions
    if speaker is None or instruct is None:
        sm = _get_settings_manager()
        if sm:
            ms = sm.get_settings(request.model)
            if speaker is None and ms.default_voice:
                speaker = ms.default_voice
            if instruct is None and ms.default_instruct:
                instruct = ms.default_instruct

    # Load the engine via pool
    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except ImportError as exc:
        # Engine load needed an optional dep (mlx-audio).  Surface the
        # engine's actionable install message via 501 — the server lacks
        # the functionality until ops installs the dep, not a transient
        # 503 condition.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, TTSEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a text-to-speech model",
        )

    speed = request.speed
    if not (0.25 <= speed <= 4.0):
        raise HTTPException(
            status_code=400,
            detail=f"Speed must be between 0.25 and 4.0, got {speed}",
        )

    if request.stream:
        if speed != 1.0:
            raise HTTPException(
                status_code=400,
                detail="Speed adjustment is not supported in streaming mode",
            )
        if fmt not in ("wav", "pcm"):
            raise HTTPException(
                status_code=400,
                detail=f"Streaming only supports wav/pcm format, got {fmt}",
            )

        # Streaming TTS: hold the engine lease for the entire stream and
        # use upstream's _stream_speech_response helper, which tries native
        # PCM streaming first (engine.stream_synthesize_pcm) and falls back
        # to per-segment synthesize when the model lacks native streaming.
        # The _release_once / weakref.finalize pair guarantees the lease
        # is released exactly once, including the un-iterated case where
        # the client disconnects between handler return and Starlette's
        # body_iterator entry (Python closes an un-started async generator
        # without running its try/finally; without the finalizer the
        # engine_pool active_uses lease leaks and produces LIVELOCK_SUSPECT
        # on the next non-pinned model load — same fix applied to the chat
        # streaming handlers in server.py).
        pool = _get_engine_pool()
        sm = _get_settings_manager()
        resolved = pool.resolve_model_id(request.model, sm) if sm else request.model
        try:
            engine = await pool.get_engine(resolved)
        except ImportError as exc:
            # See non-streaming branch above — 501 for missing optional dep.
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        if not isinstance(engine, TTSEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a text-to-speech model",
            )
        pool.acquire_engine(resolved)

        stream_ref_audio_path = _write_ref_audio_tempfile(ref_audio_bytes)
        released_flag = [False]

        def _release_once() -> None:
            if not released_flag[0]:
                released_flag[0] = True
                pool.release_engine(resolved)
            # _stream_speech_response also unlinks ref_audio_path in its
            # own finally block, but _cleanup_tempfile is idempotent.
            _cleanup_tempfile(stream_ref_audio_path)

        async def _release_after(stream_iter):
            try:
                async for chunk in stream_iter:
                    yield chunk
            finally:
                _release_once()

        inner = _stream_speech_response(
            engine, request, stream_ref_audio_path, streaming_interval,
        )
        try:
            first_chunk = await inner.__anext__()
        except StopAsyncIteration as exc:
            _release_once()
            raise HTTPException(
                status_code=500,
                detail="TTS streaming produced no audio output",
            ) from exc
        except HTTPException:
            _release_once()
            raise
        except Exception as exc:
            _release_once()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        body = _release_after(_stream_with_prefetched_chunk(first_chunk, inner))
        # Fallback for the never-iterated case: fires on GC of `body`.
        weakref.finalize(body, _release_once)

        # Record on first-chunk readiness (matches non-stream recording
        # at request-completion: by here the engine has produced its
        # first PCM bytes and the response is committed).
        _record_audio_request(resolved)
        return StreamingResponse(
            body,
            media_type="audio/wav",
        )

    start_time = time.perf_counter()

    # Materialize ref_audio tempfile only now — past all sync validation.
    nonstream_ref_audio_path = _write_ref_audio_tempfile(ref_audio_bytes)
    try:
        # Pass the already-resolved id (computed at line 807) so the lease
        # path's get_engine uses the same model id as the validator above.
        # See Issue 7 resolve-once pattern — without this the get_engine
        # call inside _use_engine re-resolves and can land on a different
        # entry if the alias map churns mid-request.
        async with _use_engine(resolved_model) as engine:
            if not isinstance(engine, TTSEngine):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{request.model}' is not a text-to-speech model",
                )
            try:
                output = await engine.synthesize(
                    text=request.input,
                    voice=speaker,
                    instructions=instruct,
                    speed=speed,
                    ref_audio=nonstream_ref_audio_path,
                    ref_text=request.ref_text,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    max_tokens=request.max_tokens,
                )
            except VoiceCloningError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except AudioError as e:
                raise HTTPException(status_code=422, detail=str(e))
    finally:
        _cleanup_tempfile(nonstream_ref_audio_path)

    elapsed = time.perf_counter() - start_time

    # Convert format and/or apply speed if needed
    audio_bytes = output.audio_bytes
    if fmt != "wav" or speed != 1.0:
        if fmt == "pcm":
            # Strip WAV header for raw PCM
            audio_bytes = audio_bytes[44:]
        else:
            audio_bytes = _convert_wav(audio_bytes, fmt, speed=speed)

    media_type = _FORMAT_MEDIA_TYPES[fmt]
    ext = _FORMAT_EXTENSIONS[fmt]

    logger.info(
        f"Speech [{req_id}]: {elapsed:.3f}s, "
        f"duration={output.duration:.2f}s, "
        f"sample_rate={output.sample_rate}, "
        f"format={fmt}"
    )

    _record_audio_request(resolved_model)
    return StreamingResponse(
        iter([audio_bytes]),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{ext}",
        },
    )


@router.post("/v1/audio/process")
async def process_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """Audio processing endpoint (speech enhancement, source separation, STS).

    Accepts a multipart audio file upload and a model identifier, processes
    the audio through an STS engine (e.g. DeepFilterNet, MossFormer2,
    SAMAudio, LFM2.5-Audio), and returns WAV bytes of the processed audio.
    """
    from omlx.engine.sts import STSEngine

    model = _resolve_model(model)

    # Hold the engine lease for the duration of process() so STS doesn't
    # race onto the single-threaded MLX executor alongside an exclusive
    # VLM. Mirrors the LLM/embedding/rerank endpoints' acquire_engine
    # pattern in server.py.
    async with _use_engine(model) as engine:
        if not isinstance(engine, STSEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not a speech-to-speech / audio processing model",
            )

        # Save uploaded file to a temp path so the engine can open it by path.
        # Remap video container extensions to .m4a so mlx-audio routes them
        # through ffmpeg instead of miniaudio (which can't decode containers).
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        if suffix.lower() in _VIDEO_CONTAINERS:
            suffix = ".m4a"
        tmp_path = None
        try:
            content = await _read_upload(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                tmp.write(content)

            wav_bytes = await engine.process(tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    _record_audio_request(model)
    return Response(content=wav_bytes, media_type="audio/wav")


@router.get("/v1/audio/speakers")
async def list_speakers(
    model: str | None = None,
) -> SpeakersResponse:
    """
    List available TTS speakers for a CustomVoice model.

    Reads the speaker list from the model's config.json on disk — does NOT
    load the engine, so this endpoint is cheap and won't trigger model swaps.

    Returns an empty list for VoiceDesign/Base models (use instructions instead).
    """
    pool = _get_engine_pool()

    if model is None:
        # Find any TTS model (prefer loaded, fall back to any discovered)
        for mid, entry in pool._entries.items():
            if entry.engine_type == "audio_tts" and entry.engine is not None:
                model = mid
                break
        if model is None:
            for mid, entry in pool._entries.items():
                if entry.engine_type == "audio_tts":
                    model = mid
                    break
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="No TTS model found"
            )

    entry = pool._entries.get(model)
    if entry is None or entry.engine_type != "audio_tts":
        raise HTTPException(status_code=400, detail=f"Model {model} is not a TTS model")

    speakers = _read_speakers_from_config(entry.model_path)
    return SpeakersResponse(speakers=speakers)


@router.get("/v1/audio/languages")
async def list_languages(
    model: str | None = None,
) -> LanguagesResponse:
    """
    List supported languages for an ASR model.

    If no model is specified, uses the first available ASR model.
    """
    if model is None:
        pool = _get_engine_pool()
        for mid, entry in pool._entries.items():
            if entry.engine_type == "audio_stt":
                model = mid
                break
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="No ASR model specified and none available"
            )

    async with _use_engine(model) as engine:
        from omlx.engine.stt import STTEngine
        if not isinstance(engine, STTEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not a speech-to-text model",
            )
        languages = engine.get_languages()

    return LanguagesResponse(
        languages=languages,
        model=model,
    )
