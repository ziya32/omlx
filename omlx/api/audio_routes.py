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
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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

# Video container extensions that should be routed through ffmpeg decoding.
# mlx-audio only recognises audio-specific extensions (m4a, aac, ogg, opus),
# so we remap video containers to .m4a before handing off. ffmpeg detects the
# actual format from file content, not the extension.
_VIDEO_CONTAINERS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"}


# ---------------------------------------------------------------------------
# Auth dependency — wired by server.py via set_auth_dependency()
# ---------------------------------------------------------------------------

_auth_dependency = None
_security = HTTPBearer(auto_error=False)


def set_auth_dependency(dep):
    """Set the auth dependency function (called by server.py after verify_api_key is defined)."""
    global _auth_dependency
    _auth_dependency = dep


async def _verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> bool:
    """Forward to server's verify_api_key if wired, otherwise reject."""
    if _auth_dependency is None:
        raise HTTPException(status_code=401, detail="Auth not configured")
    return await _auth_dependency(request=request, credentials=credentials)


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

    Converts ModelNotFoundError to HTTP 404 so callers don't need to
    handle it individually.
    """
    from omlx.exceptions import ModelNotFoundError

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

    pool.acquire_engine(resolved)
    try:
        yield engine
    finally:
        pool.release_engine(resolved)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    http_request: Request,
    _: bool = Depends(_verify_auth),
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

    pool = _get_engine_pool()
    model = _resolve_model(model)

    if audio_file is None:
        raise HTTPException(status_code=400, detail="Audio file is required")
    if model is None:
        raise HTTPException(status_code=400, detail="Model is required")

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
                output = await engine.transcribe(
                    tmp_path,
                    language=language,
                    prompt=str(prompt) if prompt else None,
                )
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

        return _format_transcription_response(output, response_format)
    finally:
        os.unlink(tmp_path)


@router.post("/v1/audio/speech")
async def create_speech(
    request: AudioSpeechRequest,
    stream: bool = False,
    _: bool = Depends(_verify_auth),
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

    Query params:
    - stream: if true, stream raw PCM chunks as they're generated

    Returns binary WAV audio (24 kHz) or streaming PCM.
    """
    from omlx.engine.tts import (
        TTSEngine,
        _SUPPORTED_FORMATS,
        _FORMAT_MEDIA_TYPES,
        _FORMAT_EXTENSIONS,
        _convert_wav,
    )
    from omlx.exceptions import AudioError, VoiceCloningError

    req_id = str(uuid.uuid4())

    # Validate input is non-empty
    if not request.input:
        raise HTTPException(status_code=400, detail="'input' field must not be empty")

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

    if stream:
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

        # For streaming TTS, acquire the engine for the entire stream
        pool = _get_engine_pool()
        sm = _get_settings_manager()
        resolved = pool.resolve_model_id(request.model, sm) if sm else request.model
        engine = await pool.get_engine(resolved)
        if not isinstance(engine, TTSEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a text-to-speech model",
            )
        pool.acquire_engine(resolved)

        async def _stream_pcm():
            try:
                async for chunk in engine.stream_synthesize(
                    text=request.input,
                    voice=speaker,
                    instructions=instruct,
                    ref_audio=request.ref_audio,
                    ref_text=request.ref_text,
                ):
                    yield chunk.audio_bytes
            finally:
                pool.release_engine(resolved)

        return StreamingResponse(
            _stream_pcm(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Bits-Per-Sample": "16",
            },
        )

    start_time = time.perf_counter()

    async with _use_engine(request.model) as engine:
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
                ref_audio=request.ref_audio,
                ref_text=request.ref_text,
            )
        except VoiceCloningError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except AudioError as e:
            raise HTTPException(status_code=422, detail=str(e))

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
    _: bool = Depends(_verify_auth),
):
    """Audio processing endpoint (speech enhancement, source separation, STS).

    Accepts a multipart audio file upload and a model identifier, processes
    the audio through an STS engine (e.g. DeepFilterNet, MossFormer2,
    SAMAudio, LFM2.5-Audio), and returns WAV bytes of the processed audio.
    """
    from omlx.engine.sts import STSEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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

    return Response(content=wav_bytes, media_type="audio/wav")


@router.get("/v1/audio/speakers")
async def list_speakers(
    model: str | None = None,
    _: bool = Depends(_verify_auth),
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
    _: bool = Depends(_verify_auth),
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
