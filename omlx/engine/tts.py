# SPDX-License-Identifier: Apache-2.0
"""
TTS (Text-to-Speech) engine for oMLX.

This module provides an engine for speech synthesis using mlx-audio.
Unlike LLM engines, TTS engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from ..mx_buffer_lock import gtrace, locked_free_and_clear, run_locked
from ..exceptions import AudioError, VoiceCloningError
from .audio_utils import DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE
from .audio_utils import audio_to_wav_bytes as _audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"wav", "mp3", "opus", "flac", "pcm"}

_FORMAT_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

_FORMAT_EXTENSIONS = {
    "wav": "wav",
    "mp3": "mp3",
    "opus": "opus",
    "flac": "flac",
    "pcm": "pcm",
}


def _convert_wav(wav_bytes: bytes, output_format: str, speed: float = 1.0) -> bytes:
    """Convert WAV bytes to another audio format using ffmpeg.

    Args:
        wav_bytes: Input WAV audio bytes.
        output_format: Target format ("mp3", "opus", "flac").
        speed: Playback speed multiplier (0.25-4.0). 1.0 = no change.

    Returns:
        Converted audio bytes.

    Raises:
        AudioError: If ffmpeg is not available or conversion fails.
    """
    import shutil
    import subprocess

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise AudioError(
            "ffmpeg is required for audio format conversion but was not found. "
            "Install with: brew install ffmpeg"
        )

    cmd = [ffmpeg_path, "-i", "pipe:0", "-f", output_format]

    # Apply speed change via atempo filter
    if speed != 1.0:
        filters = []
        remaining = speed
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining *= 2.0
        filters.append(f"atempo={remaining}")
        cmd.extend(["-af", ",".join(filters)])

    cmd.extend(["-y", "pipe:1"])

    try:
        result = subprocess.run(
            cmd,
            input=wav_bytes,
            capture_output=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise AudioError("ffmpeg not found")
    except subprocess.TimeoutExpired:
        raise AudioError("Audio conversion timed out (>5 min)")

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise AudioError(f"ffmpeg conversion failed: {stderr}")

    return result.stdout


@dataclass
class SpeechOutput:
    """Output from TTS synthesis."""

    audio_bytes: bytes  # WAV file bytes
    sample_rate: int = 24000
    duration: float = 0.0  # seconds


class TTSEngine(BaseNonStreamingEngine):
    """
    Engine for speech synthesis (Text-to-Speech).

    This engine wraps mlx-audio TTS models and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since synthesis is computed in a single forward pass.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the TTS engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs
        self._variant = "custom_voice"  # Detected during start() via _detect_variant

    @staticmethod
    def _audio_array_to_pcm_bytes(audio: Any) -> bytes:
        audio_array = np.array(audio).flatten()
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return (audio_array * 32767).astype(np.int16).tobytes()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    # Peak GPU working-set transient per generated speech code, in bytes.
    # Calibrated to the observed +3.76 GiB at max_tokens=4096 on this box
    # (ceil(3.76 GiB / 4096) ≈ 0.94 MiB, rounded up to 1 MiB). The transient
    # is the conv decode-stack activation for the whole-waveform
    # ``speech_tokenizer.decode(codes)`` → ``mx.eval`` — it scales with the
    # generated-code COUNT, not audio bytes. The single re-tunable knob:
    # re-measure if TTS precision / weights change. Backstops: the clamp in
    # estimate_working_set_bytes and the 2 GiB margin in EnginePool.
    _C_TTS_BYTES_PER_CODE: int = 1024 * 1024
    _CODES_MIN: int = 512        # clamp floor: short clips still reserve a little
    _CODES_MAX: int = 8192       # clamp ceiling: backstop a raised max_tokens
    _DEFAULT_MAX_CODES: int = 4096  # model default when max_tokens is unset
    _STREAM_CODES: int = 512     # native-streaming decodes per chunk, not all-at-once

    def estimate_working_set_bytes(self, **call_kwargs: Any) -> int:
        """Reserve for the all-at-once (non-streaming) ``synthesize`` decode.

        ``est = C_TTS * clamp(codes, 512, 8192)`` where ``codes`` is the
        generated-code count: ``max_tokens or 4096``, further capped by a
        real per-text upper bound (``max(75, len(text) * 6)``) so short clips
        reserve far less than the 4 GiB default (§4).
        """
        max_tokens = call_kwargs.get("max_tokens")
        text = call_kwargs.get("text")
        codes = max_tokens if max_tokens else self._DEFAULT_MAX_CODES
        if text is not None:
            codes = min(codes, max(75, len(text) * 6))
        codes = max(self._CODES_MIN, min(codes, self._CODES_MAX))
        return self._C_TTS_BYTES_PER_CODE * codes

    def _estimate_streaming_working_set_bytes(self) -> int:
        """Reserve for native streaming, which decodes per chunk (≈0.5 GiB).

        Far smaller than the all-at-once estimate so the healthy streaming
        path never spuriously evicts a coexisting model (§4).
        """
        return self._C_TTS_BYTES_PER_CODE * self._STREAM_CODES

    def supports_native_tts_streaming(self) -> bool:
        """Return whether the loaded model exposes model-native audio streaming."""
        if self._model is None:
            return False
        import inspect

        try:
            gen_params = inspect.signature(self._model.generate).parameters
        except (TypeError, ValueError):
            return False
        return "stream" in gen_params and "streaming_interval" in gen_params

    @staticmethod
    def _detect_variant(model_path: str) -> str:
        """Detect TTS variant from config.json tts_model_type field."""
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                tts_type = config.get("tts_model_type", "")
                if tts_type == "voice_design":
                    return "voice_design"
                if tts_type == "base":
                    return "base"
            except Exception:
                pass
        return "custom_voice"

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        mlx-audio is imported here (lazily) to avoid module-level errors
        when the package is not installed.
        """
        if self._model is not None:
            return

        logger.info(f"Starting TTS engine: {self._model_name}")

        try:
            from mlx_audio.tts.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for TTS inference. "
                'Install it with: pip install "omlx[audio]"'
            ) from exc

        model_name = self._model_name

        def _load_sync():
            try:
                return _load_model(model_name, strict=True)
            except ValueError as exc:
                if "Expected shape" not in str(exc):
                    raise
                # mlx-audio bug: sanitize() merges quantization scales into
                # weights before apply_quantization() can detect them, causing
                # shape mismatches for quantized models (e.g. VibeVoice 8-bit).
                # Retry with strict=False so mismatched layers are skipped.
                logger.warning(
                    "Strict weight loading failed for %s (likely quantized "
                    "model with mlx-audio compatibility issue), retrying "
                    "with strict=False: %s", model_name, exc,
                )
                return _load_model(model_name, strict=False)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), lambda: run_locked(_load_sync))
        self._variant = self._detect_variant(self._model_name)
        logger.info(
            f"TTS engine started: {self._model_name} (variant={self._variant})"
        )

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        # Latch the cooperative-abort flag BEFORE clearing _model so any
        # handler racing with stop sees RequestAbortedError (-> 503) on
        # its next _raise_if_aborted checkpoint rather than the plain
        # "Engine not started" RuntimeError (-> 500). Issue 4.
        self._mark_stopped()
        if self._model is None:
            return

        logger.info(f"Stopping TTS engine: {self._model_name}")
        # Free the model ref + gc on the executor under the buffer lock, so the
        # eviction's buffer frees serialize with in-flight generation on the
        # executor instead of racing it from the event-loop thread (#85).
        holder = [self._model]
        self._model = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: locked_free_and_clear(holder.clear)
        )
        logger.info(f"TTS engine stopped: {self._model_name}")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> SpeechOutput:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Optional voice/speaker identifier
            speed: Speech speed multiplier (1.0 = normal)
            instructions: Optional voice description for instruct-capable models
            ref_audio: Optional path to reference audio file (voice cloning)
            ref_text: Optional transcript of the reference audio
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty for generation
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            SpeechOutput with WAV-encoded audio bytes, sample rate, and duration
        """
        # Cooperative-abort checkpoint BEFORE the model-None guard. Issue 4.
        self._raise_if_aborted()
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        import time

        logger.info(
            "TTS synthesize: model=%s, text_len=%d, voice=%s, speed=%.1f, ref_audio=%s",
            self._model_name, len(text), voice, speed,
            "yes" if ref_audio else "no",
        )

        model = self._model
        t0 = time.monotonic()

        def _build_generate_kwargs() -> Dict[str, Any]:
            # VoiceDesign variant: always uses instruct, no signature inspection
            if self._variant == "voice_design":
                return {
                    "text": text,
                    "instruct": instructions or "A neutral narrator",
                    **kwargs,
                }

            # Base variant with reference audio (voice cloning)
            if self._variant == "base" and ref_audio:
                kw: Dict[str, Any] = {"text": text, "ref_audio": ref_audio}
                if ref_text:
                    kw["ref_text"] = ref_text
                kw.update(kwargs)
                return kw

            gen_kwargs: Dict[str, Any] = {
                "text": text,
                "verbose": False,
            }
            import inspect
            gen_params = inspect.signature(model.generate).parameters
            if voice is not None:
                # Route voice to the correct generate() kwarg.
                # Models with 'voice' param (CustomVoice, Kokoro) get it as
                # a speaker name. Models with only 'instruct' (non-Qwen TTS)
                # get it as a voice description fallback.
                if "voice" in gen_params:
                    gen_kwargs["voice"] = voice
                elif "instruct" in gen_params:
                    gen_kwargs["instruct"] = voice
            if instructions is not None and "instruct" in gen_params:
                gen_kwargs["instruct"] = instructions
            if speed != 1.0:
                gen_kwargs["speed"] = speed
            if ref_audio is not None and "ref_audio" in gen_params:
                gen_kwargs["ref_audio"] = ref_audio
                gen_kwargs["ref_text"] = ref_text
            # Generation params (only add non-None values)
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            gen_kwargs.update(kwargs)
            return gen_kwargs

        def _create_gen():
            # iter() so a model.generate() that returns a list (mocks,
            # eager iterables) still supports next(g, _SEG_DONE) per
            # segment. Map raw model errors to AudioError/VoiceCloning-
            # Error so the audio endpoint surfaces 422 instead of 500.
            try:
                return iter(model.generate(**_build_generate_kwargs()))
            except Exception as e:
                if self._variant == "base" and ref_audio:
                    raise VoiceCloningError(
                        f"Voice cloning failed: {e}"
                    ) from e
                raise AudioError(
                    f"TTS generation failed: {e}"
                ) from e

        _SEG_DONE = object()

        def _next_seg(g):
            # Same wrapping: per-segment errors from mlx-audio's iterator
            # become AudioError, not HTTP 500.
            try:
                gtrace("gen.next.enter")
                seg = next(g, _SEG_DONE)
                gtrace("gen.next.exit", "DONE" if seg is _SEG_DONE else "seg")
            except Exception as e:
                gtrace("gen.next.error", repr(e))
                raise AudioError(
                    f"TTS segment generation failed: {e}"
                ) from e
            if seg is _SEG_DONE:
                return _SEG_DONE, None
            # Materialize the segment audio HERE — on the single MLX executor
            # thread and under the buffer-access lock. Reading the Metal buffer
            # (np.array, plus the bf16 astype) off the executor / unlocked lets
            # a concurrent model-switch buffer-pool reclaim (clear_cache or a
            # load allocation, run under the same lock) reclaim the pool mid
            # read, corrupting the in-flight GPU command buffer -> garbled audio
            # or SIGABRT. This mirrors _extract_tensor_bytes and the STT/STS/
            # streaming paths, which all read on the executor. See issue #85.
            try:
                audio = seg.audio
                if isinstance(audio, mx.array) and audio.dtype == mx.bfloat16:
                    audio = audio.astype(mx.float32)
                audio_np = run_locked(lambda: np.array(audio))
            except (AudioError, VoiceCloningError):
                raise
            except Exception as exc:
                raise AudioError(
                    f"TTS segment processing failed: {exc}"
                ) from exc
            seg_sr = getattr(seg, "sample_rate", None)
            seg_sr = int(seg_sr) if isinstance(seg_sr, (int, float)) else None
            gtrace("gen.seg.materialized", f"samples={getattr(audio_np, 'size', '?')} sr={seg_sr}")
            return audio_np, seg_sr

        def _finalize(chunks, sample_rate):
            audio = np.concatenate(chunks, axis=0)
            wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
            duration = len(audio) / sample_rate if sample_rate else 0.0
            return SpeechOutput(
                audio_bytes=wav_bytes,
                sample_rate=sample_rate,
                duration=duration,
            )

        activity_id = self._begin_activity(
            "synthesizing speech",
            detail="Synthesizing speech",
            metadata={"text_length": len(text)},
        )
        try:
            # Reserve the whole-waveform decode transient against the Metal
            # wall BEFORE running it (§3d). The non-streaming decode
            # materializes the full waveform in one mx.eval — a multi-GB
            # transient no resident-weight accounting models. reserve_inference
            # is a strict no-op when est<=0 / no Metal cap. Guarded so
            # non-pooled / unit-test use (self._pool is None) never touches the
            # pool. Held across the generate/decode loop; released on every
            # exit path (normal / AudioError / abort / cancel).
            async with contextlib.AsyncExitStack() as _reserve_stack:
                if self._pool is not None:
                    await _reserve_stack.enter_async_context(
                        self._pool.reserve_inference(
                            self._model_name,
                            self.estimate_working_set_bytes(
                                text=text, max_tokens=max_tokens
                            ),
                        )
                    )
                loop = asyncio.get_running_loop()
                executor = get_mlx_executor()

                # Step 1: create the generator on the executor (quick).
                gen = await loop.run_in_executor(executor, _create_gen)
                self._raise_if_aborted()

                # Prefer model.sample_rate (Qwen3-TTS), fall back to the first
                # segment's sample_rate, finally _DEFAULT_SAMPLE_RATE.
                model_sr = getattr(model, "sample_rate", None)
                sample_rate = int(model_sr) if isinstance(model_sr, (int, float)) else None

                # Step 2: consume one segment per executor call. Yields the
                # event loop between segments so LLM/VLM scheduler.step() can
                # interleave on the same single-threaded MLX executor — without
                # this, a long TTS synth blocks every other engine.
                # _next_seg materializes each segment's audio to a NumPy array on
                # the executor (under the buffer lock), so no mx.array is touched
                # off the executor thread. Segment audio access there can raise if
                # the producing model was torn down mid-loop (e.g. enforcer evicted
                # but the iterator had a partial result in flight) — surfaced as
                # AudioError (422) not 500.
                audio_chunks: list[np.ndarray] = []
                while True:
                    seg_audio, seg_sr = await loop.run_in_executor(
                        executor, _next_seg, gen
                    )
                    # Abort between segments — discards any computed segments.
                    self._raise_if_aborted()
                    if seg_audio is _SEG_DONE:
                        break
                    audio_chunks.append(seg_audio)
                    if sample_rate is None and seg_sr is not None:
                        sample_rate = seg_sr

                if sample_rate is None:
                    sample_rate = _DEFAULT_SAMPLE_RATE
                if not audio_chunks:
                    # Under enforcer abort the iterator may exit cleanly
                    # before yielding any segment (the model's generate()
                    # loop unwinds when its scheduler is torn down).
                    # ``AudioError`` so the endpoint returns 422 — retryable —
                    # rather than 500.
                    raise AudioError("TTS model produced no audio output")

                # Step 3: concatenate + WAV-encode on the executor (numpy + io).
                # Wrap _finalize errors so np/wav-encoder failures (e.g. mx
                # array detached from a freed model) surface as 422 not 500.
                try:
                    result = await loop.run_in_executor(
                        executor, _finalize, audio_chunks, sample_rate
                    )
                except (AudioError, VoiceCloningError):
                    raise
                except Exception as exc:
                    raise AudioError(
                        f"TTS finalize failed: {exc}"
                    ) from exc

                elapsed = time.monotonic() - t0
                logger.info(
                    "TTS synthesize done: model=%s, %.2fs, %d bytes output",
                    self._model_name, elapsed, len(result.audio_bytes),
                )
                return result
        finally:
            await self._finish_activity(activity_id)

    async def stream_synthesize_pcm(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming_interval: float = 0.4,
        **kwargs,
    ) -> AsyncIterator[tuple[int, int, int, bytes]]:
        """Stream synthesized PCM chunks from models that natively support it."""
        # Cooperative-abort checkpoint BEFORE the model-None guard. Issue 4.
        self._raise_if_aborted()
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        if not self.supports_native_tts_streaming():
            raise NotImplementedError("Loaded TTS model does not expose native streaming")

        import inspect
        import time

        logger.info(
            "TTS native stream start: model=%s, text_len=%d, voice=%s, interval=%.2fs",
            self._model_name, len(text), voice, streaming_interval,
        )

        model = self._model
        t0 = time.monotonic()

        def _build_generate_kwargs() -> Dict[str, Any]:
            gen_kwargs: Dict[str, Any] = {
                "text": text,
                "verbose": False,
                "stream": True,
            }
            gen_params = inspect.signature(model.generate).parameters
            if "streaming_interval" in gen_params:
                gen_kwargs["streaming_interval"] = streaming_interval
            if voice is not None:
                if "voice" in gen_params:
                    gen_kwargs["voice"] = voice
                elif "instruct" in gen_params:
                    gen_kwargs["instruct"] = voice
            if instructions is not None and "instruct" in gen_params:
                gen_kwargs["instruct"] = instructions
            if speed != 1.0:
                gen_kwargs["speed"] = speed
            if ref_audio is not None and "ref_audio" in gen_params:
                gen_kwargs["ref_audio"] = ref_audio
                gen_kwargs["ref_text"] = ref_text
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            gen_kwargs.update(kwargs)
            return gen_kwargs

        iterator: Any = None
        sentinel = object()
        chunk_count = 0
        total_bytes = 0

        def _next_pcm_chunk():
            nonlocal iterator
            if iterator is None:
                iterator = iter(model.generate(**_build_generate_kwargs()))
            try:
                result = next(iterator)
            except StopIteration:
                return sentinel
            audio = getattr(result, "audio", None)
            if audio is None:
                return None
            sample_rate = int(
                getattr(result, "sample_rate", getattr(model, "sample_rate", _DEFAULT_SAMPLE_RATE))
            )
            return sample_rate, 1, 2, self._audio_array_to_pcm_bytes(audio)

        activity_id = self._begin_activity(
            "streaming speech",
            detail="Streaming speech",
            metadata={"text_length": len(text)},
        )
        try:
            # Native streaming decodes per chunk (decoder.streaming_step), not
            # all-at-once, so it under-reserves (≈0.5 GiB) vs synthesize — the
            # healthy streaming path never spuriously evicts a coexisting
            # model (§3d / §4). No-op without a Metal cap or a pool. The
            # reservation is held across the whole stream and released when the
            # generator is exhausted / closed (client disconnect cancels it).
            async with contextlib.AsyncExitStack() as _reserve_stack:
                if self._pool is not None:
                    await _reserve_stack.enter_async_context(
                        self._pool.reserve_inference(
                            self._model_name,
                            self._estimate_streaming_working_set_bytes(),
                        )
                    )
                loop = asyncio.get_running_loop()
                while True:
                    chunk = await loop.run_in_executor(get_mlx_executor(), _next_pcm_chunk)
                    # Abort between PCM chunks — discards any in-flight chunk
                    # and surfaces RequestAbortedError to the streaming handler.
                    self._raise_if_aborted()
                    if chunk is sentinel:
                        break
                    if chunk is None:
                        continue
                    sample_rate, channels, sample_width, pcm_bytes = chunk
                    if not pcm_bytes:
                        continue
                    chunk_count += 1
                    total_bytes += len(pcm_bytes)
                    self._update_activity(
                        activity_id,
                        chunk_count=chunk_count,
                        output_bytes=total_bytes,
                    )
                    yield sample_rate, channels, sample_width, pcm_bytes
        finally:
            await self._finish_activity(activity_id)
            logger.info(
                "TTS native stream done: model=%s, %.2fs, chunks=%d, pcm_bytes=%d",
                self._model_name, time.monotonic() - t0, chunk_count, total_bytes,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "variant": self._variant,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata including variant and speakers."""
        if self._model is None:
            return {
                "loaded": False,
                "model_name": self._model_name,
            }
        return {
            "loaded": True,
            "model_name": self._model_name,
            "variant": self._variant,
            "speakers": self.get_speakers(),
        }

    def get_speakers(self) -> list[str]:
        """List available speakers (CustomVoice only)."""
        if self._model is None or self._variant != "custom_voice":
            return []
        try:
            talker_config = getattr(self._model.config, "talker_config", None)
            if talker_config and "spk_id" in talker_config:
                return list(talker_config["spk_id"].keys())
        except Exception:
            pass
        return []

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return (
            f"<TTSEngine model={self._model_name} "
            f"status={status} variant={self._variant}>"
        )
