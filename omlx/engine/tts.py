# SPDX-License-Identifier: Apache-2.0
"""
TTS (Text-to-Speech) engine for oMLX.

This module provides an engine for speech synthesis using mlx-audio.
Unlike LLM engines, TTS engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.

Supports chunked yielding: each audio segment is generated in a separate
executor submission so LLM token generation can interleave between segments.
"""

import asyncio
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from .audio_utils import DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE
from .audio_utils import audio_to_wav_bytes as _audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class SpeechOutput:
    """Output from TTS synthesis."""

    audio_bytes: bytes  # WAV file bytes
    sample_rate: int = 24000
    duration: float = 0.0  # seconds



def _audio_to_pcm(audio_array) -> bytes:
    """Convert audio samples to raw 16-bit PCM bytes (for streaming segments)."""
    samples = np.array(audio_array, dtype=np.float32).flatten()
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767).astype(np.int16).tobytes()



class TTSEngine(BaseNonStreamingEngine):
    """
    Engine for speech synthesis (Text-to-Speech).

    This engine wraps mlx-audio TTS models and provides async methods
    for integration with the oMLX server.

    Supports multiple model variants (auto-detected from config.json):
    - CustomVoice: preset speakers with emotional/tonal control via instruct
    - VoiceDesign: arbitrary voice synthesis from natural-language descriptions
    - Base: reference audio voice cloning

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
        self._variant: str = "custom_voice"  # "custom_voice" | "voice_design" | "base"
        self._active_operations: int = 0
        self._total_operations: int = 0
        self._total_audio_seconds: float = 0.0
        self._total_processing_seconds: float = 0.0

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def active_operations(self) -> int:
        return self._active_operations

    def _detect_variant(self, model_path: str) -> str:
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
                "Install it with: pip install mlx-audio"
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
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        self._variant = self._detect_variant(self._model_name)
        logger.info(
            f"TTS engine started: {self._model_name} (variant={self._variant})"
        )

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping TTS engine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"TTS engine stopped: {self._model_name}")

    def _build_generate_kwargs(
        self,
        text: str,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        **extra,
    ) -> dict:
        """Build kwargs for model.generate() based on variant and model signature.

        Uses inspect.signature to route voice/instruct to the correct parameter
        name, ensuring compatibility across Qwen3-TTS, Kokoro, VibeVoice, etc.
        """
        import inspect

        model = self._model
        gen_params = inspect.signature(model.generate).parameters

        # Base variant with reference audio (voice cloning)
        if self._variant == "base" and ref_audio:
            kw: dict = {"text": text, "ref_audio": ref_audio}
            if ref_text:
                kw["ref_text"] = ref_text
            return kw

        # VoiceDesign variant
        if self._variant == "voice_design":
            return {"text": text, "instruct": instructions or "A neutral narrator"}

        # Default: CustomVoice / generic — route via model signature
        gen_kwargs: Dict[str, Any] = {"text": text, "verbose": False}

        if voice is not None:
            if "voice" in gen_params:
                gen_kwargs["voice"] = voice
            elif "instruct" in gen_params:
                gen_kwargs["instruct"] = voice
        if instructions is not None and "instruct" in gen_params:
            gen_kwargs["instruct"] = instructions
        if speed != 1.0:
            gen_kwargs["speed"] = speed
        gen_kwargs.update(extra)

        return gen_kwargs

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ) -> SpeechOutput:
        """
        Synthesize speech from text.

        Uses chunked yielding: each audio segment is generated in a separate
        executor submission so LLM token generation can interleave between
        segments instead of being blocked for the entire synthesis.

        Args:
            text: Input text to synthesize
            voice: Optional voice/speaker identifier
            speed: Speech speed multiplier (1.0 = normal)
            instructions: Optional voice description for instruct-capable models
            ref_audio: For Base variant: path to reference audio for voice cloning
            ref_text: For Base variant: transcript of the reference audio
            **kwargs: Additional model-specific parameters

        Returns:
            SpeechOutput with WAV audio bytes, sample_rate, and duration
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        logger.info(
            "TTS synthesize: model=%s, text_len=%d, voice=%s, speed=%.1f",
            self._model_name, len(text), voice, speed,
        )

        model = self._model
        gen_kwargs = self._build_generate_kwargs(
            text, voice, instructions, ref_audio, ref_text, speed, **kwargs
        )
        loop = asyncio.get_running_loop()
        executor = get_mlx_executor()
        self._active_operations += 1
        start_time = time.perf_counter()

        try:
            # Step 1: Create generator on executor (quick)
            def _create_gen():
                logger.debug(f"[TTS] Synthesizing: variant={self._variant}")
                return model.generate(**gen_kwargs)

            gen = await loop.run_in_executor(executor, _create_gen)

            # Step 2: Consume one segment at a time, yielding executor between
            # each so LLM scheduler.step() calls can interleave
            segments = []

            def _next_seg(g):
                return next(g, None)

            while True:
                seg = await loop.run_in_executor(executor, _next_seg, gen)
                if seg is None:
                    break
                segments.append(seg)

            if not segments:
                raise RuntimeError("TTS model produced no audio output")

            # Step 3: Concatenate and encode WAV (quick)
            def _finalize(segs):
                audio_chunks = []
                sample_rate = _DEFAULT_SAMPLE_RATE
                for s in segs:
                    audio_chunks.append(np.array(s.audio))
                    if hasattr(s, "sample_rate"):
                        sample_rate = s.sample_rate
                audio = np.concatenate(audio_chunks, axis=0)
                duration = len(audio) / sample_rate
                return SpeechOutput(
                    audio_bytes=_audio_to_wav_bytes(audio, int(sample_rate)),
                    sample_rate=int(sample_rate),
                    duration=duration,
                )

        with self._active_lock:
            self._active_count += 1
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                get_mlx_executor(), _synthesize_sync
            )

            elapsed = time.monotonic() - t0
            logger.info(
                "TTS synthesize done: model=%s, %.2fs, %d bytes output",
                self._model_name, elapsed, len(result),
            )
            return result
        finally:
            with self._active_lock:
                self._active_count -= 1

    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[SpeechOutput]:
        """
        Generate speech audio from text, yielding each segment as it's produced.

        Each segment is independently eval'd and converted to WAV/PCM so the
        client can begin playback before the full synthesis completes.

        Args:
            Same as synthesize().

        Yields:
            SpeechOutput for each audio segment
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        model = self._model
        gen_kwargs = self._build_generate_kwargs(
            text, voice, instructions, ref_audio, ref_text, speed, **kwargs
        )
        loop = asyncio.get_running_loop()
        executor = get_mlx_executor()
        with self._active_lock:
            self._active_count += 1
        start_time = time.perf_counter()
        total_duration = 0.0

        try:
            gen = await loop.run_in_executor(
                executor, lambda: model.generate(**gen_kwargs)
            )

            def _next_seg(g):
                return next(g, None)

            def _segment_to_pcm(seg):
                audio = np.array(seg.audio)
                sr = getattr(seg, "sample_rate", _DEFAULT_SAMPLE_RATE)
                dur = len(audio) / sr
                return SpeechOutput(
                    audio_bytes=_audio_to_pcm(audio),
                    sample_rate=int(sr),
                    duration=dur,
                )

            while True:
                seg = await loop.run_in_executor(executor, _next_seg, gen)
                if seg is None:
                    break
                chunk = await loop.run_in_executor(executor, _segment_to_pcm, seg)
                total_duration += chunk.duration
                yield chunk
        finally:
            with self._active_lock:
                self._active_count -= 1

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

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "variant": self._variant,
            "active_operations": self._active_operations,
            "total_operations": self._total_operations,
            "total_audio_seconds": round(self._total_audio_seconds, 2),
            "total_processing_seconds": round(self._total_processing_seconds, 2),
        }

    def get_model_info(self) -> Dict[str, Any]:
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return {
            "loaded": True,
            "model_name": self._model_name,
            "variant": self._variant,
            "speakers": self.get_speakers(),
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<TTSEngine model={self._model_name} variant={self._variant} status={status}>"
