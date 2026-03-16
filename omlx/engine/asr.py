# SPDX-License-Identifier: Apache-2.0
"""
ASR (Automatic Speech Recognition) engine for oMLX.

This module provides an engine for speech-to-text transcription using
mlx_audio.stt (Whisper, Qwen3-ASR, etc.). Unlike LLM engines, ASR engines
don't support streaming or chat completion.
"""

import asyncio
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionOutput:
    """Output from ASR transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None  # Raw segment dicts from mlx-audio


class ASREngine(BaseNonStreamingEngine):
    """
    Engine for speech-to-text transcription.

    This engine wraps mlx_audio.stt and provides async methods
    for integration with the oMLX server.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._active_operations: int = 0
        self._total_operations: int = 0
        self._total_audio_seconds: float = 0.0
        self._total_processing_seconds: float = 0.0

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Starting ASR engine: {self._model_name}")
        from mlx_audio import stt

        def _load_asr_sync():
            logger.debug(f"[ASR] Loading model on MLX executor thread: {self._model_name}")
            return stt.load(self._model_name)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_asr_sync)
        logger.info(f"ASR engine started: {self._model_name}")

    async def stop(self) -> None:
        if self._model is None:
            return

        logger.info(f"Stopping ASR engine: {self._model_name}")
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), mx.clear_cache)
        logger.info(f"ASR engine stopped: {self._model_name}")

    async def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        prompt: str | None = None,
    ) -> TranscriptionOutput:
        """
        Transcribe audio file to text.

        All MLX GPU operations run on the global MLX executor to prevent
        Metal command buffer races with concurrent engine inference.

        For language="auto", performs two-pass detection:
          1. Quick 30s prefix for language detection
          2. Full transcription with detected language

        Args:
            audio_path: Path to the audio file
            language: ISO language code or "auto" for detection
            prompt: Optional prompt to guide transcription (Whisper models)

        Returns:
            TranscriptionOutput with transcribed text, language, duration, and segments
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        model = self._model
        self._active_operations += 1
        start_time = time.perf_counter()

        try:
            return await self._do_transcribe(model, audio_path, language, prompt)
        finally:
            elapsed = time.perf_counter() - start_time
            self._active_operations -= 1
            self._total_operations += 1
            self._total_processing_seconds += elapsed

    async def _do_transcribe(
        self,
        model: Any,
        audio_path: str,
        language: str,
        prompt: str | None,
    ) -> TranscriptionOutput:
        def _transcribe_sync() -> TranscriptionOutput:
            from mlx_audio.stt.generate import generate_transcription

            logger.debug(f"[ASR] Transcribing on MLX executor thread: {audio_path}")

            kw: dict[str, Any] = {}
            if language and language != "auto":
                kw["language"] = language
            if prompt:
                kw["initial_prompt"] = prompt

            result = generate_transcription(
                model=model,
                audio=audio_path,
                **kw,
            )

            text = result.text or ""
            # result.language may be a list (e.g. ['en']) in newer mlx_audio versions
            raw_lang = result.language
            if isinstance(raw_lang, list):
                detected_lang = raw_lang[0] if raw_lang else None
            else:
                detected_lang = raw_lang
            # Normalize string "None" to actual None
            if detected_lang == "None":
                detected_lang = None
            duration = None
            segments = None
            if result.segments:
                last_seg = result.segments[-1]
                if isinstance(last_seg, dict):
                    duration = last_seg.get("end")
                # Preserve segment data for verbose_json responses
                segments = list(result.segments)

            return TranscriptionOutput(
                text=text,
                language=detected_lang,
                duration=duration,
                segments=segments,
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_mlx_executor(), _transcribe_sync)
        if result.duration:
            self._total_audio_seconds += result.duration
        return result

    def get_languages(self) -> list[str]:
        """List supported languages for this ASR model."""
        if self._model is None:
            return []
        try:
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer:
                # Whisper tokenizer pattern
                all_langs = getattr(tokenizer, "all_language_tokens", None)
                if isinstance(all_langs, dict):
                    return sorted(all_langs.keys())
                # Fallback: check for language_to_id mapping
                lang_map = getattr(tokenizer, "language_to_id", None)
                if isinstance(lang_map, dict):
                    return sorted(lang_map.keys())
        except Exception:
            pass
        return []

    @property
    def active_operations(self) -> int:
        return self._active_operations

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
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
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<ASREngine model={self._model_name} status={status}>"
