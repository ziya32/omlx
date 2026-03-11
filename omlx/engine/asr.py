# SPDX-License-Identifier: Apache-2.0
"""
ASR (Automatic Speech Recognition) engine for oMLX.

This module provides an engine for speech-to-text transcription using
mlx_audio.stt (Whisper, Qwen3-ASR, etc.). Unlike LLM engines, ASR engines
don't support streaming or chat completion.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict

import mlx.core as mx

from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionOutput:
    """Output from ASR transcription."""

    text: str
    language: str | None = None
    duration: float | None = None


class ASREngine(BaseNonStreamingEngine):
    """
    Engine for speech-to-text transcription.

    This engine wraps mlx_audio.stt and provides async methods
    for integration with the oMLX server.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Starting ASR engine: {self._model_name}")
        from mlx_audio import stt

        self._model = stt.load(self._model_name)
        logger.info(f"ASR engine started: {self._model_name}")

    async def stop(self) -> None:
        if self._model is None:
            return

        logger.info(f"Stopping ASR engine: {self._model_name}")
        self._model = None
        gc.collect()
        mx.clear_cache()
        logger.info(f"ASR engine stopped: {self._model_name}")

    async def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
    ) -> TranscriptionOutput:
        """
        Transcribe audio file to text.

        For language="auto", performs two-pass detection:
          1. Quick 30s prefix for language detection
          2. Full transcription with detected language

        Args:
            audio_path: Path to the audio file
            language: ISO language code or "auto" for detection

        Returns:
            TranscriptionOutput with transcribed text, language, and duration
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        from mlx_audio.stt.generate import generate_transcription

        kwargs = {}
        if language and language != "auto":
            kwargs["language"] = language

        result = generate_transcription(
            model=self._model,
            audio=audio_path,
            **kwargs,
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
        if result.segments:
            last_seg = result.segments[-1]
            if isinstance(last_seg, dict):
                duration = last_seg.get("end")

        return TranscriptionOutput(
            text=text,
            language=detected_lang,
            duration=duration,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
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
