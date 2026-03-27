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
from ..exceptions import AudioError, InvalidAudioFormatError
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
        on_progress: Any | None = None,
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
            return await self._do_transcribe(model, audio_path, language, prompt, on_progress)
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
        on_progress: Any | None = None,
    ) -> TranscriptionOutput:
        loop = asyncio.get_running_loop()
        executor = get_mlx_executor()

        # Step 1: Load audio and split into chunks on executor.
        # For audio <= 60s this returns a single chunk and falls
        # through to the fast single-call path.
        def _load_and_split():
            import numpy as np
            from mlx_audio.stt.utils import load_audio
            from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
                split_audio_into_chunks,
            )

            logger.debug(f"[ASR] Loading audio: {audio_path}")
            try:
                audio = load_audio(audio_path)
            except (OSError, IOError) as e:
                raise InvalidAudioFormatError(
                    f"Failed to read audio file: {e}"
                ) from e
            except Exception as e:
                raise InvalidAudioFormatError(
                    f"Invalid audio format: {e}"
                ) from e
            audio_np = (
                np.array(audio) if isinstance(audio, mx.array) else audio
            )
            sr = getattr(model, "sample_rate", 16000)
            # Use 60s chunks so each executor hold is ~2-5s, allowing
            # other engines (LLM, embedding) to interleave.
            return split_audio_into_chunks(
                audio_np, sr=sr, chunk_duration=60.0
            )

        try:
            chunks = await loop.run_in_executor(executor, _load_and_split)
        except (ImportError, AttributeError, TypeError):
            # Model doesn't support split_audio_into_chunks (not Qwen3-ASR)
            # or has incompatible types (e.g. non-numeric sample_rate)
            # — fall back to single-call path
            chunks = None

        if chunks is None or len(chunks) == 1:
            # Short audio or unsupported model — single executor call
            return await self._transcribe_single(
                model, audio_path, language, prompt
            )

        # Step 2: Long audio — process each chunk separately, yielding
        # the executor between chunks so LLM token generation can
        # interleave during transcription of very long audio (20+ min).
        logger.info(
            f"[ASR] Long audio split into {len(chunks)} chunks, "
            "yielding executor between chunks"
        )
        all_texts: list[str] = []
        all_segments: list[dict] = []
        detected_lang: str | None = None

        for chunk_audio, offset_sec in chunks:

            def _transcribe_chunk(
                audio_chunk=chunk_audio, offset=offset_sec
            ):
                from mlx_audio.stt.generate import generate_transcription

                kw: dict[str, Any] = {}
                if language and language != "auto":
                    kw["language"] = language
                if prompt:
                    kw["initial_prompt"] = prompt

                try:
                    return generate_transcription(
                        model=model, audio=audio_chunk, **kw
                    ), offset
                except Exception as e:
                    raise AudioError(
                        f"Transcription failed on chunk at {offset:.0f}s: {e}"
                    ) from e

            result, offset = await loop.run_in_executor(
                executor, _transcribe_chunk
            )

            all_texts.append(result.text or "")

            if on_progress is not None:
                await on_progress(
                    chunk=len(all_texts),
                    total_chunks=len(chunks),
                    chunk_text=result.text or "",
                )

            raw_lang = result.language
            if isinstance(raw_lang, list):
                detected_lang = raw_lang[0] if raw_lang else detected_lang
            elif raw_lang and raw_lang != "None":
                detected_lang = raw_lang

            if result.segments:
                for seg in result.segments:
                    if isinstance(seg, dict):
                        adjusted = dict(seg)
                        adjusted["start"] = seg.get("start", 0.0) + offset
                        adjusted["end"] = seg.get("end", 0.0) + offset
                        all_segments.append(adjusted)

        # CJK languages don't use spaces between words
        sep = "" if detected_lang in ("zh", "ja", "ko", "yue") else " "
        full_text = sep.join(t for t in all_texts if t)
        duration = all_segments[-1].get("end") if all_segments else None

        output = TranscriptionOutput(
            text=full_text,
            language=detected_lang,
            duration=duration,
            segments=all_segments if all_segments else None,
        )
        if output.duration:
            self._total_audio_seconds += output.duration
        return output

    async def _transcribe_single(
        self,
        model: Any,
        audio_path: str,
        language: str,
        prompt: str | None,
    ) -> TranscriptionOutput:
        """Transcribe audio in a single executor call (short audio or fallback)."""

        def _transcribe_sync() -> TranscriptionOutput:
            from mlx_audio.stt.generate import generate_transcription

            logger.debug(f"[ASR] Transcribing on MLX executor thread: {audio_path}")

            kw: dict[str, Any] = {}
            if language and language != "auto":
                kw["language"] = language
            if prompt:
                kw["initial_prompt"] = prompt

            try:
                result = generate_transcription(
                    model=model, audio=audio_path, **kw
                )
            except (OSError, IOError) as e:
                raise InvalidAudioFormatError(
                    f"Failed to read audio file: {e}"
                ) from e
            except Exception as e:
                err_msg = str(e).lower()
                if "audio" in err_msg or "wav" in err_msg or "format" in err_msg:
                    raise InvalidAudioFormatError(
                        f"Invalid audio format: {e}"
                    ) from e
                raise AudioError(
                    f"Transcription failed: {e}"
                ) from e

            text = result.text or ""
            raw_lang = result.language
            if isinstance(raw_lang, list):
                detected_lang = raw_lang[0] if raw_lang else None
            else:
                detected_lang = raw_lang
            if detected_lang == "None":
                detected_lang = None
            duration = None
            segments = None
            if result.segments:
                last_seg = result.segments[-1]
                if isinstance(last_seg, dict):
                    duration = last_seg.get("end")
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
