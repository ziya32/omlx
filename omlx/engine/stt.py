# SPDX-License-Identifier: Apache-2.0
"""
STT (Speech-to-Text) engine for oMLX.

This module provides an engine for audio transcription using mlx-audio.
Unlike LLM engines, STT engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..mx_buffer_lock import locked_free_and_clear, locked_sync_and_clear_cache, run_locked
from ..exceptions import AudioError, InvalidAudioFormatError
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionOutput:
    """Output from STT transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: List[Dict[str, Any]] | None = None  # Raw segment dicts from mlx-audio


# Lowercase full-names work for both Qwen3-ASR (its _build_prompt lowercases
# the supported-language list before lookup) and Whisper (its TO_LANGUAGE_CODE
# normalizer maps lowercase names to ISO codes). Capitalized names would break
# Whisper because `<|Chinese|>` is not a valid language token.
_ISO_TO_STT_LANG: dict[str, str] = {
    "zh": "chinese",
    "yue": "cantonese",
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ko": "korean",
    "ja": "japanese",
}


def _normalize_stt_generate_language(language: str | None) -> str | None:
    """Map OpenAI-style ISO codes to language names accepted by mlx-audio backends."""
    if language is None:
        return None

    normalized = language.strip()
    if not normalized:
        return None

    return _ISO_TO_STT_LANG.get(normalized.lower(), normalized)


# ---------------------------------------------------------------------------
# Error helpers (#800): turn opaque mlx-audio/HF processor failures into
# actionable RuntimeErrors that tell users which file is missing and where
# to find a compatible variant.
# ---------------------------------------------------------------------------


_MISSING_PROCESSOR_HINTS = (
    "preprocessor_config.json",
    "feature extractor",
    "featureextractor",
)


def _looks_like_missing_processor(message: str) -> bool:
    """True if the error text from mlx-audio / HF points at a missing processor."""
    lowered = message.lower()
    return any(h in lowered for h in _MISSING_PROCESSOR_HINTS)


def _missing_processor_hint(model_name: str) -> str:
    return (
        f"STT model '{model_name}' is missing the HuggingFace processor / "
        "feature-extractor configuration (preprocessor_config.json and/or "
        "tokenizer files). MLX-converted repositories sometimes omit these. "
        "Fix: either use an HF-compatible variant of the model or copy "
        "preprocessor_config.json, tokenizer.json and special_tokens_map.json "
        "from the upstream HuggingFace repo into the local model directory."
    )


def _wrap_stt_load_error(model_name: str, exc: Exception) -> Exception:
    """Return a clearer exception for known mlx-audio STT load failures."""
    message = str(exc)
    if _looks_like_missing_processor(message):
        return RuntimeError(
            f"{_missing_processor_hint(model_name)} Original error: {message}"
        )
    return exc


def _validate_stt_processor(model_name: str, model: Any) -> None:
    """Fail fast if a Whisper-family mlx-audio model loaded without a processor."""
    module_name = type(model).__module__ or ""
    is_whisper_like = "whisper" in module_name.lower()
    if not is_whisper_like:
        return
    # mlx-audio Whisper attaches a HF processor to ``_processor``; it's set
    # to None when WhisperProcessor.from_pretrained() failed on load.
    if not hasattr(model, "_processor"):
        return
    if model._processor is not None:
        return
    raise RuntimeError(_missing_processor_hint(model_name))


class STTEngine(BaseNonStreamingEngine):
    """
    Engine for audio transcription (Speech-to-Text).

    This engine wraps mlx-audio STT models and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since transcription is computed in a single forward pass.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the STT engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        mlx-audio is imported here (lazily) to avoid module-level errors
        when the package is not installed.
        """
        if self._model is not None:
            return

        logger.info(f"Starting STT engine: {self._model_name}")

        try:
            from mlx_audio.stt.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for STT inference. "
                'Install it with: pip install "omlx[audio]"'
            ) from exc

        model_name = self._model_name

        def _load_sync():
            # load_model returns a single nn.Module, not a tuple
            return _load_model(model_name)

        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(get_mlx_executor(), lambda: run_locked(_load_sync))
        except Exception as exc:
            # #800: MLX-packaged repos (Qwen3-ASR-*-MLX-*, some mlx-community
            # whisper variants) often omit preprocessor_config.json, which
            # mlx-audio / HuggingFace AutoFeatureExtractor reports with an
            # opaque OSError. Re-raise with an actionable message instead.
            raise _wrap_stt_load_error(model_name, exc) from exc

        # #800: Whisper models in mlx-audio load silently without a
        # HuggingFace processor when preprocessor_config.json is missing
        # (mlx-audio only emits a warning). Fail fast at start so callers
        # see the real problem instead of a downstream "Processor not found"
        # 500 during transcribe.
        _validate_stt_processor(model_name, model)

        self._model = model
        logger.info(f"STT engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        # Latch the cooperative-abort flag BEFORE clearing _model so any
        # handler racing with stop sees RequestAbortedError (-> 503) on
        # its next _raise_if_aborted checkpoint rather than the plain
        # "Engine not started" RuntimeError (-> 500). Issue 4.
        self._mark_stopped()
        if self._model is None:
            return

        logger.info(f"Stopping STT engine: {self._model_name}")
        # Free the model ref + gc on the executor under the buffer lock, so the
        # eviction's buffer frees serialize with in-flight generation on the
        # executor instead of racing it from the event-loop thread (#85).
        holder = [self._model]
        self._model = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: locked_free_and_clear(holder.clear)
        )
        logger.info(f"STT engine stopped: {self._model_name}")

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        on_progress: Any | None = None,
        **kwargs,
    ) -> TranscriptionOutput:
        """
        Transcribe an audio file.

        For audio longer than 60 s, splits into chunks and transcribes each
        chunk on a separate executor turn so other engines (LLM, embedding)
        can interleave during long transcription. Invokes ``on_progress``
        after each chunk with ``chunk``, ``total_chunks``, and ``chunk_text``
        so the HTTP layer can emit SSE keepalive frames.

        Args:
            audio_path: Path to the audio file to transcribe.
            language: Optional language code (``en``/``fr``/...); ``auto`` /
                None triggers detection.
            prompt: Optional initial-prompt string (Whisper models).
            on_progress: Optional ``async`` callback invoked after each chunk
                with kwargs ``chunk``, ``total_chunks``, ``chunk_text``.
            **kwargs: Additional model-specific parameters passed to the
                single-call path (long-audio path uses fixed kwargs).

        Returns:
            TranscriptionOutput with transcribed text, language, duration,
            and segments.
        """
        # Cooperative-abort checkpoint BEFORE the model-None guard, so a
        # handler racing with enforcer eviction sees RequestAbortedError
        # (-> 503) rather than RuntimeError (-> 500).
        self._raise_if_aborted()
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        import os

        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info(
            "STT transcribe: model=%s, file=%s (%d bytes), language=%s",
            self._model_name, os.path.basename(audio_path), file_size, language,
        )

        model = self._model
        t0 = time.monotonic()

        activity_id = self._begin_activity(
            "transcribing",
            detail="Transcribing",
            metadata={"file_size_bytes": file_size},
        )
        try:
            result = await self._do_transcribe(
                model, audio_path, language, prompt, on_progress, kwargs,
            )
            elapsed = time.monotonic() - t0
            text_len = len(result.text)
            logger.info(
                "STT transcribe done: model=%s, %.2fs, %d chars output",
                self._model_name, elapsed, text_len,
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    # ------------------------------------------------------------------
    # Long-audio chunking and per-chunk progress events
    # ------------------------------------------------------------------

    async def _do_transcribe(
        self,
        model: Any,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        on_progress: Any | None,
        extra_kwargs: dict[str, Any],
    ) -> TranscriptionOutput:
        """Route between chunked path (long audio) and single-call path."""
        loop = asyncio.get_running_loop()
        executor = get_mlx_executor()

        # Step 1: try to load + split. For audio <= 60 s this returns a
        # single chunk and we fall through to the fast single-call path.
        # Models that don't expose split_audio_into_chunks (anything other
        # than Qwen3-ASR) also skip to the single-call path.
        def _load_and_split():
            import numpy as np
            from mlx_audio.stt.utils import load_audio
            from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
                split_audio_into_chunks,
            )

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
            audio_np = np.array(audio) if isinstance(audio, mx.array) else audio
            sr = getattr(model, "sample_rate", 16000)
            # 60 s chunks keep each executor hold under ~5 s so other
            # engines can interleave during multi-hour transcription.
            return split_audio_into_chunks(
                audio_np, sr=sr, chunk_duration=60.0
            )

        try:
            chunks = await loop.run_in_executor(executor, _load_and_split)
            self._raise_if_aborted()
        except (ImportError, AttributeError, TypeError):
            chunks = None

        if chunks is None or len(chunks) <= 1:
            return await self._transcribe_single(
                model, audio_path, language, prompt, extra_kwargs,
            )

        # Step 2: chunked transcription with per-chunk progress.
        logger.info(
            "[STT] Long audio split into %d chunks; yielding executor "
            "between chunks", len(chunks),
        )
        all_texts: list[str] = []
        all_segments: list[dict] = []
        detected_lang: str | None = None

        for chunk_audio, offset_sec in chunks:
            def _transcribe_chunk(audio_chunk=chunk_audio, offset=offset_sec):
                from mlx_audio.stt.generate import generate_transcription

                kw: dict[str, Any] = {}
                if language and language != "auto":
                    kw["language"] = language
                if prompt:
                    kw["initial_prompt"] = prompt
                try:
                    return generate_transcription(
                        model=model, audio=audio_chunk, **kw,
                    ), offset
                except Exception as e:
                    raise AudioError(
                        f"Transcription failed on chunk at {offset:.0f}s: {e}"
                    ) from e

            result, offset = await loop.run_in_executor(
                executor, _transcribe_chunk
            )
            self._raise_if_aborted()
            all_texts.append(result.text or "")

            if on_progress is not None:
                try:
                    await on_progress(
                        chunk=len(all_texts),
                        total_chunks=len(chunks),
                        chunk_text=result.text or "",
                    )
                except Exception as exc:
                    logger.warning("on_progress raised: %s", exc)

            raw_lang = getattr(result, "language", None)
            if isinstance(raw_lang, list):
                detected_lang = raw_lang[0] if raw_lang else detected_lang
            elif raw_lang and raw_lang != "None":
                detected_lang = raw_lang

            raw_segs = getattr(result, "segments", None) or []
            for seg in raw_segs:
                if isinstance(seg, dict):
                    adjusted = dict(seg)
                    adjusted["start"] = seg.get("start", 0.0) + offset
                    adjusted["end"] = seg.get("end", 0.0) + offset
                    all_segments.append(adjusted)

        sep = "" if detected_lang in ("zh", "ja", "ko", "yue") else " "
        full_text = sep.join(t for t in all_texts if t)
        duration = all_segments[-1].get("end") if all_segments else None

        return TranscriptionOutput(
            text=full_text,
            language=detected_lang or language,
            duration=duration,
            segments=all_segments if all_segments else None,
        )

    async def _transcribe_single(
        self,
        model: Any,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        extra_kwargs: dict[str, Any],
    ) -> TranscriptionOutput:
        """Short-audio / fallback path — single model.generate call."""

        def _normalize_segment(s) -> dict:
            if isinstance(s, dict):
                return s
            import dataclasses
            if dataclasses.is_dataclass(s) and not isinstance(s, type):
                return dataclasses.asdict(s)
            if hasattr(s, "__dict__"):
                return vars(s)
            return {"text": str(s)}

        def _normalize_language(raw_lang):
            if isinstance(raw_lang, list):
                raw_lang = raw_lang[0] if raw_lang else None
            if isinstance(raw_lang, str) and raw_lang.lower() == "none":
                return None
            return raw_lang

        def _transcribe_sync():
            gen_kwargs = dict(extra_kwargs)
            generate_language = _normalize_stt_generate_language(language)
            if generate_language is not None:
                gen_kwargs["language"] = generate_language
            if prompt:
                gen_kwargs.setdefault("initial_prompt", prompt)
            try:
                result = model.generate(audio_path, **gen_kwargs)
            except (InvalidAudioFormatError, AudioError):
                # Already typed — propagate as-is so the endpoint can
                # map to 400/422.
                raise
            except Exception as e:
                # Wrap raw mlx-audio errors as AudioError so the audio
                # endpoint surfaces 422 (retryable) instead of 500
                # (unhandled). Same pattern as TTS engine.synthesize().
                raise AudioError(
                    f"Transcription failed: {e}"
                ) from e

            if hasattr(result, "text"):
                raw_lang = _normalize_language(getattr(result, "language", None))
                if raw_lang is None:
                    raw_lang = language
                raw_segs = getattr(result, "segments", None)
                segments = [
                    _normalize_segment(s) for s in raw_segs
                ] if raw_segs else []
                return TranscriptionOutput(
                    text=result.text or "",
                    language=raw_lang,
                    segments=segments,
                    duration=getattr(result, "total_time", 0.0),
                )
            return TranscriptionOutput(
                text=str(result),
                language=language,
                segments=[],
                duration=0.0,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_mlx_executor(), _transcribe_sync)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def get_languages(self) -> list[str]:
        """List supported languages for this STT model."""
        if self._model is None:
            return []
        try:
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer:
                all_langs = getattr(tokenizer, "all_language_tokens", None)
                if isinstance(all_langs, dict):
                    return sorted(all_langs.keys())
                lang_map = getattr(tokenizer, "language_to_id", None)
                if isinstance(lang_map, dict):
                    return sorted(lang_map.keys())
        except Exception:
            pass
        return []

    def get_model_info(self) -> dict[str, Any]:
        """Get basic model metadata."""
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return {
            "loaded": True,
            "model_name": self._model_name,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<STTEngine model={self._model_name} status={status}>"
