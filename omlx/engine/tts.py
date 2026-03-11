# SPDX-License-Identifier: Apache-2.0
"""
TTS (Text-to-Speech) engine for oMLX.

This module provides an engine for speech synthesis using Qwen3-TTS models
via mlx-audio (≥ 0.4.0). Supports two model variants:

- CustomVoice: 9 preset speakers with emotional/tonal control via instruct
- VoiceDesign: arbitrary voice synthesis from natural-language descriptions

Output: 24 kHz WAV audio.
"""

import gc
import io
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class SpeechOutput:
    """Output from TTS synthesis."""

    audio_bytes: bytes  # WAV file bytes
    sample_rate: int = 24000
    duration: float = 0.0  # seconds


def _audio_to_wav(audio: mx.array, sample_rate: int) -> bytes:
    """Convert an mx.array of float audio samples to WAV bytes (16-bit PCM)."""
    import numpy as np

    samples = np.array(audio, dtype=np.float32).flatten()
    # Clamp to [-1, 1] and convert to int16
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)

    buf = io.BytesIO()
    num_channels = 1
    sample_width = 2  # 16-bit
    data_size = len(pcm) * sample_width

    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())

    return buf.getvalue()


class TTSEngine(BaseNonStreamingEngine):
    """
    Engine for text-to-speech synthesis.

    Supports Qwen3-TTS models via mlx-audio ≥ 0.4.0:
    - CustomVoice: preset speakers + emotional instruct
    - VoiceDesign: free-form voice description

    The variant is auto-detected from the model's config.json
    (tts_model_type field).
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._variant: str = "custom_voice"  # "custom_voice" | "voice_design"

    @property
    def model_name(self) -> str:
        return self._model_name

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
        if self._model is not None:
            return

        logger.info(f"Starting TTS engine: {self._model_name}")
        from mlx_audio.tts.utils import load_model

        self._model = load_model(Path(self._model_name))
        self._variant = self._detect_variant(self._model_name)
        logger.info(
            f"TTS engine started: {self._model_name} (variant={self._variant})"
        )

    async def stop(self) -> None:
        if self._model is None:
            return

        logger.info(f"Stopping TTS engine: {self._model_name}")
        self._model = None
        gc.collect()
        mx.clear_cache()
        logger.info(f"TTS engine stopped: {self._model_name}")

    async def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        instruct: str | None = None,
    ) -> SpeechOutput:
        """
        Generate speech audio from text.

        Args:
            text: The text to synthesize
            speaker: For CustomVoice: preset speaker name (e.g., 'ryan', 'vivian').
                     Ignored for VoiceDesign.
            instruct: For CustomVoice: emotional/tonal instruction
                      (e.g., 'Speak warmly').
                      For VoiceDesign: voice description
                      (e.g., 'A warm female narrator with gentle pace').

        Returns:
            SpeechOutput with WAV audio bytes
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        if self._variant == "voice_design":
            results = list(self._model.generate(
                text=text,
                instruct=instruct or "A neutral narrator",
            ))
        else:
            # custom_voice or base
            kwargs = {"text": text, "voice": speaker or "vivian"}
            if instruct:
                kwargs["instruct"] = instruct
            results = list(self._model.generate(**kwargs))

        # Concatenate audio segments
        audio = mx.concatenate([r.audio for r in results])
        mx.eval(audio)

        sample_rate = getattr(self._model, "sample_rate", 24000)
        duration = audio.shape[0] / sample_rate

        wav_bytes = _audio_to_wav(audio, sample_rate)

        return SpeechOutput(
            audio_bytes=wav_bytes,
            sample_rate=sample_rate,
            duration=duration,
        )

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
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "variant": self._variant,
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
