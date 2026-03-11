# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible Audio API.

These models define the request and response schemas for:
- POST /v1/audio/transcriptions (ASR)
- POST /v1/audio/speech (TTS)
- GET  /v1/audio/speakers (speaker list)
"""

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Response from audio transcription (ASR)."""

    text: str
    """The transcribed text."""

    language: str | None = None
    """Detected or specified language code (e.g., 'en', 'zh')."""

    duration: float | None = None
    """Duration of the audio in seconds."""


class SpeechRequest(BaseModel):
    """
    Request for text-to-speech synthesis.

    OpenAI-compatible request format for the /v1/audio/speech endpoint.
    """

    model: str
    """ID of the TTS model to use."""

    input: str
    """The text to synthesize into speech."""

    voice: str = "default"
    """
    Voice to use for synthesis.
    For CustomVoice models: one of the preset speakers (e.g., 'ryan', 'vivian').
    For VoiceDesign models: ignored (use instructions for voice description).
    """

    instructions: str | None = None
    """
    Optional instructions for voice control.
    For CustomVoice models: emotional/tonal control (e.g., 'Speak warmly').
    For VoiceDesign models: voice description (e.g., 'A warm female narrator').
    """

    response_format: str = "wav"
    """Audio output format. Currently only 'wav' is supported."""

    speed: float = 1.0
    """Speaking speed (reserved for future use, currently ignored)."""


class SpeakersResponse(BaseModel):
    """Response listing available TTS speakers."""

    speakers: list[str] = Field(default_factory=list)
    """List of available speaker names."""

    languages: list[str] = Field(default_factory=list)
    """List of supported languages."""
