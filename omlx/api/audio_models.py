# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible audio API.

These models define the request and response schemas for:
- Audio transcription (speech-to-text)
- Audio speech synthesis (text-to-speech)
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio with timestamps."""

    id: int
    start: float
    end: float
    text: str


class AudioTranscriptionRequest(BaseModel):
    """OpenAI-compatible audio transcription request."""

    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0


class AudioTranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[dict]] = None


class VerboseTranscriptionResponse(BaseModel):
    """Verbose response from audio transcription with segment timestamps."""

    task: str = "transcribe"
    language: str | None = None
    duration: float | None = None
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    instructions: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    speed: Optional[float] = 1.0
    response_format: Optional[str] = "wav"


class AudioProcessRequest(BaseModel):
    """Request model for audio processing (speech enhancement / STS).

    Used by POST /v1/audio/process — the audio file is submitted as a
    multipart upload alongside this model field.
    """

    model: str


class SpeakersResponse(BaseModel):
    """Response listing available TTS speakers."""

    speakers: list[str] = Field(default_factory=list)
    """List of available speaker names."""

    languages: list[str] = Field(default_factory=list)
    """List of supported languages."""


class LanguagesResponse(BaseModel):
    """Response listing supported ASR languages."""

    languages: list[str] = Field(default_factory=list)
    """List of supported ISO language codes."""

    model: str | None = None
    """The ASR model these languages are from."""
