# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for oMLX inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- VLMBatchedEngine: Vision-language model engine with image support
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models
- LLMRerankerEngine: LLM-based reranking using prefill_only logits mode
- STTEngine: Speech-to-text transcription using mlx-audio
- TTSEngine: Text-to-speech synthesis using Qwen3-TTS via mlx-audio

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
from .base import BaseEngine, BaseNonStreamingEngine, GenerationOutput
from .batched import BatchedEngine
from .embedding import EmbeddingEngine
from .llm_reranker import LLMRerankerEngine
from .reranker import RerankerEngine
from .stt import STTEngine
from .sts import STSEngine
from .tts import TTSEngine
from .vlm import VLMBatchedEngine

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    "LLMRerankerEngine",
    "STTEngine",
    "STSEngine",
    "TTSEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
