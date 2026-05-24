# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine for oMLX.

This module provides an engine for generating text embeddings using
mlx-embeddings. Unlike LLM engines, embedding engines don't support
streaming or chat completion.
"""

import asyncio
import gc
import logging
from typing import Any, Dict, List, Optional, Union

from ..engine_core import get_mlx_executor
from ..mx_buffer_lock import locked_sync_and_clear_cache, run_locked
from ..models.embedding import EmbeddingOutput, MLXEmbeddingModel
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class EmbeddingEngine(BaseNonStreamingEngine):
    """
    Engine for generating text embeddings.

    This engine wraps MLXEmbeddingModel and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since embeddings are computed in a single forward pass.
    """

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        """
        Initialize the embedding engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Allow loaders to execute custom Python shipped
                with the model repo. Off by default for security (issue #926).
        """
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model: Optional[MLXEmbeddingModel] = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def processor(self) -> Any:
        """Get the processor/tokenizer."""
        return self._model.processor if self._model else None

    @property
    def hidden_size(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._model.hidden_size if self._model else None

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        """
        if self._model is not None:
            return

        logger.info(f"Starting embedding engine: {self._model_name}")
        self._model = MLXEmbeddingModel(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        # Hold the buffer-access lock across the load: it allocates the model's
        # weights, which makes MLX reclaim cached buffers, racing the off-thread
        # phase-2 KV-cache save (_extract_tensor_bytes) that may be in flight
        # from a just-finished generation -> GPU command-buffer abort. (#1106)
        await loop.run_in_executor(
            get_mlx_executor(), lambda: run_locked(self._model.load)
        )
        logger.info(f"Embedding engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping embedding engine: {self._model_name}")
        # Mark terminal BEFORE clearing the model ref so any handler
        # racing with stop() sees RequestAbortedError via
        # _raise_if_aborted instead of a RuntimeError from the model
        # guard. See docs/enforcer-eviction-review.md #4.
        self._mark_stopped()
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), locked_sync_and_clear_cache
        )
        logger.info(f"Embedding engine stopped: {self._model_name}")

    async def embed(
        self,
        texts: Union[List[str], List[Dict[str, str]]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        instruction: str | None = None,
    ) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum token length for each text
            padding: Whether to pad shorter sequences
            truncation: Whether to truncate longer sequences
            instruction: Task instruction for instruction-aware models
                (e.g. Qwen3-Embedding). When provided, inputs are formatted
                as 'Instruct: {instruction}\\nQuery:{text}'. Use for queries
                only — documents should be embedded without instruction.

        Returns:
            EmbeddingOutput with embeddings and token count
        """
        # Check abort FIRST so a handler racing with stop() sees the
        # typed RequestAbortedError (→ HTTP 503) rather than the plain
        # RuntimeError from the model guard (→ HTTP 500).
        self._raise_if_aborted()
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        # Apply instruction prefix for instruction-aware embedding models
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]

        model = self._model

        def _embed_sync():
            return model.embed(
                inputs=texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )

        activity_id = self._begin_activity(
            "embedding",
            detail="Embedding",
            total_items=len(texts),
            metadata={"input_count": len(texts)},
        )
        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(get_mlx_executor(), _embed_sync)
            # Discard result if the enforcer aborted us while the MLX
            # kernel was running on the executor thread.
            self._raise_if_aborted()
            self._update_activity(
                activity_id,
                token_count=output.total_tokens,
                dimensions=output.dimensions,
            )
            return output
        finally:
            if self._end_activity(activity_id):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    locked_sync_and_clear_cache,
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "hidden_size": self.hidden_size,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return self._model.get_model_info()

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<EmbeddingEngine model={self._model_name} status={status}>"
