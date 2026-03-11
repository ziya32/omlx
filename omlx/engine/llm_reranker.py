# SPDX-License-Identifier: Apache-2.0
"""
LLM-based reranker engine for oMLX.

This module provides an engine for document reranking using generative LLMs
(e.g., Qwen3-Reranker) via prefill_only logits mode. Unlike the classification-based
RerankerEngine, this wraps a BatchedEngine and scores relevance by computing
P("yes") / (P("yes") + P("no")) from the last-token log-probabilities.

Key advantages over standalone reranker:
- Continuous batching for concurrent requests
- Prefix caching (shared instruction prompt cached across documents)
- SSD cache support
- Shared EnginePool memory management
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

import mlx.core as mx

from ..models.reranker import RerankOutput
from ..request import SamplingParams
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class LLMRerankerEngine(BaseNonStreamingEngine):
    """LLM-based reranker using prefill_only logits mode.

    Wraps a BatchedEngine internally. Scores each query-document pair by:
    1. Formatting with the yes/no prompt template
    2. Running prefill_only with logits capture
    3. Computing P("yes") / (P("yes") + P("no")) as relevance score
    """

    PROMPT_TEMPLATE = (
        '<|im_start|>system\nJudge whether the Document is relevant to '
        'the Query. Answer only "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n<Query>{query}</Query>\n'
        '<Document>{document}</Document><|im_end|>\n'
        '<|im_start|>assistant\n'
    )

    def __init__(self, model_name: str, scheduler_config=None):
        self._model_name = model_name
        self._scheduler_config = scheduler_config
        self._batched = None  # BatchedEngine instance

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        """Get the tokenizer from the underlying BatchedEngine."""
        if self._batched is None:
            return None
        return self._batched.tokenizer

    async def start(self) -> None:
        if self._batched is not None:
            return

        logger.info(f"Starting LLM reranker engine: {self._model_name}")
        from .batched import BatchedEngine

        self._batched = BatchedEngine(
            model_name=self._model_name,
            scheduler_config=self._scheduler_config,
        )
        await self._batched.start()
        logger.info(f"LLM reranker engine started: {self._model_name}")

    async def stop(self) -> None:
        if self._batched is None:
            return

        logger.info(f"Stopping LLM reranker engine: {self._model_name}")
        await self._batched.stop()
        self._batched = None
        logger.info(f"LLM reranker engine stopped: {self._model_name}")

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> RerankOutput:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return (None = all)

        Returns:
            RerankOutput with scores, sorted indices, and token count
        """
        if self._batched is None:
            raise RuntimeError("Engine not started. Call start() first.")

        tokenizer = self._batched.tokenizer
        yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("no", add_special_tokens=False)[0]

        total_tokens = 0
        scores = []

        for doc in documents:
            prompt = self.PROMPT_TEMPLATE.format(query=query, document=doc)
            sampling = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                prefill_only=True,
                prefill_output="logits",
            )

            # Use the underlying EngineCore for direct prompt + sampling_params API
            output = await self._batched._engine.generate(
                prompt=prompt,
                sampling_params=sampling,
            )

            total_tokens += output.prompt_tokens + output.completion_tokens

            if output.last_logits is not None:
                logits = mx.array(output.last_logits)
                probs = mx.softmax(logits[mx.array([yes_id, no_id])])
                mx.eval(probs)
                scores.append(float(probs[0].item()))
            else:
                # Fallback: no logits captured, score as 0
                logger.warning(
                    f"No logits captured for document reranking, "
                    f"request_id={output.request_id}"
                )
                scores.append(0.0)

        # Sort by score descending
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in indexed]

        if top_n is not None and top_n < len(indices):
            indices = indices[:top_n]

        return RerankOutput(
            scores=scores,
            indices=indices,
            total_tokens=total_tokens,
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "model_name": self._model_name,
            "loaded": self._batched is not None,
            "engine_type": "llm_reranker",
        }
        if self._batched is not None:
            stats.update(self._batched.get_stats())
        return stats

    def get_model_info(self) -> Dict[str, Any]:
        if self._batched is None:
            return {"loaded": False, "model_name": self._model_name}
        return {
            "loaded": True,
            "model_name": self._model_name,
            "engine_type": "llm_reranker",
        }

    def __repr__(self) -> str:
        status = "running" if self._batched is not None else "stopped"
        return f"<LLMRerankerEngine model={self._model_name} status={status}>"
