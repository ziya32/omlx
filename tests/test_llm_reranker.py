# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.engine.llm_reranker module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from omlx.engine.llm_reranker import LLMRerankerEngine
from omlx.models.reranker import RerankOutput
from omlx.request import RequestOutput, SamplingParams


class TestLLMRerankerEngine:
    """Test cases for LLMRerankerEngine."""

    def test_init(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        assert engine.model_name == "/path/to/model"
        assert engine._batched is None

    def test_tokenizer_before_start(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        assert engine.tokenizer is None

    def test_repr_stopped(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        assert "stopped" in repr(engine)

    def test_repr_running(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        engine._batched = MagicMock()
        assert "running" in repr(engine)

    def test_get_stats_not_loaded(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        stats = engine.get_stats()
        assert stats["loaded"] is False
        assert stats["engine_type"] == "llm_reranker"

    def test_get_model_info_not_loaded(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        info = engine.get_model_info()
        assert info["loaded"] is False

    def test_prompt_template_format(self):
        prompt = LLMRerankerEngine.PROMPT_TEMPLATE.format(
            query="What is AI?",
            document="Artificial intelligence is a branch of computer science."
        )
        assert "<Query>What is AI?</Query>" in prompt
        assert "Artificial intelligence is a branch of computer science." in prompt
        assert '<|im_start|>system' in prompt
        assert '<|im_start|>assistant' in prompt
        assert 'Answer only "yes" or "no"' in prompt

    async def test_rerank_not_started(self):
        engine = LLMRerankerEngine(model_name="/path/to/model")
        with pytest.raises(RuntimeError, match="Engine not started"):
            await engine.rerank(query="test", documents=["doc1"])


def _make_engine_with_mock(logits_responses):
    """Create an LLMRerankerEngine with mocked internals.

    Args:
        logits_responses: List of last_logits values, one per document.
    """
    engine = LLMRerankerEngine(model_name="/path/to/model")

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: {
        "yes": [1234],
        "no": [5678],
    }[text]

    # Mock BatchedEngine
    mock_batched = MagicMock()
    mock_batched.tokenizer = mock_tokenizer

    # Mock EngineCore.generate
    mock_engine_core = AsyncMock()
    responses = []
    for logits in logits_responses:
        output = RequestOutput(
            request_id="mock",
            prompt_tokens=10,
            completion_tokens=1,
            finished=True,
            finish_reason="length",
            last_logits=logits,
        )
        responses.append(output)
    mock_engine_core.generate.side_effect = responses
    mock_batched._engine = mock_engine_core

    engine._batched = mock_batched
    return engine


class TestLLMRerankerScoring:
    """Test LLM reranker scoring logic with mocked BatchedEngine."""

    async def test_rerank_scores_ordering(self):
        """Test that documents are ranked by P(yes) score."""
        vocab_size = 10000
        logits_doc0 = [0.0] * vocab_size
        logits_doc0[1234] = -1.0  # low yes score
        logits_doc0[5678] = 1.0   # high no score

        logits_doc1 = [0.0] * vocab_size
        logits_doc1[1234] = 2.0   # high yes score
        logits_doc1[5678] = -2.0  # low no score

        engine = _make_engine_with_mock([logits_doc0, logits_doc1])
        output = await engine.rerank(
            query="What is AI?",
            documents=["irrelevant doc", "very relevant doc"],
        )

        assert isinstance(output, RerankOutput)
        assert len(output.scores) == 2
        assert output.scores[1] > output.scores[0]
        assert output.indices[0] == 1
        assert output.indices[1] == 0

    async def test_rerank_top_n(self):
        """Test top_n filtering."""
        vocab_size = 10000
        logits_list = []
        for i in range(5):
            logits = [0.0] * vocab_size
            logits[1234] = float(i)   # Increasing yes score
            logits[5678] = 0.0
            logits_list.append(logits)

        engine = _make_engine_with_mock(logits_list)
        output = await engine.rerank(
            query="test",
            documents=[f"doc {i}" for i in range(5)],
            top_n=2,
        )

        assert len(output.indices) == 2
        assert output.indices[0] == 4
        assert output.indices[1] == 3

    async def test_rerank_token_count(self):
        """Test that total_tokens is accumulated correctly."""
        vocab_size = 10000
        logits = [0.0] * vocab_size
        logits[1234] = 1.0

        engine = _make_engine_with_mock([logits, logits, logits])
        output = await engine.rerank(
            query="test",
            documents=["a", "b", "c"],
        )

        # 3 documents * (10 prompt + 1 completion) = 33
        assert output.total_tokens == 33

    async def test_rerank_no_logits_fallback(self):
        """Test that missing logits produces score 0."""
        engine = _make_engine_with_mock([None])
        output = await engine.rerank(
            query="test",
            documents=["doc"],
        )

        assert output.scores == [0.0]

    async def test_rerank_sampling_params(self):
        """Test that correct sampling params are sent."""
        vocab_size = 100
        logits = [0.0] * vocab_size

        engine = _make_engine_with_mock([logits])
        await engine.rerank(query="test", documents=["doc"])

        # Verify the generate call
        call_args = engine._batched._engine.generate.call_args
        sampling = call_args.kwargs.get("sampling_params") or call_args[1].get("sampling_params")
        assert sampling.max_tokens == 1
        assert sampling.temperature == 0.0
        assert sampling.prefill_only is True
        assert sampling.prefill_output == "logits"


class TestLLMRerankerModelDiscovery:
    """Test that llm_reranker type is properly handled in model discovery."""

    def test_llm_reranker_in_model_type(self):
        from omlx.model_discovery import ModelType
        assert "llm_reranker" in ModelType.__args__

    def test_llm_reranker_in_engine_type(self):
        from omlx.model_discovery import EngineType
        assert "llm_reranker" in EngineType.__args__

    def test_engine_pool_model_type_mapping(self):
        from omlx.engine_pool import EnginePool
        pool = EnginePool(max_model_memory=None)
        assert pool._MODEL_TYPE_TO_ENGINE["llm_reranker"] == "llm_reranker"
