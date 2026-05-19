# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.server module - sampling parameter resolution and exception handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from omlx.exceptions import ModelNotFoundError
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import EngineType, SamplingDefaults, ServerState, app, get_engine, get_sampling_params
from omlx.settings import GlobalSettings


class TestGetSamplingParams:
    """Tests for get_sampling_params function."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_returns_10_tuple(self):
        """Test that get_sampling_params returns a 10-tuple."""
        result = get_sampling_params(None, None)
        assert isinstance(result, tuple)
        assert len(result) == 10

    def test_defaults(self):
        """Test default values with no request or model params."""
        temp, top_p, top_k, rep_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_prob, xtc_thresh = get_sampling_params(None, None)
        assert temp == 1.0
        assert top_p == 0.95
        assert top_k == 0
        assert rep_penalty == 1.0
        assert min_p == 0.0
        assert presence_penalty == 0.0
        assert frequency_penalty == 0.0
        assert max_tokens == 32768

    def test_request_overrides(self):
        """Test request params override global defaults."""
        temp, top_p, top_k, rep_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_prob, xtc_thresh = get_sampling_params(
            0.5, 0.8, req_min_p=0.1, req_presence_penalty=0.5, req_frequency_penalty=0.3,
            req_max_tokens=1024,
        )
        assert temp == 0.5
        assert top_p == 0.8
        assert top_k == 0  # not overridable via request
        assert rep_penalty == 1.0
        assert min_p == 0.1
        assert presence_penalty == 0.5
        assert frequency_penalty == 0.3
        assert max_tokens == 1024

    def test_xtc_defaults_when_none(self):
        """Test XTC params default when not specified."""
        *_, xtc_prob, xtc_thresh = get_sampling_params(None, None)
        assert xtc_prob == 0.0
        assert xtc_thresh == 0.1

    def test_xtc_request_passthrough(self):
        """Test XTC params pass through from request values."""
        *_, xtc_prob, xtc_thresh = get_sampling_params(
            None, None, req_xtc_probability=0.5, req_xtc_threshold=0.1,
        )
        assert xtc_prob == 0.5
        assert xtc_thresh == 0.1

    def test_xtc_partial_override(self):
        """Test setting only xtc_probability uses safe default threshold."""
        *_, xtc_prob, xtc_thresh = get_sampling_params(
            None, None, req_xtc_probability=0.3,
        )
        assert xtc_prob == 0.3
        assert xtc_thresh == 0.1

    def test_model_settings_override(self):
        """Test model settings override global defaults."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(
                temperature=0.3, top_k=50, repetition_penalty=1.2,
                min_p=0.05, presence_penalty=0.3, max_tokens=2048,
            )
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            temp, top_p, top_k, rep_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_prob, xtc_thresh = get_sampling_params(
                None, None, "test-model"
            )
            assert temp == 0.3
            assert top_p == 0.95  # falls back to global
            assert top_k == 50
            assert rep_penalty == 1.2
            assert min_p == 0.05
            assert presence_penalty == 0.3
            assert frequency_penalty == 0.0
            assert max_tokens == 2048

    def test_request_over_model(self):
        """Test request params take priority over model settings."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(temperature=0.3, min_p=0.05, max_tokens=2048)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            temp, top_p, top_k, rep_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_prob, xtc_thresh = get_sampling_params(
                0.7, None, "test-model", req_min_p=0.1, req_max_tokens=4096,
            )
            assert temp == 0.7  # request wins
            assert min_p == 0.1  # request wins over model
            assert max_tokens == 4096  # request wins over model

    def test_model_repetition_penalty(self):
        """Test model-level repetition_penalty overrides global."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(repetition_penalty=1.5)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            _, _, _, rep_penalty, _, _, _, _, _, _ = get_sampling_params(None, None, "test-model")
            assert rep_penalty == 1.5

    def test_global_repetition_penalty(self):
        """Test global repetition_penalty is used when no model override."""
        self._state.sampling = SamplingDefaults(repetition_penalty=1.3)

        _, _, _, rep_penalty, _, _, _, _, _, _ = get_sampling_params(None, None)
        assert rep_penalty == 1.3

    def test_force_sampling(self):
        """Test force_sampling ignores request params."""
        self._state.sampling = SamplingDefaults(
            temperature=0.5, top_p=0.8, max_tokens=4096, force_sampling=True
        )

        temp, top_p, _, _, _, _, _, max_tokens, _, _ = get_sampling_params(
            0.9, 0.99, req_max_tokens=8192
        )
        assert temp == 0.5  # forced, not request
        assert top_p == 0.8  # forced, not request
        assert max_tokens == 4096  # forced, not request

    def test_force_sampling_model_max_tokens(self):
        """Test force_sampling with model-level max_tokens overrides global."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(max_tokens=8192, force_sampling=True)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            _, _, _, _, _, _, _, max_tokens, _, _ = get_sampling_params(
                None, None, "test-model", req_max_tokens=1024
            )
            assert max_tokens == 8192  # model setting wins in force mode

    def test_max_tokens_no_request_uses_model_settings(self):
        """Test that model max_tokens is used when request doesn't specify it."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(max_tokens=8192)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager
            self._state.sampling = SamplingDefaults(max_tokens=4096)

            _, _, _, _, _, _, _, max_tokens, _, _ = get_sampling_params(
                None, None, "test-model"
            )
            assert max_tokens == 8192  # model setting, not global 4096


class TestExceptionHandlers:
    """Tests for global exception handlers that log API errors."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app, raise_server_exceptions=False)

    def test_http_exception_logged(self, client, caplog):
        """Test that HTTPException responses are logged."""
        # /v1/models requires startup, so a 404 on a non-existent route works
        response = client.get("/v1/nonexistent-endpoint")
        assert response.status_code == 404

    def test_validation_error_logged(self, client, caplog):
        """Test that request validation errors (422) are logged."""
        # POST to /v1/chat/completions with invalid body triggers validation
        response = client.post(
            "/v1/chat/completions",
            json={"invalid_field": "bad"},
        )
        # Should be 422 (validation error) or 500 (server not initialized)
        assert response.status_code in (422, 500)

    def test_exception_handler_returns_json(self, client):
        """Test that exception handlers return proper JSON responses."""
        response = client.get("/v1/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data

    def test_api_validation_error_openai_format(self, client):
        """Test that /v1/* validation errors use OpenAI-compatible format."""
        response = client.post(
            "/v1/chat/completions",
            json={"invalid_field": "bad"},
        )
        # 422 validation or 500 if server not init - both should have error key
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "type" in data["error"]
        assert "param" in data["error"]

    def test_non_api_route_detail_format(self, client):
        """Test that non-/v1/ routes keep the traditional detail format."""
        response = client.get("/nonexistent-page")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestModelFallback:
    """Tests for model fallback to default when requested model not found."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def _setup_pool(self, found_model=None):
        """Create a mock engine pool."""
        pool = MagicMock()
        pool.resolve_model_id.side_effect = lambda mid, _sm: mid

        if found_model:
            mock_engine = MagicMock()

            async def mock_get_engine(model_id):
                if model_id == found_model:
                    return mock_engine
                raise ModelNotFoundError(model_id, [found_model])

            pool.get_engine = AsyncMock(side_effect=mock_get_engine)
        else:
            pool.get_engine = AsyncMock(
                side_effect=ModelNotFoundError("unknown", [])
            )

        self._state.engine_pool = pool
        return pool

    @pytest.mark.asyncio
    async def test_fallback_disabled_returns_404(self):
        """When model_fallback is off, unknown model returns 404."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = False
        self._state.default_model = "default-model"
        self._setup_pool(found_model="default-model")

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("unknown-model", EngineType.LLM)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_fallback_enabled_returns_default(self):
        """When model_fallback is on, unknown model falls back to default."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = True
        self._state.default_model = "default-model"
        self._setup_pool(found_model="default-model")

        engine = await get_engine("unknown-model", EngineType.LLM)
        assert engine is not None

    @pytest.mark.asyncio
    async def test_fallback_enabled_no_default_returns_404(self):
        """When model_fallback is on but no default model, returns 404."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = True
        self._state.default_model = None
        self._setup_pool()

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("unknown-model", EngineType.LLM)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_fallback_not_applied_to_embedding(self):
        """Fallback should not apply to embedding engine type."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = True
        self._state.default_model = "default-model"
        self._setup_pool(found_model="default-model")

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("unknown-model", EngineType.EMBEDDING)
        assert exc_info.value.status_code == 404


class TestGetEngineLLMTypeValidation:
    """LLM endpoints must reject non-LLM engines with a clean 400 (#507).

    Issue #507: POST /v1/chat/completions against an STT/TTS/STS/Embedding
    model was producing an unhandled 500 with `'STTEngine' object has no
    attribute 'model_type'` because `get_engine(..., EngineType.LLM)` never
    validated that the resolved engine was actually an LLM. The fix adds an
    isinstance check mirroring the one already in place for EMBEDDING and
    RERANKER.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def _pool_returning(self, engine):
        pool = MagicMock()
        pool.resolve_model_id.side_effect = lambda mid, _sm: mid
        pool.get_engine = AsyncMock(return_value=engine)
        self._state.engine_pool = pool
        return pool

    @pytest.mark.asyncio
    async def test_llm_rejects_stt_engine(self):
        """Requesting an STT model on an LLM endpoint returns HTTP 400, not 500."""
        from omlx.engine.stt import STTEngine
        stt = MagicMock(spec=STTEngine)
        self._pool_returning(stt)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("whisper-large-v3-turbo", EngineType.LLM)
        assert exc_info.value.status_code == 400
        detail = str(exc_info.value.detail).lower()
        assert "not an llm" in detail or "not a chat" in detail or "not a text" in detail

    @pytest.mark.asyncio
    async def test_llm_rejects_tts_engine(self):
        """Requesting a TTS model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.tts import TTSEngine
        tts = MagicMock(spec=TTSEngine)
        self._pool_returning(tts)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("qwen3-tts", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_rejects_sts_engine(self):
        """Requesting an STS model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.sts import STSEngine
        sts = MagicMock(spec=STSEngine)
        self._pool_returning(sts)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("deepfilternet", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_rejects_embedding_engine(self):
        """Requesting an embedding model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.embedding import EmbeddingEngine
        emb = MagicMock(spec=EmbeddingEngine)
        self._pool_returning(emb)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("bge-small", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_rejects_reranker_engine(self):
        """Requesting a reranker model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.reranker import RerankerEngine
        rr = MagicMock(spec=RerankerEngine)
        self._pool_returning(rr)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("jina-reranker", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_accepts_llm_engine(self):
        """A genuine LLM engine passes validation and is returned as-is."""
        from omlx.engine.base import BaseEngine
        llm = MagicMock(spec=BaseEngine)
        self._pool_returning(llm)

        engine = await get_engine("llama-3", EngineType.LLM)
        assert engine is llm

class TestCancelEndpoint:
    """Tests for POST /v1/cancel/{request_id}.

    The endpoint exists so the gateway can abort an in-flight chat
    completion out-of-band, without depending on TCP-close detection.
    The gateway stamps a UUID into the chat completion request body
    (via ChatCompletionRequest.request_id) and posts to this endpoint
    when its asyncio handler task is cancelled.
    """

    @pytest.fixture
    def client(self):
        from omlx.server import _server_state
        original_api_key = _server_state.api_key
        _server_state.api_key = "test-key"
        yield TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": "Bearer test-key"},
        )
        _server_state.api_key = original_api_key

    def _setup_pool_with_engines(self, *engines):
        """Build a mock engine_pool with the given (model_id, engine) pairs."""
        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = [m for m, _ in engines]
        entries = {}
        for model_id, engine in engines:
            entry = MagicMock()
            entry.engine = engine
            entries[model_id] = entry
        pool.get_entry.side_effect = lambda mid: entries.get(mid)
        return pool

    def test_cancel_returns_503_when_pool_uninitialized(self, client):
        """Without an engine pool the endpoint must surface 503, not 500."""
        with patch("omlx.server.get_engine_pool", return_value=None):
            response = client.post("/v1/cancel/abc-123")
        assert response.status_code == 503

    def test_cancel_finds_request_on_first_engine(self, client):
        """Successful cancel: scheduler finds the request and aborts it."""
        engine = MagicMock()
        engine.abort_request = AsyncMock(return_value=True)
        pool = self._setup_pool_with_engines(("model-a", engine))

        with patch("omlx.server.get_engine_pool", return_value=pool):
            response = client.post("/v1/cancel/abc-123")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "request_id": "abc-123",
            "found": True,
            "cancelled": True,
        }
        engine.abort_request.assert_awaited_once_with("abc-123")

    def test_cancel_walks_engines_until_match(self, client):
        """If the first engine's scheduler doesn't have the ID, walk on."""
        engine_a = MagicMock()
        engine_a.abort_request = AsyncMock(return_value=False)
        engine_b = MagicMock()
        engine_b.abort_request = AsyncMock(return_value=True)
        engine_c = MagicMock()
        engine_c.abort_request = AsyncMock(return_value=False)
        pool = self._setup_pool_with_engines(
            ("model-a", engine_a),
            ("model-b", engine_b),
            ("model-c", engine_c),
        )

        with patch("omlx.server.get_engine_pool", return_value=pool):
            response = client.post("/v1/cancel/abc-123")

        assert response.status_code == 200
        assert response.json()["cancelled"] is True
        engine_a.abort_request.assert_awaited_once()
        engine_b.abort_request.assert_awaited_once()
        # Should stop on first match — engine_c not consulted
        engine_c.abort_request.assert_not_called()

    def test_cancel_returns_not_found_when_no_engine_has_id(self, client):
        """All engines say no → cancelled=False, found=False, status 200."""
        engine_a = MagicMock()
        engine_a.abort_request = AsyncMock(return_value=False)
        engine_b = MagicMock()
        engine_b.abort_request = AsyncMock(return_value=False)
        pool = self._setup_pool_with_engines(
            ("model-a", engine_a),
            ("model-b", engine_b),
        )

        with patch("omlx.server.get_engine_pool", return_value=pool):
            response = client.post("/v1/cancel/abc-123")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "request_id": "abc-123",
            "found": False,
            "cancelled": False,
        }

    def test_cancel_swallows_per_engine_exception(self, client):
        """If one engine's abort_request raises, we keep walking the rest."""
        bad_engine = MagicMock()
        bad_engine.abort_request = AsyncMock(side_effect=RuntimeError("boom"))
        good_engine = MagicMock()
        good_engine.abort_request = AsyncMock(return_value=True)
        pool = self._setup_pool_with_engines(
            ("model-bad", bad_engine),
            ("model-good", good_engine),
        )

        with patch("omlx.server.get_engine_pool", return_value=pool):
            response = client.post("/v1/cancel/abc-123")

        assert response.status_code == 200
        assert response.json()["cancelled"] is True

    def test_cancel_skips_unloaded_engines(self, client):
        """Entries with engine=None are skipped without raising."""
        loaded = MagicMock()
        loaded.abort_request = AsyncMock(return_value=True)
        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = ["unloaded", "loaded"]
        unloaded_entry = MagicMock()
        unloaded_entry.engine = None
        loaded_entry = MagicMock()
        loaded_entry.engine = loaded
        pool.get_entry.side_effect = lambda mid: {
            "unloaded": unloaded_entry, "loaded": loaded_entry,
        }.get(mid)

        with patch("omlx.server.get_engine_pool", return_value=pool):
            response = client.post("/v1/cancel/abc-123")

        assert response.status_code == 200
        assert response.json()["cancelled"] is True
        loaded.abort_request.assert_awaited_once()

    def test_cancel_requires_auth(self):
        """The cancel endpoint is auth-gated like every other /v1/* route."""
        from omlx.server import _server_state
        original = _server_state.api_key
        _server_state.api_key = "test-key"
        try:
            client = TestClient(app, raise_server_exceptions=False)
            # No Authorization header
            response = client.post("/v1/cancel/abc-123")
            assert response.status_code in (401, 403)
        finally:
            _server_state.api_key = original
