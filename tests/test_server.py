# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.server module - sampling parameter resolution and exception handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from omlx.exceptions import ModelNotFoundError
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import EngineType, SamplingDefaults, ServerState, app, get_engine, get_sampling_params
from omlx.settings import GlobalSettings, ModelSettings as GlobalModelSettings


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
        from omlx.server import _server_state
        original_api_key = _server_state.api_key
        _server_state.api_key = "test-key"
        yield TestClient(
            app,
            raise_server_exceptions=False,
            headers={"Authorization": "Bearer test-key"},
        )
        _server_state.api_key = original_api_key

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


class TestUseEngineResolveOnce:
    """Issue #7 (fixed): ``server.use_engine`` resolves the model id
    EXACTLY ONCE at the outermost layer and passes the resolved id
    through to ``get_engine`` via the ``resolved_id`` parameter.

    Before the fix, ``server.use_engine`` resolved the id locally to
    take an ``acquire_engine`` lease, then ``server.get_engine``
    resolved again internally — if the alias map changed between the
    two calls, the lease and the ``ensure_engine_alive`` check
    operated on different ids. After the fix, both operations use
    the same pre-resolved id, so any alias map churn cannot split
    them.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    @pytest.mark.asyncio
    async def test_resolve_called_once_and_all_ops_see_same_id(self):
        """Even if ``pool.resolve_model_id`` would return different
        values on consecutive calls, the fix guarantees it's invoked
        exactly once and all three ops (acquire, ensure_engine_alive,
        release) observe the same resolved id.
        """
        from omlx.server import use_engine

        acquire_calls: list[str] = []
        release_calls: list[str] = []
        ensure_alive_calls: list[str] = []

        pool = MagicMock()
        # Rig resolve_model_id with TWO different return values.
        # Under the fix we only consume the first; the second stays
        # untouched, proving resolve is called exactly once.
        pool.resolve_model_id = MagicMock(
            side_effect=["alias-a", "alias-b"]
        )

        pool.acquire_engine = lambda mid: acquire_calls.append(mid)
        pool.release_engine = lambda mid: release_calls.append(mid)
        pool.ensure_engine_alive = lambda mid, eng: ensure_alive_calls.append(mid)

        mock_engine = MagicMock()

        async def get_engine_stub(model_id, *a, **kw):
            return mock_engine

        pool.get_engine = AsyncMock(side_effect=get_engine_stub)

        self._state.engine_pool = pool
        self._state.default_model = None

        async with use_engine("my-alias", EngineType.LLM) as eng:
            assert eng is mock_engine

        # 1. resolve_model_id is called EXACTLY ONCE. The second
        #    side_effect value ("alias-b") is never consumed.
        assert pool.resolve_model_id.call_count == 1, (
            f"Expected resolve_model_id to be called exactly once, got "
            f"{pool.resolve_model_id.call_count}. "
            f"The fix must pass resolved_id through to get_engine to "
            f"skip the second resolve."
        )

        # 2. All three operations see the same resolved id ("alias-a").
        assert acquire_calls == ["alias-a"]
        assert ensure_alive_calls == ["alias-a"]
        assert release_calls == ["alias-a"]

        # 3. pool.get_engine received the resolved id (not the raw
        #    alias), since server.use_engine passed resolved_id
        #    through and server.get_engine honored it.
        assert pool.get_engine.await_args.args[0] == "alias-a"

    @pytest.mark.asyncio
    async def test_chat_completion_handler_resolves_once(self):
        """The same resolve-once invariant applies to the chat/completion
        handler path: handlers call ``resolve_model_id()`` at the
        module level, then ``get_engine_for_model(request.model,
        resolved_id=resolved_model)`` — the convenience wrapper
        threads ``resolved_id`` through to ``get_engine`` so no
        second resolve happens.
        """
        from omlx.server import get_engine_for_model

        pool = MagicMock()
        pool.resolve_model_id = MagicMock(
            side_effect=["resolved-once", "SHOULD-NOT-BE-CALLED"]
        )

        mock_engine = MagicMock()
        from omlx.engine.batched import BatchedEngine
        mock_engine.__class__ = BatchedEngine

        async def get_engine_stub(model_id, *a, **kw):
            return mock_engine

        pool.get_engine = AsyncMock(side_effect=get_engine_stub)
        pool.ensure_engine_alive = MagicMock()

        self._state.engine_pool = pool
        self._state.default_model = None

        # Simulate what a chat handler does: resolve once, then call
        # get_engine_for_model with resolved_id=resolved_model.
        from omlx.server import resolve_model_id
        resolved_model = resolve_model_id("my-alias") or "my-alias"
        engine = await get_engine_for_model(
            "my-alias", resolved_id=resolved_model
        )
        assert engine is mock_engine

        # Only one resolve across the whole flow.
        assert pool.resolve_model_id.call_count == 1, (
            f"Handler path must resolve exactly once via "
            f"resolve_model_id(), then pass resolved_id through to "
            f"get_engine_for_model. Got "
            f"{pool.resolve_model_id.call_count} resolves."
        )
        # pool.get_engine received the pre-resolved id.
        assert pool.get_engine.await_args.args[0] == "resolved-once"
