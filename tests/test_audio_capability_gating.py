# SPDX-License-Identifier: Apache-2.0
"""Tests for audio router capability gating.

Background: an earlier version of server.py wrapped the audio router
registration in ``try: import mlx_audio; app.include_router(...)``.  When
mlx-audio wasn't installed, every ``/v1/audio/*`` endpoint returned a
silent 404 with no hint about the actual cause — and even the routes that
don't need mlx-audio (``/v1/audio/speakers``, ``/v1/audio/languages``)
disappeared as collateral damage.

The current behaviour:
- The audio router is registered unconditionally (audio_routes.py and its
  module-level imports don't pull mlx_audio).
- Endpoints that load engines (TTS / STT / STS) catch the ImportError that
  the engine's start() raises and surface it as 501 (Not Implemented) with
  the engine's actionable ``"pip install 'omlx[audio]'"`` message.
- 501 (not 503) because the missing dep is a deployment-level capability
  gap, not a transient overload — clients with retry-on-503 policies must
  not pound the endpoint.
- Endpoints that don't touch mlx-audio (speakers / languages) continue to
  work even when it's not installed.

These tests use TestClient with a mocked engine pool — no real mlx-audio
or model loading required, so they exercise the same 501 path that
production hits when the optional dep isn't present.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


_INSTALL_HINT = (
    "mlx-audio is required for TTS inference. "
    "Install it with: pip install 'omlx[audio]'"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lifespan_safe_pool() -> MagicMock:
    """Pool with the AsyncMocks the FastAPI lifespan needs to start cleanly."""
    pool = MagicMock()
    pool._entries = {}
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.acquire_engine = MagicMock()
    pool.release_engine = MagicMock()
    pool.ensure_engine_alive = MagicMock()
    return pool


@pytest.fixture
def import_error_pool():
    """Pool whose ``get_engine`` raises ImportError exactly like an audio
    engine's ``start()`` would when mlx-audio isn't installed."""
    pool = _make_lifespan_safe_pool()
    pool.resolve_model_id = MagicMock(side_effect=lambda mid, _sm=None: mid)
    pool.get_engine = AsyncMock(side_effect=ImportError(_INSTALL_HINT))
    return pool


@pytest.fixture
def client_with_import_error_pool(import_error_pool):
    """TestClient wired up so engine acquisition raises ImportError, and
    auth is short-circuited to ``True`` so we can hit the routes without
    forging tokens."""
    from omlx.server import app

    with patch("omlx.server._server_state") as mock_state:
        mock_state.engine_pool = import_error_pool
        mock_state.global_settings = None
        mock_state.process_memory_enforcer = None
        mock_state.hf_downloader = None
        mock_state.ms_downloader = None
        mock_state.mcp_manager = None
        # Auth-OFF here: this suite is about capability gating, not auth
        # — the server-level verify_api_key dependency would otherwise
        # gate the request before it reaches the router (api_key=None
        # makes verify_api_key return True at server.py:267).
        mock_state.api_key = None
        mock_state.settings_manager = MagicMock()
        mock_state.settings_manager.get_settings.return_value = MagicMock(
            model_alias=None,
            aliases=None,
            default_voice=None,
            default_instruct=None,
            default_language=None,
        )

        # Bypass the audio router's _verify_auth dep (separate layer from
        # server-level verify_api_key).
        import omlx.api.audio_routes as ar
        original_auth = ar._auth_dependency

        async def _allow(request, credentials=None):
            return True

        ar._auth_dependency = _allow
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client
        finally:
            ar._auth_dependency = original_auth


# ---------------------------------------------------------------------------
# Router registration is unconditional
# ---------------------------------------------------------------------------


class TestAudioRoutesAlwaysRegistered:
    """The audio router must register without mlx-audio installed.

    Importing ``omlx.server`` should expose all five ``/v1/audio/*``
    endpoints regardless of whether mlx-audio is importable — see the
    block at the bottom of server.py and its commit comment.
    """

    def test_all_five_audio_routes_present(self):
        from omlx.server import app

        audio_paths = sorted({
            r.path for r in app.routes if r.path.startswith("/v1/audio/")
        })
        assert audio_paths == [
            "/v1/audio/languages",
            "/v1/audio/process",
            "/v1/audio/speakers",
            "/v1/audio/speech",
            "/v1/audio/transcriptions",
        ]


# ---------------------------------------------------------------------------
# _use_engine maps ImportError -> 501
# ---------------------------------------------------------------------------


class TestUseEngineImportErrorMapping:
    """Direct unit test for the ``_use_engine`` context manager —
    independent of route plumbing."""

    @pytest.mark.asyncio
    async def test_import_error_becomes_501(self, import_error_pool):
        from omlx.api.audio_routes import _use_engine

        with patch(
            "omlx.api.audio_routes._get_engine_pool",
            return_value=import_error_pool,
        ), patch(
            "omlx.api.audio_routes._get_settings_manager",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as excinfo:
                async with _use_engine("anything"):
                    pytest.fail("engine yielded despite ImportError")

        assert excinfo.value.status_code == 501
        assert "mlx-audio is required" in excinfo.value.detail
        assert "pip install 'omlx[audio]'" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_model_not_found_still_404(self):
        """The new ImportError branch must not steal the existing 404
        path used by ``ModelNotFoundError``."""
        from omlx.api.audio_routes import _use_engine
        from omlx.exceptions import ModelNotFoundError

        pool = MagicMock()
        pool._entries = {}
        pool.resolve_model_id = MagicMock(side_effect=lambda mid, _sm=None: mid)
        pool.get_engine = AsyncMock(
            side_effect=ModelNotFoundError("nope", available_models=["a", "b"])
        )

        with patch(
            "omlx.api.audio_routes._get_engine_pool",
            return_value=pool,
        ), patch(
            "omlx.api.audio_routes._get_settings_manager",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as excinfo:
                async with _use_engine("nope"):
                    pytest.fail("engine yielded despite ModelNotFoundError")

        assert excinfo.value.status_code == 404
        assert "Available: a, b" in excinfo.value.detail


# ---------------------------------------------------------------------------
# Route-level: every audio inference endpoint surfaces 501 with the message
# ---------------------------------------------------------------------------


class TestAudioEndpointsWithoutMlxAudio:
    """Every endpoint that triggers an audio engine load must return 501
    + install message when mlx-audio is missing — not a generic 500 and
    not a misleading retryable 503."""

    def test_speech_non_streaming(self, client_with_import_error_pool):
        resp = client_with_import_error_pool.post(
            "/v1/audio/speech",
            json={"model": "fake-tts", "input": "hi", "voice": "default"},
        )
        assert resp.status_code == 501
        assert _INSTALL_HINT in resp.json()["error"]["message"]

    def test_speech_streaming(self, client_with_import_error_pool):
        # The streaming branch in create_speech has its own engine load
        # path — separate from the non-streaming branch — so it needs
        # its own coverage.
        resp = client_with_import_error_pool.post(
            "/v1/audio/speech",
            json={
                "model": "fake-tts",
                "input": "hi",
                "voice": "default",
                "stream": True,
            },
        )
        assert resp.status_code == 501
        assert _INSTALL_HINT in resp.json()["error"]["message"]

    def test_transcriptions(self, client_with_import_error_pool):
        # Goes through _use_engine, not the inline try/except path.
        resp = client_with_import_error_pool.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(b"RIFF\x00\x00\x00\x00WAVE"), "audio/wav")},
            data={"model": "fake-asr"},
        )
        assert resp.status_code == 501
        assert "mlx-audio is required" in resp.json()["error"]["message"]

    def test_process(self, client_with_import_error_pool):
        # STS endpoint — has its own inline engine load like create_speech.
        resp = client_with_import_error_pool.post(
            "/v1/audio/process",
            files={"file": ("a.wav", io.BytesIO(b"RIFF\x00\x00\x00\x00WAVE"), "audio/wav")},
            data={"model": "fake-sts"},
        )
        assert resp.status_code == 501
        assert "mlx-audio is required" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Routes that don't need mlx-audio still work
# ---------------------------------------------------------------------------


class TestNoMlxAudioRoutes:
    """Endpoints that read from disk / config (and never call into mlx-audio)
    must remain reachable when the dep isn't installed.  Previously the
    ``try: import mlx_audio`` gate hid these as collateral damage."""

    def test_speakers_returns_empty_list_for_unknown_model_dir(self):
        from omlx.server import app

        # Pool entry whose model_path doesn't contain a real config — the
        # speakers reader returns [] gracefully.  Crucially, we never hit
        # mlx-audio.
        entry = MagicMock()
        entry.engine_type = "audio_tts"
        entry.engine = None
        entry.model_path = "/nonexistent"

        pool = _make_lifespan_safe_pool()
        pool._entries = {"fake-tts": entry}

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = pool
            # Auth-OFF (see fixture above) so the server-level
            # verify_api_key dep doesn't 401 before the router runs.
            mock_state.api_key = None
            mock_state.settings_manager = None
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None

            import omlx.api.audio_routes as ar
            original_auth = ar._auth_dependency

            async def _allow(request, credentials=None):
                return True

            ar._auth_dependency = _allow
            try:
                with TestClient(app, raise_server_exceptions=False) as client:
                    resp = client.get("/v1/audio/speakers?model=fake-tts")
            finally:
                ar._auth_dependency = original_auth

        assert resp.status_code == 200
        assert resp.json() == {"speakers": []}
