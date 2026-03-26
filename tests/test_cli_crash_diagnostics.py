# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI native crash dump helpers (SIGABRT diagnostics)."""

import io
import json
from unittest.mock import MagicMock, patch

from omlx.cli import _write_abort_memory_snapshot, _write_engine_pool_crash_snapshot
from omlx.engine_pool import EnginePool


class TestWriteAbortMemorySnapshot:
    def test_writes_sections(self):
        buf = io.StringIO()
        _write_abort_memory_snapshot(buf)
        out = buf.getvalue()
        assert "--- crash memory snapshot (best-effort) ---" in out
        assert "rusage:" in out
        assert "mlx.get_active_memory():" in out
        assert "psutil process:" in out

    def test_handles_write_error(self):
        class BadWriter:
            def write(self, _):
                raise OSError("boom")

            def flush(self):
                pass

        # Must not raise
        _write_abort_memory_snapshot(BadWriter())


class TestWriteEnginePoolCrashSnapshot:
    def test_engine_pool_not_initialized(self):
        buf = io.StringIO()
        st = MagicMock()
        st.engine_pool = None
        st.process_memory_enforcer = None

        with patch("omlx.server.get_server_state", return_value=st):
            _write_engine_pool_crash_snapshot(buf)

        out = buf.getvalue()
        assert "--- engine pool / server snapshot ---" in out
        assert "engine_pool: not initialized" in out

    def test_includes_pool_and_enforcer(self, tmp_path):
        model_a = tmp_path / "model-a"
        model_a.mkdir()
        (model_a / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_a / "model.safetensors").write_bytes(b"0" * 1024)

        pool = EnginePool(max_model_memory=10 * 1024**3)
        pool.discover_models(str(tmp_path))

        st = MagicMock()
        st.engine_pool = pool
        enf = MagicMock()
        enf.get_status.return_value = {"enabled": True, "max_bytes": 99}
        st.process_memory_enforcer = enf

        buf = io.StringIO()
        with patch("omlx.server.get_server_state", return_value=st):
            _write_engine_pool_crash_snapshot(buf)

        out = buf.getvalue()
        assert "--- engine pool / server snapshot ---" in out
        assert '"debug_pool"' in out
        assert '"models"' in out
        assert "process_memory_enforcer" in out
        assert '"enabled": true' in out

    def test_import_error_surfaces_as_unavailable(self):
        buf = io.StringIO()

        def boom():
            raise RuntimeError("no server")

        with patch("omlx.server.get_server_state", side_effect=boom):
            _write_engine_pool_crash_snapshot(buf)

        out = buf.getvalue()
        assert "unavailable" in out
        assert "no server" in out
