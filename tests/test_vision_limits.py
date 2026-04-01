# SPDX-License-Identifier: Apache-2.0
"""Tests for vision processing limits — compute_vision_limits, _read_context_window,
_model_status_entry, and error code propagation."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omlx.engine_pool import EngineEntry, EnginePool, EngineState


def _make_entry(
    model_id: str = "test-vlm",
    model_path: str = "/tmp/test-vlm",
    model_type: str = "vlm",
    engine_type: str = "vlm",
    estimated_size: int = 40_000_000_000,  # 40 GB
    is_pinned: bool = False,
    state: EngineState = EngineState.UNLOADED,
) -> EngineEntry:
    entry = EngineEntry(
        model_id=model_id,
        model_path=model_path,
        model_type=model_type,
        engine_type=engine_type,
        estimated_size=estimated_size,
    )
    entry.is_pinned = is_pinned
    entry.state = state
    return entry


def _make_pool_with_enforcer(
    max_bytes: int,
    entries: dict[str, EngineEntry] | None = None,
) -> EnginePool:
    pool = EnginePool(max_model_memory=max_bytes)
    enforcer = MagicMock()
    enforcer.max_bytes = max_bytes
    pool._process_memory_enforcer = enforcer
    if entries:
        pool._entries = entries
    return pool


# ── _read_context_window ─────────────────────────────────────────────────────


class TestReadContextWindow:

    def test_reads_from_text_config(self, tmp_path):
        model_dir = tmp_path / "vlm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "text_config": {"max_position_embeddings": 262144},
        }))
        entry = _make_entry(model_path=str(model_dir))
        assert EnginePool._read_context_window(entry) == 262144

    def test_reads_from_root(self, tmp_path):
        model_dir = tmp_path / "vlm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "max_position_embeddings": 131072,
        }))
        entry = _make_entry(model_path=str(model_dir))
        assert EnginePool._read_context_window(entry) == 131072

    def test_text_config_takes_precedence(self, tmp_path):
        model_dir = tmp_path / "vlm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "max_position_embeddings": 100000,
            "text_config": {"max_position_embeddings": 262144},
        }))
        entry = _make_entry(model_path=str(model_dir))
        assert EnginePool._read_context_window(entry) == 262144

    def test_missing_config_returns_zero(self, tmp_path):
        entry = _make_entry(model_path=str(tmp_path / "nonexistent"))
        assert EnginePool._read_context_window(entry) == 0

    def test_no_position_embeddings_returns_zero(self, tmp_path):
        model_dir = tmp_path / "vlm-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
        entry = _make_entry(model_path=str(model_dir))
        assert EnginePool._read_context_window(entry) == 0


# ── compute_vision_limits ────────────────────────────────────────────────────


class TestComputeVisionLimits:

    def _make_vlm_with_config(self, tmp_path, ctx_window: int = 262144):
        model_dir = tmp_path / "vlm-model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps({
            "text_config": {"max_position_embeddings": ctx_window},
        }))
        return _make_entry(model_path=str(model_dir))

    def test_basic_limits(self, tmp_path):
        vlm = self._make_vlm_with_config(tmp_path)
        embed = _make_entry(
            model_id="embed", model_type="embedding", engine_type="embedding",
            estimated_size=4_000_000_000, is_pinned=True,
        )
        pool = _make_pool_with_enforcer(
            max_bytes=56_000_000_000,  # 56 GB (64 GB machine)
            entries={"embed": embed, "test-vlm": vlm},
        )

        limits = pool.compute_vision_limits(vlm)
        assert limits["max_vision_tokens"] == int(262144 * 0.8)
        assert limits["chunk_budget_pixels"] > 0
        assert limits["memory_headroom_bytes"] > 0

    def test_headroom_excludes_unpinned_models(self, tmp_path):
        vlm = self._make_vlm_with_config(tmp_path)
        embed = _make_entry(
            model_id="embed", model_type="embedding", engine_type="embedding",
            estimated_size=4_000_000_000, is_pinned=True,
            state=EngineState.ACTIVE,
        )
        tts = _make_entry(
            model_id="tts", model_type="tts", engine_type="tts",
            estimated_size=8_000_000_000, is_pinned=False,
            state=EngineState.ACTIVE,
        )
        pool = _make_pool_with_enforcer(
            max_bytes=56_000_000_000,
            entries={"embed": embed, "tts": tts, "test-vlm": vlm},
        )

        limits = pool.compute_vision_limits(vlm)
        # TTS (8 GB, unpinned) should NOT reduce headroom
        # headroom = 56GB - 4GB(embed) - 40GB*1.25(vlm) = 56 - 4 - 50 = 2 GB
        expected_headroom = 56_000_000_000 - 4_000_000_000 - int(40_000_000_000 * 1.25)
        assert limits["memory_headroom_bytes"] == expected_headroom

    def test_no_enforcer_returns_empty(self):
        pool = EnginePool(max_model_memory=None)
        pool._process_memory_enforcer = None
        entry = _make_entry()
        assert pool.compute_vision_limits(entry) == {}

    def test_scales_with_memory(self, tmp_path):
        """More memory → larger chunk budget."""
        vlm = self._make_vlm_with_config(tmp_path)

        pool_small = _make_pool_with_enforcer(
            max_bytes=56_000_000_000,
            entries={"test-vlm": vlm},
        )
        pool_large = _make_pool_with_enforcer(
            max_bytes=96_000_000_000,
            entries={"test-vlm": vlm},
        )

        small = pool_small.compute_vision_limits(vlm)
        large = pool_large.compute_vision_limits(vlm)

        assert large["chunk_budget_pixels"] > small["chunk_budget_pixels"]
        assert large["memory_headroom_bytes"] > small["memory_headroom_bytes"]
        # Context-derived limits should be the same (same model)
        assert large["max_vision_tokens"] == small["max_vision_tokens"]

    def test_zero_headroom(self, tmp_path):
        """When VLM fills all memory, chunk budget should be 0."""
        vlm = self._make_vlm_with_config(tmp_path)
        vlm.estimated_size = 56_000_000_000  # VLM alone fills 56GB * 1.25 > 56GB

        pool = _make_pool_with_enforcer(
            max_bytes=56_000_000_000,
            entries={"test-vlm": vlm},
        )

        limits = pool.compute_vision_limits(vlm)
        assert limits["chunk_budget_pixels"] == 0
        assert limits["memory_headroom_bytes"] == 0


# ── _model_status_entry ──────────────────────────────────────────────────────


class TestModelStatusEntry:

    def test_vlm_has_vision_limits(self, tmp_path):
        model_dir = tmp_path / "vlm"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "text_config": {"max_position_embeddings": 262144},
        }))
        vlm = _make_entry(model_path=str(model_dir))
        pool = _make_pool_with_enforcer(
            max_bytes=56_000_000_000,
            entries={"test-vlm": vlm},
        )

        entry = pool._model_status_entry("test-vlm", vlm)
        assert "vision_limits" in entry
        assert entry["vision_limits"]["max_vision_tokens"] > 0

    def test_non_vlm_has_no_vision_limits(self):
        llm = _make_entry(model_type="llm", engine_type="batched")
        pool = _make_pool_with_enforcer(
            max_bytes=56_000_000_000,
            entries={"test-llm": llm},
        )

        entry = pool._model_status_entry("test-llm", llm)
        assert "vision_limits" not in entry


# ── Error code propagation (server-level) ────────────────────────────────────


class TestVisionTokenLimitErrorFormat:
    """Test that the error detail format is parseable."""

    def test_error_detail_format(self):
        """The vision_token_limit_exceeded detail should be parseable."""
        import re

        # Simulate the detail string generated by vlm.py
        vision_tokens = 164151
        max_tokens = 209715
        ctx = 262144
        safe = int(max_tokens / (vision_tokens / 549))
        detail = (
            f"vision_token_limit_exceeded: "
            f"~{vision_tokens} vision tokens, limit is "
            f"{max_tokens} (80% of {ctx} context). "
            f"Reduce frame count to ~{safe}."
        )

        # Verify prefix extraction works
        assert detail.startswith("vision_token_limit_exceeded:")
        body = detail[len("vision_token_limit_exceeded:"):].strip()

        # Verify regex matches
        m = re.search(
            r"~(\d+) vision tokens, limit is (\d+).*?of (\d+) context.*?~(\d+)\.",
            body,
        )
        assert m is not None
        assert int(m.group(1)) == 164151
        assert int(m.group(2)) == 209715
        assert int(m.group(3)) == 262144
