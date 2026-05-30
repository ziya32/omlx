# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.model_loading.maybe_load_custom_quantization."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import (
    _has_mtp_weights,
    maybe_apply_pre_load_patches,
    maybe_load_custom_quantization,
)


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


def _write_index(tmp_path, keys) -> str:
    """Write a minimal safetensors index mapping *keys* to one shard."""
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}})
    )
    return str(tmp_path)


class TestNoDispatch:
    """Cases where the dispatcher should return None and let the caller
    fall back to the standard mlx-lm/mlx-vlm load path."""

    def test_missing_config_returns_none(self, tmp_path):
        # tmp_path has no config.json
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_malformed_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, "{not valid json")
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_no_quantization_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_empty_quant_method_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"quantization_config": {}}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_unknown_quant_method_returns_none(self, tmp_path):
        # Methods we don't dispatch on (mlx-lm may or may not handle them
        # natively; either way the dispatcher stays out of the way).
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "awq"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) is None


def _install_paroquant_stub(monkeypatch, load_impl):
    """Register a minimal paroquant.inference.backends.mlx.load stub."""
    names = [
        "paroquant",
        "paroquant.inference",
        "paroquant.inference.backends",
        "paroquant.inference.backends.mlx",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    load_mod = types.ModuleType("paroquant.inference.backends.mlx.load")
    load_mod.load = load_impl
    monkeypatch.setitem(sys.modules, "paroquant.inference.backends.mlx.load", load_mod)


class TestParoquantDispatch:
    """Cases where quant_method == 'paroquant'."""

    def test_paroquant_missing_raises_install_hint(self, tmp_path, monkeypatch):
        # Force the import to fail: shadow the package with a sentinel that
        # blocks submodule resolution. setitem(..., None) makes `import X`
        # raise ImportError in the standard machinery.
        for name in [
            "paroquant",
            "paroquant.inference",
            "paroquant.inference.backends",
            "paroquant.inference.backends.mlx",
            "paroquant.inference.backends.mlx.load",
        ]:
            monkeypatch.setitem(sys.modules, name, None)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ImportError, match="paroquant"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_paroquant_text_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is True
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=False)
        assert result == ("MODEL", "PROC")

    def test_paroquant_vlm_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is False
            return "VLM_MODEL", "VLM_PROC", True

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=True)
        assert result == ("VLM_MODEL", "VLM_PROC")

    def test_paroquant_text_only_for_vlm_load_raises(self, tmp_path, monkeypatch):
        # is_vlm=True but the loader returned (..., loaded_is_vlm=False).
        def fake_load(model_path, force_text):
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ValueError, match="text-only"):
            maybe_load_custom_quantization(path, is_vlm=True)

    def test_quant_method_case_insensitive(self, tmp_path, monkeypatch):
        # The dispatcher lowercases quant_method, so mixed-case configs
        # (e.g. produced by other tooling) still hit the paroquant path.
        captured = {}

        def fake_load(model_path, force_text):
            captured["called"] = True
            return "M", "P", False

        _install_paroquant_stub(monkeypatch, fake_load)
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "ParoQuant"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) == ("M", "P")
        assert captured["called"] is True


class TestVlmMtpPreLoadDispatch:
    """maybe_apply_pre_load_patches must wire the mlx-vlm MTP sanitize
    patch alongside the runtime patch for MTP-capable VLM checkpoints.

    The dense Qwen3.5/3.6 VLM runtime patch does not touch Model.sanitize;
    it relies on apply_mlx_vlm_mtp_patch having installed the mtp.*
    preservation first. If only the runtime patch runs, stock mlx-vlm
    sanitize strips every mtp.* key and the MTP head loads at random
    init (PR #1320).

    MoE VLMs without declared MTP heads still need the sanitize replacement
    so pre-converted switch_mlp weights load (issue #1261); runtime patch
    must not run on that path."""

    def _stub_patches(self, monkeypatch):
        """Replace the patch modules with mocks that record call order.

        Returns the recorded-order list plus the sanitize/runtime mocks."""
        calls: list[str] = []
        sanitize_mock = MagicMock(side_effect=lambda: calls.append("sanitize") or True)
        runtime_mock = MagicMock(side_effect=lambda: calls.append("runtime") or True)
        # Side-step the real mlx-lm load_config monkey-patch.
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(
                set_mtp_active=MagicMock(),
                apply_mlx_lm_mtp_patch=MagicMock(return_value=True),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_vlm_mtp",
            MagicMock(
                apply_mlx_vlm_mtp_patch=sanitize_mock,
                apply_mlx_vlm_mtp_runtime_patch=runtime_mock,
                set_mtp_weights_present=MagicMock(),
            ),
        )
        return calls, sanitize_mock, runtime_mock

    @staticmethod
    def _set_present_mock():
        return sys.modules["omlx.patches.mlx_vlm_mtp"].set_mtp_weights_present

    def test_sanitize_patch_runs_before_runtime_for_vlm_mtp(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        # qwen3_5 (dense VLM) declaring an MTP head under text_config.
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        # Ordering matters: the dense runtime patch assumes sanitize was
        # already installed by apply_mlx_vlm_mtp_patch.
        assert calls == ["sanitize", "runtime"]

    def test_vlm_patches_applied_when_mtp_disabled_for_vlm(self, tmp_path, monkeypatch):
        # Issue #1404: persisted ``mtp.*`` weights must still get a binding
        # site on the LanguageModel tree when entering through VLMBatchedEngine
        # even with mtp_enabled=False. Otherwise mlx-vlm's strict load_weights
        # fails with "parameters not in model" and the engine falls back to
        # LLM, silently dropping vision.
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        assert calls == ["sanitize", "runtime"]

    def test_vlm_patches_skipped_when_not_for_vlm(self, tmp_path, monkeypatch):
        # BatchedEngine / DFlashEngine / LLM loader paths must NOT touch
        # mlx-vlm classes even when the model declares MTP heads. for_vlm
        # defaults to False so they pass through without invoking mlx-vlm
        # patches.
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []

    def test_qwen36_moe_vlm_sanitize_when_no_mtp_heads(self, tmp_path, monkeypatch):
        # mlx-lm Qwen3.6 MoE VLMs without MTP heads still need the mlx-vlm
        # sanitize replacement so pre-converted switch_mlp weights load.
        # Runtime MTP patch must NOT run — there is no mtp.* tree to bind.
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 0}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_not_called()
        assert calls == ["sanitize"]

    def test_qwen36_moe_vlm_sanitize_skipped_without_for_vlm(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 0}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []

    def test_weights_present_flag_true_when_mtp_weights_in_checkpoint(
        self, tmp_path, monkeypatch
    ):
        # Config declares MTP heads AND the checkpoint ships mtp.* weights →
        # the runtime patch is told to attach the head.
        self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_index(
            tmp_path,
            [
                "language_model.mtp.fc.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
            ],
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        self._set_present_mock().assert_called_once_with(True)

    def test_weights_present_flag_false_when_mtp_weights_stripped(
        self, tmp_path, monkeypatch
    ):
        # Regression for the Qwen3.6-35B-A3B-emee8bit fallback: config
        # advertises mtp_num_hidden_layers=1 but the checkpoint has zero
        # mtp.* tensors (stripped at quant time). The head must NOT be
        # attached — otherwise strict load_weights fails ("Missing N
        # parameters: language_model.mtp.*") and VLMBatchedEngine silently
        # downgrades to a text-only LLM, dropping vision.
        self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_index(
            tmp_path, ["language_model.model.layers.0.self_attn.q_proj.weight"]
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        self._set_present_mock().assert_called_once_with(False)

    def test_warns_when_mtp_enabled_but_weights_absent(
        self, tmp_path, monkeypatch, caplog
    ):
        self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_index(
            tmp_path, ["language_model.model.layers.0.self_attn.q_proj.weight"]
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        with caplog.at_level("WARNING"):
            maybe_apply_pre_load_patches(
                path, model_settings=settings, for_vlm=True
            )

        self._set_present_mock().assert_called_once_with(False)
        assert any(
            "ships no mtp.* weights" in r.getMessage() for r in caplog.records
        )

    def test_no_warning_when_mtp_disabled_and_weights_absent(
        self, tmp_path, monkeypatch, caplog
    ):
        # Common case for stripped checkpoints (mtp off, no weights): silent,
        # head simply not attached, vision preserved.
        self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_index(
            tmp_path, ["language_model.model.layers.0.self_attn.q_proj.weight"]
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        with caplog.at_level("WARNING"):
            maybe_apply_pre_load_patches(
                path, model_settings=settings, for_vlm=True
            )

        assert not any(
            "ships no mtp.* weights" in r.getMessage() for r in caplog.records
        )


class TestHasMtpWeights:
    """_has_mtp_weights inspects only safetensors key names to decide whether
    a checkpoint actually ships mtp.* tensors (no tensor data is loaded)."""

    def test_index_with_mtp_key_returns_true(self, tmp_path):
        _write_index(
            tmp_path,
            ["language_model.mtp.fc.weight", "language_model.model.norm.weight"],
        )
        assert _has_mtp_weights(str(tmp_path)) is True

    def test_index_without_mtp_key_returns_false(self, tmp_path):
        _write_index(
            tmp_path,
            [
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "vision_tower.blocks.0.attn.proj.weight",
            ],
        )
        assert _has_mtp_weights(str(tmp_path)) is False

    def test_substring_mtp_is_not_a_false_positive(self, tmp_path):
        # 'mtp' only counts as a full dot-separated path segment.
        _write_index(tmp_path, ["model.something_mtp_extra.weight"])
        assert _has_mtp_weights(str(tmp_path)) is False

    def test_non_directory_path_returns_true(self):
        # Can't introspect → preserve prior unconditional-attach behavior.
        assert _has_mtp_weights("/no/such/path/here") is True

    def test_dir_without_index_or_shards_returns_true(self, tmp_path):
        assert _has_mtp_weights(str(tmp_path)) is True

    def test_shard_scan_fallback_detects_mtp(self, tmp_path):
        np = pytest.importorskip("numpy")
        st = pytest.importorskip("safetensors.numpy")
        st.save_file(
            {"language_model.mtp.norm.weight": np.zeros((2,), dtype=np.float32)},
            str(tmp_path / "model.safetensors"),
        )
        assert _has_mtp_weights(str(tmp_path)) is True

    def test_shard_scan_fallback_without_mtp_returns_false(self, tmp_path):
        np = pytest.importorskip("numpy")
        st = pytest.importorskip("safetensors.numpy")
        st.save_file(
            {"language_model.model.norm.weight": np.zeros((2,), dtype=np.float32)},
            str(tmp_path / "model.safetensors"),
        )
        assert _has_mtp_weights(str(tmp_path)) is False
