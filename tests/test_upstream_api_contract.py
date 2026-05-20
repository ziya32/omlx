# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the upstream symbols oMLX reaches for at runtime.

The merge-resolution bug fixed in commit 7ac0a5f (`dflash-mlx@1ba6713`
moved ``load_target_bundle`` out of ``runtime/__init__`` into
``runtime.bundle``, and ``generate_dflash_once`` was removed entirely)
slipped past the entire test suite because no test ever imported those
symbols against the bundled dflash-mlx — the structural tests in
``test_dflash_engine.py`` use ``"test-draft"`` paths and bail before
the upstream import runs.

These tests fail loudly the next time mlx-vlm, mlx-lm, or dflash-mlx
rev a public symbol oMLX depends on, without needing real model
weights. Two flavors per symbol:

1. **Import test** — fails if the dotted-path moves or the symbol is
   removed. Equivalent to the production ImportError at runtime.

2. **Signature test** — fails if a kwarg oMLX passes is no longer
   accepted (and the function would raise ``TypeError`` at call time).
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def _accepts(callable_obj, kwarg: str) -> bool:
    """True if ``callable_obj`` accepts ``kwarg`` as a parameter (or **kwargs)."""
    sig = inspect.signature(callable_obj)
    if kwarg in sig.parameters:
        return True
    # Accept ``**kwargs`` as a catch-all.
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


# =============================================================================
# DFlash (dflash-mlx) — pinned in pyproject.toml at @1ba6713 (v0.1.7).
# Engine consumer: omlx/engine/dflash.py.
# =============================================================================


class TestDFlashMLXContract:
    """Lock the dflash-mlx API surface DFlashEngine relies on."""

    def test_load_target_bundle_present(self):
        from dflash_mlx.runtime.bundle import load_target_bundle  # noqa: F401

    def test_load_draft_bundle_present(self):
        from dflash_mlx.runtime.bundle import load_draft_bundle  # noqa: F401

    def test_stream_dflash_generate_present(self):
        from dflash_mlx.runtime import stream_dflash_generate  # noqa: F401

    def test_get_stop_token_ids_present(self):
        from dflash_mlx.generate import get_stop_token_ids  # noqa: F401

    def test_stream_dflash_generate_signature(self):
        """The kwargs DFlashEngine passes must all be accepted."""
        from dflash_mlx.runtime import stream_dflash_generate

        expected_kwargs = {
            "target_model",
            "tokenizer",
            "draft_model",
            "prompt",
            "max_new_tokens",
            "stop_token_ids",
            "prompt_tokens_override",
            "runtime_context",
        }
        for kw in expected_kwargs:
            assert _accepts(stream_dflash_generate, kw), (
                f"dflash-mlx stream_dflash_generate no longer accepts "
                f"kwarg '{kw}' — omlx/engine/dflash.py would raise "
                f"TypeError on every generation. See commit 7ac0a5f "
                f"for the analogous temperature= drop."
            )

    def test_runtime_config_from_defaults_signature(self):
        """The tuning kwargs DFlashEngine builds runtime_context with."""
        from dflash_mlx.runtime.config import runtime_config_from_defaults

        expected_kwargs = {
            "draft_window_size",
            "draft_sink_size",
            "verify_mode",
        }
        for kw in expected_kwargs:
            assert _accepts(runtime_config_from_defaults, kw), (
                f"runtime_config_from_defaults no longer accepts '{kw}' — "
                f"the #1276 tuning settings would silently revert to dflash "
                f"defaults. See commit a3f7454."
            )

    def test_build_runtime_context_present(self):
        from dflash_mlx.runtime.context import build_runtime_context  # noqa: F401

    @pytest.mark.parametrize("removed_symbol", ["generate_dflash_once"])
    def test_removed_symbols_stay_removed(self, removed_symbol):
        """If upstream re-introduces a name we used to import, the
        emergency-fix path (drain stream_dflash_generate) becomes
        unnecessary and we can simplify."""
        from dflash_mlx import runtime

        assert not hasattr(runtime, removed_symbol), (
            f"dflash-mlx re-introduced '{removed_symbol}'. Check whether "
            f"the sync-drain workaround in DFlashEngine.generate "
            f"(commit 7ac0a5f) can revert to the simpler one-shot API."
        )


# =============================================================================
# mlx-vlm + mlx-lm MTP — pinned in pyproject.toml at mlx-vlm@f96138e.
# Consumers: omlx/speculative/vlm_mtp.py, omlx/patches/mlx_lm_mtp/*.py.
# =============================================================================


class TestMLXVLMMTPContract:
    """Lock the mlx-vlm MTP draft/verify entry points omlx wraps."""

    def test_mtp_rounds_present(self):
        from mlx_vlm.speculative.utils import _mtp_rounds  # noqa: F401

    def test_mtp_rounds_batch_present(self):
        from mlx_vlm.speculative.utils import _mtp_rounds_batch  # noqa: F401

    def test_mtp_rounds_signature(self):
        """The kwargs vlm_mtp.run_vlm_mtp_decode passes to single-row variant."""
        from mlx_vlm.speculative.utils import _mtp_rounds

        expected_kwargs = {
            "first_bonus",
            "max_tokens",
            "sampler",
            "draft_block_size",
            "token_dtype",
        }
        for kw in expected_kwargs:
            assert _accepts(_mtp_rounds, kw), (
                f"mlx-vlm _mtp_rounds no longer accepts kwarg '{kw}' — "
                f"omlx/speculative/vlm_mtp.py would TypeError on every "
                f"VLM MTP decode."
            )

    def test_mtp_rounds_batch_signature(self):
        """Batch variant — accepts everything _mtp_rounds does plus stop_check / eos."""
        from mlx_vlm.speculative.utils import _mtp_rounds_batch

        expected_kwargs = {
            "first_bonus",
            "max_tokens",
            "sampler",
            "draft_block_size",
            "token_dtype",
            "stop_check",
            "eos_token_ids",
        }
        for kw in expected_kwargs:
            assert _accepts(_mtp_rounds_batch, kw), (
                f"mlx-vlm _mtp_rounds_batch no longer accepts kwarg '{kw}' — "
                f"omlx/speculative/vlm_mtp.py batch path would TypeError."
            )

    def test_load_drafter_present(self):
        from mlx_vlm.speculative import load_drafter  # noqa: F401

    def test_load_model_present(self):
        from mlx_vlm.utils import load_model  # noqa: F401


class TestMLXLMMTPContract:
    """Lock the mlx-lm symbols omlx's MTP patches reach for."""

    def test_generation_batch_present(self):
        from mlx_lm.generate import GenerationBatch  # noqa: F401

    def test_kv_cache_present(self):
        from mlx_lm.models.cache import KVCache  # noqa: F401

    def test_arrays_cache_present(self):
        from mlx_lm.models.cache import ArraysCache  # noqa: F401

    def test_create_attention_mask_present(self):
        from mlx_lm.models.base import create_attention_mask  # noqa: F401

    def test_gated_delta_update_present(self):
        from mlx_lm.models.gated_delta import gated_delta_update  # noqa: F401

    def test_qwen3_5_module_importable(self):
        import mlx_lm.models.qwen3_5  # noqa: F401

    def test_qwen3_5_moe_module_importable(self):
        import mlx_lm.models.qwen3_5_moe  # noqa: F401


# =============================================================================
# mlx-vlm — pinned at @f96138e. Consumer: omlx/engine/vlm.py.
# Catches the kind of breakage commit 7ac0a5f exemplified, generalized to vlm.
# =============================================================================


class TestMLXVLMVLMContract:
    """Lock the mlx-vlm symbols VLMBatchedEngine reaches for at load."""

    @pytest.mark.parametrize(
        "dotted",
        [
            "mlx_vlm.utils.load_model",
            "mlx_vlm.utils.load_config",
        ],
    )
    def test_vlm_load_symbols_present(self, dotted):
        module_path, attr = dotted.rsplit(".", 1)
        m = importlib.import_module(module_path)
        assert hasattr(m, attr), (
            f"mlx-vlm moved/removed {dotted}. omlx/engine/vlm.py's "
            f"_load_vlm_sync expects this path."
        )
