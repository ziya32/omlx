# SPDX-License-Identifier: Apache-2.0
"""
Integration test for Scheduler._set_model_info_for_monitor() across all
discovered LM/VLM models in the configured model dirs.

This test catches the silent failure mode where the function is unable
to extract `num_hidden_layers` / `num_kv_heads` / `head_dim` from a
model's config and the MemoryMonitor stays uninitialized — which makes
estimate_prompt_kv_bytes() return 0 and silently disables the admission
guard's cached_overhead and the preflight memory check.

Each model is loaded for real (mlx-lm or mlx-vlm), passed through the
Scheduler constructor, then the resulting MemoryMonitor is inspected
to confirm the dimensions were populated to non-zero values.

Marked `slow` because each model load takes 10s-2min and consumes GPU
memory. To run only a subset:

    pytest tests/test_model_info_extraction.py -v -k "Qwen3.5-9B"

To run against a custom model dir:

    OMLX_TEST_MODEL_BASE_DIR=/path/to/custom/dir pytest tests/test_model_info_extraction.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from omlx.model_discovery import DiscoveredModel, discover_models_from_dirs
from omlx.scheduler import Scheduler, SchedulerConfig

logger = logging.getLogger(__name__)

# Model types this test exercises. Embedding/reranker/audio engines do
# not go through the same Scheduler path so they're skipped.
SCHEDULER_BACKED_TYPES = {"llm", "vlm"}

# Default omlx user-config base directory.
DEFAULT_BASE_DIR = Path.home() / ".omlx"


def _resolve_model_dirs() -> List[Path]:
    """Read configured model dirs from ~/.omlx/settings.json (or override).

    Returns:
        List of Path objects to model directories.

    Raises:
        pytest.skip if no settings file or no model dirs configured.
    """
    base_dir = Path(os.environ.get("OMLX_TEST_MODEL_BASE_DIR", str(DEFAULT_BASE_DIR)))
    settings_path = base_dir / "settings.json"

    if not settings_path.exists():
        pytest.skip(f"No settings file at {settings_path}")

    try:
        settings = json.loads(settings_path.read_text())
    except Exception as exc:
        pytest.skip(f"Failed to parse {settings_path}: {exc}")

    model_section = settings.get("model", {}) or {}
    raw_dirs = model_section.get("model_dirs") or model_section.get("model_dir")
    if raw_dirs is None:
        pytest.skip(f"No model_dirs configured in {settings_path}")
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]

    dirs: List[Path] = []
    for d in raw_dirs:
        p = Path(d).expanduser()
        if p.exists() and p.is_dir():
            dirs.append(p)
        else:
            logger.warning("Configured model dir does not exist, skipping: %s", p)

    if not dirs:
        pytest.skip("None of the configured model dirs exist on disk")

    return dirs


def _list_scheduler_backed_models() -> List[DiscoveredModel]:
    """Discover models and filter to LM/VLM only."""
    dirs = _resolve_model_dirs()
    discovered = discover_models_from_dirs(dirs)
    models = [
        info for info in discovered.values()
        if info.model_type in SCHEDULER_BACKED_TYPES
    ]
    if not models:
        pytest.skip("No LM/VLM models found in configured dirs")
    return sorted(models, key=lambda m: m.model_id)


def _load_model_for_scheduler(info: DiscoveredModel) -> Tuple[object, object]:
    """Load a model the same way the engine path does, returning
    `(model, tokenizer)` ready to pass to the Scheduler.

    For LLM models, returns the mlx-lm model + tokenizer.
    For VLM models, returns the VLMModelAdapter + tokenizer.

    On model-load failure (e.g. missing torchvision, weight format
    mismatch), pytest.skip is raised so the failure does not mask
    the actual extraction-correctness check this test exists for.
    """
    if info.model_type == "llm":
        from mlx_lm.utils import load as lm_load
        try:
            model, tokenizer = lm_load(info.model_path)
        except Exception as exc:
            pytest.skip(
                f"Could not load LLM {info.model_id} via mlx-lm: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )
        return model, tokenizer

    elif info.model_type == "vlm":
        # Apply the same patches the engine applies before vlm_load
        # so VLMs with video_preprocessor_config.json or Gemma4 vision
        # tower load correctly without torchvision.
        from omlx.engine.vlm import (
            _patch_gemma4_vision_tower,
            _patch_video_processor_bug,
        )
        from mlx_vlm.utils import load as vlm_load

        from omlx.models.vlm import VLMModelAdapter

        _patch_video_processor_bug()
        _patch_gemma4_vision_tower(None)

        try:
            vlm_model, processor = vlm_load(info.model_path)
        except Exception as exc:
            pytest.skip(
                f"Could not load VLM {info.model_id} via mlx-vlm: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )

        try:
            adapter = VLMModelAdapter(vlm_model=vlm_model, decode_model=None)
        except Exception as exc:
            pytest.skip(
                f"Could not wrap {info.model_id} in VLMModelAdapter: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )

        # Processor exposes a tokenizer attr for VLMs.
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        return adapter, tokenizer

    raise ValueError(f"Unsupported model_type: {info.model_type}")


def _release(*objs) -> None:
    """Drop references and run GC + MLX cache clear to free memory between models."""
    for o in objs:
        try:
            del o  # noqa: F821
        except Exception:
            pass
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Parametrized test
# ----------------------------------------------------------------------


def _generate_params() -> Iterator[Tuple[str, DiscoveredModel]]:
    """Yield (id, DiscoveredModel) pairs for parametrize."""
    try:
        models = _list_scheduler_backed_models()
    except Exception as exc:
        # Discovery itself failed — yield nothing so pytest skips cleanly.
        logger.warning("Model discovery failed: %s", exc)
        return
    for info in models:
        yield pytest.param(info, id=info.model_id)


@pytest.mark.slow
@pytest.mark.parametrize("model_info", list(_generate_params()))
def test_set_model_info_for_monitor_extracts_dimensions(
    model_info: DiscoveredModel,
) -> None:
    """For each discovered LM/VLM, the Scheduler must populate
    MemoryMonitor with non-zero num_layers / num_kv_heads / head_dim
    so that estimate_prompt_kv_bytes() returns > 0.

    A failure here indicates _set_model_info_for_monitor() can't find
    the transformer hyperparameters in this model's config — usually
    because the config nests them under a non-standard key (not
    `text_config`) or uses non-standard field names.
    """
    model = None
    tokenizer = None
    scheduler = None
    try:
        model, tokenizer = _load_model_for_scheduler(model_info)

        # Construct the scheduler — _set_model_info_for_monitor() runs
        # inside __init__ as part of the MemoryMonitor wiring.
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(),
        )

        assert scheduler.memory_monitor is not None, (
            f"Scheduler.memory_monitor is None for {model_info.model_id} — "
            f"MemoryMonitor instantiation failed in __init__"
        )

        mm = scheduler.memory_monitor
        # The three fields _set_model_info_for_monitor must populate
        # for estimate_prompt_kv_bytes() to return non-zero.
        assert mm._num_layers, (
            f"{model_info.model_id}: _num_layers={mm._num_layers!r} "
            f"(expected non-zero) — model_type={model_info.model_type}, "
            f"path={model_info.model_path}"
        )
        assert mm._num_kv_heads, (
            f"{model_info.model_id}: _num_kv_heads={mm._num_kv_heads!r} "
            f"(expected non-zero) — model_type={model_info.model_type}, "
            f"path={model_info.model_path}"
        )
        assert mm._head_dim, (
            f"{model_info.model_id}: _head_dim={mm._head_dim!r} "
            f"(expected non-zero) — model_type={model_info.model_type}, "
            f"path={model_info.model_path}"
        )

        # End-to-end smoke check: estimator must produce a positive
        # number for a non-trivial token count.
        kv_bytes_for_4k = mm.estimate_prompt_kv_bytes(4096)
        assert kv_bytes_for_4k > 0, (
            f"{model_info.model_id}: estimate_prompt_kv_bytes(4096)=0 "
            f"despite layers={mm._num_layers}, kv_heads={mm._num_kv_heads}, "
            f"head_dim={mm._head_dim}"
        )

        logger.info(
            "✓ %s: layers=%d kv_heads=%d head_dim=%d → "
            "kv_bytes/4k_tokens=%.1f MB",
            model_info.model_id,
            mm._num_layers,
            mm._num_kv_heads,
            mm._head_dim,
            kv_bytes_for_4k / 1024**2,
        )

    finally:
        # Release model + scheduler so the next parametrized run has GPU room.
        _release(scheduler, model, tokenizer)


# ----------------------------------------------------------------------
# Sanity test that the discovery itself works
# ----------------------------------------------------------------------


def test_discovery_finds_at_least_one_lm_or_vlm() -> None:
    """Sanity check: the configured model dir contains at least one
    LM or VLM model. Otherwise the parametrized test above is meaningless.
    """
    models = _list_scheduler_backed_models()
    assert len(models) > 0, (
        "No LM/VLM models discovered — check OMLX_TEST_MODEL_BASE_DIR or "
        "the model_dir setting in ~/.omlx/settings.json"
    )
    logger.info(
        "Discovered %d LM/VLM models: %s",
        len(models),
        ", ".join(m.model_id for m in models),
    )
