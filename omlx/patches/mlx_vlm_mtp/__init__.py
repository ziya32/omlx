# SPDX-License-Identifier: Apache-2.0
"""Native MTP monkey-patches for mlx-vlm.

mlx-vlm ships its own ``Model.sanitize`` for VLM checkpoints
(Qwen3_5ForConditionalGeneration etc). The stock body unconditionally
strips ``mtp.*`` weights and only shifts a fixed set of backbone norm
keys by +1 — the MTP head's own norms (``mtp.norm.weight``,
``mtp.pre_fc_norm_*.weight`` and the per-block layernorms) miss the
shift, so MTP layers run with raw RMSNorm weights after quantization.

This package patches the affected mlx-vlm Model classes so the sanitize
output keeps ``mtp.*`` and applies the +1 norm shift to every MTP norm,
matching what mlx-lm PR 990 does on the LLM side.

Activation: oQ quantization invokes ``apply_mlx_vlm_mtp_patch`` before
building the VLM sanitizer. The patches are idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Process-wide flag: does the checkpoint about to load actually ship
# ``mtp.*`` weight tensors? Set per-load by
# ``omlx.utils.model_loading.maybe_apply_pre_load_patches`` (VLM path) just
# before ``mlx_vlm.utils.load`` runs; read by the runtime MTP ``__init__``
# patches to decide whether to attach ``MTPModule``.
#
# A checkpoint can declare ``mtp_num_hidden_layers > 0`` in config yet ship
# no ``mtp.*`` tensors (e.g. quantized before MTP-preserving sanitize landed,
# so stock mlx-vlm ``sanitize`` stripped them). Attaching the head for such a
# checkpoint makes strict ``load_weights`` fail with "Missing N parameters:
# language_model.mtp.*", and VLMBatchedEngine silently falls back to a
# text-only LLM — dropping vision. Defaults False so the head is never
# attached without a positive signal that its weights exist.
_MTP_WEIGHTS_PRESENT = False


def set_mtp_weights_present(present: bool) -> None:
    """Record whether the checkpoint being loaded ships ``mtp.*`` weights."""
    global _MTP_WEIGHTS_PRESENT
    _MTP_WEIGHTS_PRESENT = bool(present)


def mtp_weights_present() -> bool:
    """True iff the last ``set_mtp_weights_present`` call saw mtp.* weights."""
    return _MTP_WEIGHTS_PRESENT


def apply_mlx_vlm_mtp_patch() -> bool:
    """Apply the mlx-vlm MTP sanitize monkey-patches.

    Returns True on success (or if already applied), False if the
    sub-step refused (mlx-vlm not importable, etc.). Idempotency is
    handled by each sub-patcher via class-level flags — no module-wide
    cache flag, keeping behavior consistent with mlx_lm_mtp.
    """
    from . import qwen35_moe_vlm_model, qwen35_vlm_model

    if not qwen35_vlm_model.apply():
        logger.debug("Qwen3.5 VLM MTP sanitize patch did not apply")
    if not qwen35_moe_vlm_model.apply():
        logger.debug("Qwen3.5 MoE VLM MTP sanitize patch did not apply")

    return True


def apply_mlx_vlm_mtp_runtime_patch() -> bool:
    """Apply the mlx-vlm runtime MTP patches (attach MTPModule, mtp_forward).

    Distinct from ``apply_mlx_vlm_mtp_patch``: that one only patches
    ``sanitize`` for conversion-time MTP preservation. This one builds
    the runtime infrastructure so VLMBatchedEngine can actually invoke
    the MTP head at inference time.

    Covers both Qwen3.5-MoE (qwen3_5_moe) and dense Qwen3.5/3.6
    (qwen3_5) VLM families. Each sub-patch tracks its own ``_APPLIED``
    flag, so calling repeatedly is cheap once both have settled.
    Returns True if at least one sub-patch applied successfully — a
    given model only needs whichever matches its model_type.

    Should be called *before* ``mlx_vlm.utils.load(...)`` so the
    instantiated LanguageModel picks up the patched ``__init__``.
    """
    from . import qwen35_moe_vlm_runtime, qwen35_vlm_runtime

    moe_ok = qwen35_moe_vlm_runtime.apply()
    if not moe_ok:
        logger.debug("Qwen3.5-MoE VLM runtime MTP patch did not apply")
    dense_ok = qwen35_vlm_runtime.apply()
    if not dense_ok:
        logger.debug("Qwen3.5 (dense) VLM runtime MTP patch did not apply")

    return moe_ok or dense_ok
