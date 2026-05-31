# SPDX-License-Identifier: Apache-2.0
"""Per-Linear / per-expert global output-scale wrapper for transcoded NVFP4 models.

NVIDIA modelopt NVFP4/FP8 carries a per-tensor FP32 scale (nvfp4
``weight_scale_2``, fp8 ``weight_scale``) on top of the block scale. It factors
out of ``y = Wx`` as a post-matmul multiplier:

    y = W x = (codes ⊙ block_scale) x · global = global · ((codes⊙block) x)

MLX's ``quantized_matmul`` / ``gather_qmm`` have no global-scale slot, so we
monkey-patch the quantized forward passes to multiply the result by a stored
``_omlx_global_scale``:

* ``nn.QuantizedLinear`` (attention q/k/v/o, linear_attn in/out, shared_expert,
  lm_head): a 0-d scalar.
* ``QuantizedSwitchLinear`` (MoE ``switch_mlp.{gate,up,down}_proj``): a per-expert
  ``[num_experts]`` vector, indexed by the routed expert ids. Applied per
  projection *before* SwiGLU, which is required (the nonlinearity means the
  per-expert globals do not factor through to the block's output).

Both patches are idempotent and **gated on the attribute** -- any model whose
modules lack ``_omlx_global_scale`` is untouched. Scales are produced by
``omlx.transcode_nvfp4`` into ``global_scales.safetensors`` and attached
post-load by :func:`attach_global_scales`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_APPLIED = False

# When set (by the MTP verify cycle, via set_force_eval), the output-scale
# wrappers force an ``mx.eval`` after each multiply. This bounds the Metal
# command buffer *within* the MTP speculative-verify backbone forward: that
# forward runs S=2 with gdn_sink capture, and its op count grows with the KV
# length; combined with the per-quantized-layer wrapper ops it eventually
# overflows the GPU command-buffer limit -> GPU Hang
# (kIOGPUCommandBufferCallbackErrorHang). Off by default so ordinary decode
# (one bounded forward per step) keeps a single fused buffer with no eval
# barrier. Only the MTP backbone forward toggles it on.
_force_eval_buffers = False


def set_force_eval(enabled: bool) -> None:
    """Toggle per-multiply ``mx.eval`` in the output-scale wrappers.

    The MTP verify cycle wraps its backbone forward with this so the wrapper
    breaks the command buffer per layer; everything else leaves it off.
    """
    global _force_eval_buffers
    _force_eval_buffers = bool(enabled)


def apply() -> bool:
    """Monkey-patch the quantized forwards to honor ``_omlx_global_scale``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    import mlx.core as mx
    import mlx.nn as nn

    # 1) QuantizedLinear -- scalar output scale
    QL = nn.QuantizedLinear
    if not getattr(QL, "_omlx_gscale_patched", False):
        _orig = QL.__call__

        def _ql_call(self, x):
            y = _orig(self, x)
            g = getattr(self, "_omlx_global_scale", None)
            if g is None:
                return y
            out = y * g
            if _force_eval_buffers:
                mx.async_eval(out)
            return out

        QL.__call__ = _ql_call
        QL._omlx_gscale_patched = True

    # 2) QuantizedSwitchLinear -- per-expert vector output scale, indexed by routing
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear as QSL
    except Exception as e:  # pragma: no cover
        QSL = None
        logger.debug("QuantizedSwitchLinear not importable for global-scale patch: %s", e)

    if QSL is not None and not getattr(QSL, "_omlx_gscale_patched", False):
        _orig_s = QSL.__call__

        def _qsl_call(self, x, indices, sorted_indices=False):
            y = _orig_s(self, x, indices, sorted_indices=sorted_indices)
            g = getattr(self, "_omlx_global_scale", None)
            if g is None:
                return y
            gi = g[indices]
            gi = gi.reshape(gi.shape + (1,) * (y.ndim - gi.ndim))
            out = y * gi
            if _force_eval_buffers:
                mx.async_eval(out)
            return out

        QSL.__call__ = _qsl_call
        QSL._omlx_gscale_patched = True

    _APPLIED = True
    logger.info("global-scale output wrapper applied (QuantizedLinear + QuantizedSwitchLinear)")
    return True


def attach_global_scales(model, model_dir) -> int:
    """Load ``global_scales.safetensors`` and set ``_omlx_global_scale`` on matching modules.

    Returns the number of modules scaled (0 if the side-car is absent -- a no-op
    for ordinary checkpoints).
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    p = Path(model_dir) / "omlx_meta" / "global_scales.safetensors"
    if not p.exists():
        p = Path(model_dir) / "global_scales.safetensors"   # legacy top-level
    if not p.exists():
        return 0
    gmap = mx.load(str(p))
    n = 0
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        g = gmap.get(path)
        if g is not None:
            module._omlx_global_scale = g
            n += 1
    if n:
        logger.info("attached %d global output-scales from %s", n, p.name)
    elif gmap:
        logger.warning(
            "global_scales.safetensors had %d entries but matched 0 modules "
            "(path mismatch?)", len(gmap)
        )
    return n
