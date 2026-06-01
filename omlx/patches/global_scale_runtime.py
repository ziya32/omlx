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

# Multi-token forwards past this many tokens use a *blocking* ``mx.eval`` to cap
# the Metal command buffer (a large prefill otherwise builds one buffer big
# enough to trip the GPU watchdog -> hang -> abort -> server crash; async_eval
# only submits, so big buffers still pile up on the completion queue and hang).
# Smaller multi-token forwards -- chiefly the S=2 MTP verify -- are safely bounded
# by the much cheaper non-blocking ``async_eval``, which keeps MTP fast. Single-
# token decode skips both. The S=2 verify has n_tokens=2, comfortably below this.
_BLOCKING_EVAL_TOKENS = 8


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
            # Cap the Metal command buffer (see _BLOCKING_EVAL_TOKENS): blocking
            # eval for large forwards (prefill), cheap async for the small MTP
            # verify, nothing for single-token decode. Gated on the global-scale
            # attr above, so non-transcoded models never reach here.
            n_tokens = x.size // x.shape[-1]
            if n_tokens > _BLOCKING_EVAL_TOKENS:
                mx.eval(out)
            elif n_tokens > 1:
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
            n_tokens = x.size // x.shape[-1]   # cap the buffer (see _ql_call)
            if n_tokens > _BLOCKING_EVAL_TOKENS:
                mx.eval(out)
            elif n_tokens > 1:
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
        return 0   # not a transcoded NVFP4 checkpoint -- no-op
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
