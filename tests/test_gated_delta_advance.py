# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.gated_delta_advance.

The patch monkey-patches mlx-vlm ``Qwen3_5GatedDeltaNet.__call__`` to
call ``cache.advance(S)`` after the forward pass and to wrap the conv
state in ``mx.contiguous``. mlx-lm 0.31.3 already has both fixes
upstream; mlx-vlm e41cd25 still misses both.

mlx-vlm's ``qwen3_5_moe`` re-exports the same class object as
``Qwen3_5MoeGatedDeltaNet``, so a single class-level patch covers both
Qwen3.5 dense and Qwen3.5/3.6 A3B MoE — no need to patch the MoE
import separately.
"""

from __future__ import annotations

import pytest

from omlx.patches.gated_delta_advance import (
    _patched_classes,
    apply_gated_delta_advance_patch,
)


def test_apply_returns_true_when_at_least_one_target_present():
    """Patch should report success as long as one of the GatedDeltaNet
    classes is importable from the runtime."""
    assert apply_gated_delta_advance_patch() is True


def test_patch_is_idempotent():
    """Calling apply repeatedly must not double-wrap __call__."""
    apply_gated_delta_advance_patch()
    snapshot = set(_patched_classes)
    apply_gated_delta_advance_patch()
    assert _patched_classes == snapshot


def test_patch_accepts_model_arg_for_backward_compat():
    """Existing call sites pass a ``model`` argument; the new
    implementation must accept and ignore it without crashing."""
    fake_model = object()
    assert apply_gated_delta_advance_patch(fake_model) is True


def test_mlx_vlm_qwen3_5_class_is_patched():
    """The mlx-vlm dense class is the primary target of this patch."""
    apply_gated_delta_advance_patch()
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
    except ImportError:
        pytest.skip("mlx-vlm not installed in this environment")
    assert id(Qwen3_5GatedDeltaNet) in _patched_classes


def test_mlx_vlm_qwen3_5_moe_class_is_patched_via_reexport():
    """``qwen3_5_moe.language`` imports the same class object as
    ``Qwen3_5MoeGatedDeltaNet``, so patching the dense class covers
    both. Regression test: if mlx-vlm ever forks the MoE body into a
    separate class object (so they are no longer ``is``-identical),
    this test fails and the patch must be extended to also visit the
    MoE class explicitly — without it, Qwen3.5-35B-A3B linear-attention
    layers skip ``cache.advance(S)`` / ``cache[i] = …`` writes and
    ``ArraysCache.cache`` ends up ``[None, None]``, crashing
    ``BatchGenerator.extract_cache`` (mlx-lm cache.py:675) on the first
    batched decode with ``"'NoneType' object is not subscriptable"``.
    """
    apply_gated_delta_advance_patch()
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
        from mlx_vlm.models.qwen3_5_moe.language import (
            Qwen3_5MoeGatedDeltaNet,
        )
    except ImportError:
        pytest.skip("mlx-vlm qwen3_5_moe module not installed")
    # Today: same class object via re-export.
    assert Qwen3_5MoeGatedDeltaNet is Qwen3_5GatedDeltaNet
    # And by extension, patched.
    assert id(Qwen3_5MoeGatedDeltaNet) in _patched_classes


def test_patched_call_resolves_cache_from_positional_arg():
    """Regression test for the cache-positional-arg bug.

    The mlx-vlm decoder layer calls
    ``self.linear_attn(self.input_layernorm(x), mask, cache, gdn_sink=...)``
    with ``cache`` as the **third positional argument**, not a keyword.

    An earlier version of the wrapper resolved cache via
    ``kwargs.get("cache")`` only — which always returned ``None`` for
    positionally-passed cache, causing the body to skip every cache
    write. ``ArraysCache.cache`` then stayed ``[None, None]`` until
    mlx-lm's ``BatchGenerator.extract_cache`` crashed with
    ``"'NoneType' object is not subscriptable"`` on the first batched
    decode (surfaced to clients as
    ``"Cache corruption not recoverable after retries"``).

    The patched wrapper must extract cache from positional arg #3 (or
    arg #2 of the call after self+inputs binding). Verified here by a
    cache stub that records every ``__getitem__`` access — when the
    body sees the cache, it queries ``cache[0]`` near the top; when the
    body silently saw ``None`` (the bug), no access happened at all.
    """
    from omlx.patches.gated_delta_advance import _patch_class

    # Stubs provide just enough for the body to reach the first
    # ``cache[0]`` probe. The probe raises a sentinel to short-circuit
    # — we don't need real mlx kernels for the rest of the body.
    class _Arr:
        def __init__(self, shape, dtype="float32"):
            self.shape = shape
            self.dtype = dtype

        def reshape(self, *_a, **_kw):
            return _Arr((1, 1, 1, 1), self.dtype)

    class _Stub:
        head_v_dim = 8

        def in_proj_qkv(self, x):
            return _Arr(x.shape, x.dtype)

        def in_proj_z(self, x):
            return _Arr(x.shape, x.dtype)

        def in_proj_b(self, x):
            return _Arr(x.shape, x.dtype)

        def in_proj_a(self, x):
            return _Arr(x.shape, x.dtype)

        def __call__(self, inputs, mask=None, cache=None, gdn_sink=None):
            raise AssertionError(
                "original __call__ was invoked — fallback path leaked"
            )

    _patch_class(_Stub, "test._Stub")

    class _ProbeCache:
        """Records cache[0] access and short-circuits via sentinel."""

        class _Sentinel(Exception):
            pass

        def __init__(self):
            self.gets: list[int] = []

        def __getitem__(self, idx):
            self.gets.append(idx)
            raise _ProbeCache._Sentinel()

        def __setitem__(self, idx, value):
            pass

        def advance(self, n):
            pass

        def __bool__(self):
            return True

    cache = _ProbeCache()

    # cache as positional arg #2 of the bound call (mimics the layer's
    # ``self.linear_attn(inputs, mask, cache, gdn_sink=...)``). The
    # wrapper must resolve cache from args[1] before the body probes
    # ``cache[0] is not None``.
    with pytest.raises(_ProbeCache._Sentinel):
        _Stub()(_Arr((1, 5, 8)), None, cache)

    assert cache.gets == [0], (
        "cache[0] was not probed — wrapper didn't resolve cache from "
        "positional args (regression)"
    )


def test_body_failure_propagates():
    """Patch body must NOT swallow exceptions.

    Earlier the patch wrapped the body in ``try/except Exception:`` and
    fell back to the legacy mlx-vlm ``__call__`` on any failure. The
    legacy body is broken (missing ``cache.advance``, etc.) — that's
    why we patch it — so a silent fallback would mask real bugs in
    the patch and degrade every Qwen3.5 forward without surfacing.
    Confirm exceptions raised mid-body propagate to the caller.
    """
    from omlx.patches.gated_delta_advance import _patch_class

    class _Boom(Exception):
        pass

    class _Stub:
        # ``in_proj_qkv`` is the second statement in the body. Raising
        # here proves the wrapper does not catch and swallow.
        def in_proj_qkv(self, x):
            raise _Boom("intentional body failure")

        def __call__(self, inputs, mask=None, cache=None, gdn_sink=None):
            raise AssertionError(
                "original __call__ was invoked — fallback was not "
                "removed; the body's exception was silently swallowed"
            )

    _patch_class(_Stub, "test._Stub_boom")

    class _FakeInputs:
        shape = (1, 5, 8)

    with pytest.raises(_Boom):
        _Stub()(_FakeInputs())


# NOTE: The previous tests ``test_post_fix_failure_does_not_break_original_call``
# and ``test_patched_call_forwards_extra_kwargs`` verified the
# ``try/except → original_call(...)`` fallback that wrapped the body.
# That fallback has been deleted because it silently ran the broken
# legacy mlx-vlm body on any patch-body error, masking real bugs in
# the patch (e.g. the cache-positional-arg bug we just hit). Body
# failures now propagate to the caller — see
# ``test_body_failure_propagates`` above.
