# NVFP4 Transcode Review

Commit: `6e60155 nvfp4: transcode NVIDIA modelopt NVFP4/FP8 checkpoints to MLX (lossless)`

Files:
- `omlx/transcode_nvfp4.py` — the transcoder (CLI + re-pack logic)
- `omlx/patches/global_scale_runtime.py` — post-matmul output-scale wrapper
- `omlx/cli.py` — `omlx transcode-nvfp4` command
- `omlx/engine/vlm.py` — VLM engine global-scale attach

---

## Issues

### 1. 🔴 Missing global scale in text-only engines

The global output-scale patch (`global_scale_runtime.apply()` + `attach_global_scales()`) is **only called from the VLM engine** (`vlm.py:1046-1056`). It is absent from:

- `omlx/engine/batched.py` (the main LLM engine)
- `omlx/models/llm.py` (standalone `MLXModel` loader)
- `omlx/models/reranker.py`

These all load models via `mlx_lm.load()` or `vlm_load()` but never apply the global scale patch or attach the side-car. If someone serves a transcoded NVFP4 model as text-only (e.g. `omlx serve --no-vision` or via the LLM API), the global scales will be silently ignored — producing garbage output with no error.

Fix: Move the apply+attach into a shared post-load hook in `model_loading.py` called by all engines, not just VLM.

---

### 2. 🔴 `_verify_written` only checks mxfp8, not NVFP4

Lines 402-428: the round-trip verification (`_verify_written`) only iterates over `sfp` (FP8 bases) and breaks after `n` checks. The NVFP4 tensors written to disk are **never verified** in the round-trip path.

`_verify_sample` checks the *source* tensors against MLX dequant, but no code verifies that the *written* NVFP4 weights survive `mx.save_safetensors` → `mx.load`.

Fix: Add a second loop in `_verify_written` that checks written NVFP4 tensors (verify `dequant(written) * global == ref`).

---

### 3. 🟡 Load-order dependency on global scale monkey-patch

`global_scale_runtime.apply()` monkey-patches `nn.QuantizedLinear.__call__` and `QuantizedSwitchLinear.__call__` **globally, at module level**. It's only called from the VLM engine. If a text-only engine loads the model first (before any VLM engine), the patch is never applied — and the model silently produces wrong outputs. Conversely, if VLM loads first, it patches the classes for all engines, but subsequent text-only loads still miss `attach_global_scales`.

Fix: Call `apply()` unconditionally at import time (in `__init__.py` or `engine_core.py`), not gated behind the VLM path. The `attach_global_scales` call should move to a shared post-load hook.

---

### 4. 🟡 Custom loader dispatch doesn't recognize modelopt

`maybe_load_custom_quantization` in `model_loading.py` has no branch for `quant_method == "modelopt"`. This means the NVFP4 model can't be loaded directly from HuggingFace — users must pre-transcode with `omlx transcode-nvfp4`. By design, but worth documenting prominently in the CLI help and any user-facing docs.

---

### 5. 🟡 `--limit-layers --keep-mtp` creates broken model

When `--keep-mtp` is true with `--limit-layers 2`, the full MTP head is kept while only 2 backbone layers are transcoded. The resulting model would have an MTP head expecting inputs from layers that don't exist — `load_weights` would fail with a shape mismatch. This is a dev-testing flag, not a production path, but worth guarding with an error or warning.

---

### 6. 🟢 `global_scales.safetensors` legacy fallback

`attach_global_scales` (line 97-98) falls back from `omlx_meta/global_scales.safetensors` to a top-level `global_scales.safetensors`. The transcoder always writes to the `omlx_meta/` subdir. If a user manually moves the file to the top level, the fallback would pick it up — but the warning on line 110-112 ("path mismatch?") would fire incorrectly if some keys match and others don't. Low risk, but the fallback path could mask bugs if the file is accidentally placed at the top level.

---

### 7. 🟢 Missing round-trip test

There is no automated test that transcodes a model (even with `--limit-layers 2`) and verifies:
- The output loads via both VLM and text-only engines
- The global scales are correctly attached
- The model produces coherent (non-garbage) tokens

The inline verification (`_verify_sample` / `_verify_written`) covers bit-exact re-pack but doesn't cover the end-to-end load+infer path.

---

## Summary

| # | Severity | What |
|---|---|---|
| 1 | 🔴 Critical | Global scale not applied in text-only engines → garbage output |
| 2 | 🔴 Medium | Written NVFP4 tensors never round-trip verified |
| 3 | 🟡 Medium | Load-order fragility in monkey-patch |
| 4 | 🟡 Low | No direct-load path; users must pre-transcode |
| 5 | 🟡 Low | `--limit-layers --keep-mtp` creates broken model |
| 6 | 🟢 Nit | Legacy fallback path in scale attach |
| 7 | 🟢 Nit | No end-to-end load+infer test |

Issues 1 and 3 are the ones that would cause actual runtime failures. Fix 1 first — move the global scale attach into a shared post-load hook. Fix 3 follows naturally from that refactor.
