# SPDX-License-Identifier: Apache-2.0
"""Loss-free transcode of NVIDIA modelopt NVFP4/FP8 checkpoints to MLX layout.

The element encodings already match MLX's quant modes (verified bit-exact on
real tensors, see ``_verify_sample`` and ``--verify-only``):

* NVFP4 (experts / shared_expert / lm_head): E2M1 nibbles + E4M3 per-16 block
  scale  ->  MLX ``nvfp4`` (group_size 16). The packed weight bytes and the
  E4M3 block-scale bytes copy **verbatim** (a uint32 reinterpret); MLX reads
  the same low-nibble-first / little-endian order.
* FP8 (attention q/k/v/o, linear_attn in/out proj): E4M3 bytes + per-tensor
  scale  ->  MLX ``mxfp8`` (group_size 32). The E4M3 bytes reinterpret to
  uint32; the E8M0 block scales are set to unit (byte 127 = 2**0).
* MTP / norms / router gate / embeddings (bf16): copied as-is.

modelopt's per-tensor FP32 scales -- nvfp4 ``weight_scale_2`` and fp8
``weight_scale`` -- factor cleanly out of ``y = Wx`` as a per-output scalar
(per-expert vector once experts are stacked), so they are emitted to a side-car
``global_scales.safetensors``. The loader applies them as a post-matmul output
scale (Phase 2: the QuantizedLinear / QuantizedSwitchLinear output-scale
patch). **No arithmetic is performed on weights here -- pure data movement.**

Usage:
    python -m omlx.transcode_nvfp4 <src_modelopt_dir> <dst_mlx_dir>
    python -m omlx.transcode_nvfp4 <src_modelopt_dir> --verify-only
    python -m omlx.transcode_nvfp4 <src> <dst> --limit-layers 2   # quick logic test
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_E8M0_UNIT = 127           # E8M0 byte for 2**0 (mxfp8 unit block scale)
_MAX_SHARD_BYTES = 5_000_000_000
_NVFP4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
_MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

# ---------------------------------------------------------------------------
# FP8 / FP4 decode LUTs (verification only -- the transcode never decodes).
# ---------------------------------------------------------------------------

def _e4m3(b: int) -> float:
    s = (b >> 7) & 1; e = (b >> 3) & 0xF; m = b & 7
    sg = -1.0 if s else 1.0
    if e == 0xF and m == 7:
        return float("nan")
    return sg * ((m / 8) * 2 ** -6 if e == 0 else (1 + m / 8) * 2 ** (e - 7))


def _e2m1(n: int) -> float:
    s = (n >> 3) & 1; e = (n >> 1) & 3; m = n & 1
    sg = -1.0 if s else 1.0
    return sg * ((m / 2) if e == 0 else (1 + m / 2) * 2 ** (e - 1))


_E4M3_LUT = np.array([_e4m3(b) for b in range(256)], dtype=np.float32)
_E2M1_LUT = np.array([_e2m1(n) for n in range(16)], dtype=np.float32)

# ---------------------------------------------------------------------------
# safetensors header IO
# ---------------------------------------------------------------------------

def _read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _index(src: str) -> dict:
    out = {}
    for p in sorted(glob.glob(f"{src}/*.safetensors")):
        hdr, base = _read_header(p)
        for k, m in hdr.items():
            if k != "__metadata__":
                out[k] = (p, m, base)
    return out


def _raw(entry, n=None) -> bytes:
    p, m, base = entry
    s, e = m["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + s)
        return f.read((e - s) if n is None else min(n, e - s))


def _f32(b: bytes):
    return struct.unpack("<" + "f" * (len(b) // 4), b)


# ---------------------------------------------------------------------------
# name remap: modelopt (HF wrapper) -> mlx-vlm tree
# ---------------------------------------------------------------------------

def _remap(name: str) -> str:
    if "model.language_model" in name:
        name = name.replace("model.language_model", "language_model.model")
    elif "model.visual" in name:
        name = name.replace("model.visual", "vision_tower")
    elif name.startswith("lm_head"):
        name = "language_model." + name
    elif name.startswith("mtp."):
        name = "language_model." + name
    return name


# ---------------------------------------------------------------------------
# classification + expert grouping
# ---------------------------------------------------------------------------

def classify(idx: dict):
    nvfp4, fp8, passthrough = set(), set(), set()
    consumed = set()
    for k in idx:
        if k.endswith(".weight"):
            b = k[:-7]
            if b + ".weight_scale_2" in idx:
                nvfp4.add(b)
                consumed |= {k, b + ".weight_scale", b + ".weight_scale_2", b + ".input_scale"}
            elif b + ".weight_scale" in idx and idx[k][1]["dtype"] == "F8_E4M3":
                fp8.add(b)
                consumed |= {k, b + ".weight_scale", b + ".input_scale"}
    for k in idx:
        if k not in consumed and not k.endswith(".input_scale"):
            passthrough.add(k)
    return nvfp4, fp8, passthrough


_PROJS = ("gate_proj", "up_proj", "down_proj")


def group_experts(bases: set):
    experts = defaultdict(dict)
    standalone = set()
    for b in bases:
        if ".mlp.experts." in b and b.split(".")[-1] in _PROJS:
            head, rest = b.split(".mlp.experts.")
            idx_str, proj = rest.split(".")
            experts[(head, proj)][int(idx_str)] = b
        else:
            standalone.add(b)
    return experts, standalone


# ---------------------------------------------------------------------------
# verification (no writing)
# ---------------------------------------------------------------------------

def _verify_sample(idx: dict, nvfp4, fp8, n_each=3) -> bool:
    import mlx.core as mx
    ok = True

    def chk_nvfp4(b):
        wm = idx[b + ".weight"][1]; out, in2 = wm["shape"]; IN = in2 * 2
        W = np.frombuffer(_raw(idx[b + ".weight"]), np.uint8).reshape(out, in2)
        S = np.frombuffer(_raw(idx[b + ".weight_scale"]), np.uint8).reshape(out, IN // 16)
        g = _f32(_raw(idx[b + ".weight_scale_2"], 4))[0]
        c = np.empty((out, IN), np.uint8); c[:, 0::2] = W & 0xF; c[:, 1::2] = (W >> 4) & 0xF
        ref = _E2M1_LUT[c] * np.repeat(_E4M3_LUT[S], 16, axis=1) * g
        deq = np.array(mx.dequantize(mx.array(W.view(np.uint32)), mx.array(S),
                                     group_size=16, bits=4, mode="nvfp4").astype(mx.float32))
        return np.array_equal(deq * g, ref.astype(np.float32))

    def chk_fp8(b):
        o, i = idx[b + ".weight"][1]["shape"]
        W = np.frombuffer(_raw(idx[b + ".weight"]), np.uint8).reshape(o, i)
        ps = _f32(_raw(idx[b + ".weight_scale"], 4))[0]
        ref = _E4M3_LUT[W] * ps
        sc = mx.zeros((o, i // 32), dtype=mx.uint8) + _E8M0_UNIT
        deq = np.array(mx.dequantize(mx.array(W.view(np.uint32)), sc,
                                     group_size=32, bits=8, mode="mxfp8").astype(mx.float32))
        return np.array_equal(deq * ps, ref.astype(np.float32))

    for b in sorted(nvfp4)[:n_each]:
        r = chk_nvfp4(b); ok &= r; print(f"  [nvfp4] {b.split('.')[-1]:12s} bit-exact={r}")
    for b in sorted(fp8)[:n_each]:
        r = chk_fp8(b); ok &= r; print(f"  [mxfp8] {b.split('.')[-1]:12s} bit-exact={r}")
    return ok


# ---------------------------------------------------------------------------
# re-pack (zero-copy reinterpret) -> mx arrays
# ---------------------------------------------------------------------------

def _nvfp4_weight(entry):
    import mlx.core as mx
    out, in2 = entry[1]["shape"]
    w = np.frombuffer(_raw(entry), np.uint8).reshape(out, in2)
    return mx.array(np.ascontiguousarray(w).view(np.uint32))      # [out, in/8]


def _nvfp4_scales(entry):
    import mlx.core as mx
    out, nb = entry[1]["shape"]
    return mx.array(np.frombuffer(_raw(entry), np.uint8).reshape(out, nb).copy())


def _fp8_weight(entry):
    import mlx.core as mx
    o, i = entry[1]["shape"]
    w = np.frombuffer(_raw(entry), np.uint8).reshape(o, i)
    return mx.array(np.ascontiguousarray(w).view(np.uint32))      # [out, in/4]


# RMSNorm weights that mlx-vlm's sanitize shifts by +1 on raw checkpoints.
# The last three are MTP-head norms (only present with --keep-mtp); harmless
# suffixes otherwise.
_NORM_SFX = (".input_layernorm.weight", ".post_attention_layernorm.weight",
             "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
             ".pre_fc_norm_hidden.weight", ".pre_fc_norm_embedding.weight",
             "mtp.norm.weight")


def _passthrough_transform(rk, t):
    """Apply the sanitize transforms mlx-vlm does to RAW (bf16) tensors.

    We write ``format=mlx`` (which skips sanitize for our pre-packed quantized
    tensors), so the un-quantized passthrough tensors must get their transforms
    here, matching mlx-vlm/oQ on a raw checkpoint:
      * vision ``patch_embed.proj.weight``: PyTorch conv [O,I,T,H,W] -> MLX
        channels-last [O,T,H,W,I]  (qwen3_vl VisionModel.sanitize)
      * ``linear_attn.conv1d.weight``: ``moveaxis(2, 1)``
      * RMSNorm weights: ``+ 1.0``  (Qwen3.5 stores gamma-1)
    """
    if rk.endswith("patch_embed.proj.weight") and t.ndim == 5:
        return t.transpose(0, 2, 3, 4, 1)
    if "conv1d.weight" in rk and t.shape[-1] != 1:
        return t.moveaxis(2, 1)
    if t.ndim == 1 and rk.endswith(_NORM_SFX):
        return t + 1.0
    return t


# ---------------------------------------------------------------------------
# transcode
# ---------------------------------------------------------------------------

def transcode(src: str, dst: str, limit_layers: int | None = None,
              keep_mtp: bool = False, mtp_quant: str = "mxfp8") -> None:
    """Transcode modelopt NVFP4/FP8 -> MLX.

    ``mtp_quant`` controls how the (bf16, un-quantized-by-modelopt) MTP head is
    written when ``keep_mtp``: ``"mxfp8"`` (default) / ``"nvfp4"`` quantize its
    Linears so the draft-head forward is 2-4x cheaper (it dominates MTP cost
    otherwise -- the head is a *draft*, verified by the backbone, so the small
    quant error only trades a little accept rate for speed); ``"bf16"`` keeps it
    lossless. ``mx.quantize`` is self-contained (no global scale), so quantized
    MTP weights need no output-scale wrapper.
    """
    import mlx.core as mx
    mx.set_default_device(mx.cpu)   # CPU-only: don't contend with a running `omlx serve`

    idx = _index(src)
    nvfp4, fp8, passthrough = classify(idx)
    experts, standalone = group_experts(nvfp4)
    dst_p = Path(dst); dst_p.mkdir(parents=True, exist_ok=True)

    def keep(name: str) -> bool:
        if limit_layers is None:
            return True
        import re
        m = re.search(r"layers\.(\d+)", name)
        return (m is None) or (int(m.group(1)) < limit_layers)

    out_tensors = {}          # remapped_name -> mx.array
    globals_map = {}          # remapped_base -> mx.array (scalar or [E])
    quant_cfg = {}            # remapped_path -> per-layer quant dict
    weight_map = {}
    shard_i = 0
    _load_cache = {}

    def _bf16(name):          # passthrough loader (mx.load is lazy)
        p = idx[name][0]
        if p not in _load_cache:
            _load_cache[p] = mx.load(p)
        return _load_cache[p][name]

    def flush(force=False):
        nonlocal shard_i, out_tensors
        cur = sum(v.nbytes for v in out_tensors.values())
        if not out_tensors or (cur < _MAX_SHARD_BYTES and not force):
            return
        shard_i += 1
        fn = f"model-{shard_i:05d}-of-PLACEHOLDER.safetensors"
        mx.save_safetensors(str(dst_p / fn), out_tensors, metadata={"format": "mlx"})
        for k in out_tensors:
            weight_map[k] = fn
        out_tensors = {}
        mx.synchronize(); mx.clear_cache()

    # MTP-head Linears to quantize (draft head). switch_mlp is handled at its
    # fused-split site; router gate / shared_expert_gate / norms stay bf16.
    _MTP_LINEAR_SFX = (
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight", "mtp.fc.weight",
    )
    _mtp_cfg = dict(_NVFP4) if mtp_quant == "nvfp4" else dict(_MXFP8)

    def _emit_mtp_linear(base, w):
        """Write one MTP Linear at ``base`` (no trailing ``.weight``).

        bf16 passthrough, or ``mx.quantize`` to mxfp8/nvfp4 (self-contained --
        no global scale, so no wrapper entry) + a per-path quant_cfg override.
        """
        if mtp_quant == "bf16":
            out_tensors[base + ".weight"] = w
            return
        packed, scales = mx.quantize(
            w, group_size=_mtp_cfg["group_size"],
            bits=_mtp_cfg["bits"], mode=_mtp_cfg["mode"],
        )
        out_tensors[base + ".weight"] = packed
        out_tensors[base + ".scales"] = scales
        quant_cfg[base] = dict(_mtp_cfg)

    t0 = time.monotonic()
    # 1) passthrough (bf16)
    for k in sorted(passthrough):
        if not keep(k):
            continue
        if not keep_mtp and (k.startswith("mtp.") or ".mtp." in k):
            continue   # drop MTP head for now (bf16 MoE; clean follow-up)
        if "position_ids" in k:
            continue   # buffer, dropped by vision sanitize
        rk = _remap(k)
        is_mtp = rk.startswith("language_model.mtp.")
        # MTP MoE experts ship fused + stacked (bf16): experts.gate_up_proj
        # [E, 2*moe_inter, hidden] and experts.down_proj [E, hidden, moe_inter].
        # mlx-vlm's SwitchGLU wants switch_mlp.{gate,up,down}_proj; mlx-vlm
        # skips Model.sanitize for format=mlx, so split/rename here (mirrors
        # qwen35_moe_vlm_runtime._unfuse_layer_experts), then quantize per
        # ``mtp_quant`` (the 256-expert switch_mlp dominates draft-head cost).
        if is_mtp and rk.endswith(".mlp.experts.gate_up_proj"):
            base = rk[: -len("experts.gate_up_proj")] + "switch_mlp"
            gate_w, up_w = mx.split(_bf16(k), 2, axis=-2)
            _emit_mtp_linear(base + ".gate_proj", gate_w)
            _emit_mtp_linear(base + ".up_proj", up_w)
            flush(); continue
        if is_mtp and rk.endswith(".mlp.experts.down_proj"):
            base = rk[: -len("experts.down_proj")] + "switch_mlp"
            _emit_mtp_linear(base + ".down_proj", _bf16(k))
            flush(); continue
        if is_mtp and mtp_quant != "bf16" and rk.endswith(_MTP_LINEAR_SFX):
            _emit_mtp_linear(rk[: -len(".weight")], _bf16(k)); flush(); continue
        out_tensors[rk] = _passthrough_transform(rk, _bf16(k)); flush()
    # 2) fp8 attention -> mxfp8
    for b in sorted(fp8):
        if not keep(b):
            continue
        rb = _remap(b)
        out_tensors[rb + ".weight"] = _fp8_weight(idx[b + ".weight"])
        o, i = idx[b + ".weight"][1]["shape"]
        out_tensors[rb + ".scales"] = mx.zeros((o, i // 32), dtype=mx.uint8) + _E8M0_UNIT
        globals_map[rb] = mx.array(_f32(_raw(idx[b + ".weight_scale"], 4))[0])   # 0-d scalar
        quant_cfg[rb] = dict(_MXFP8); flush()
    # 3) standalone nvfp4 (shared_expert, lm_head)
    for b in sorted(standalone):
        if not keep(b):
            continue
        rb = _remap(b)
        out_tensors[rb + ".weight"] = _nvfp4_weight(idx[b + ".weight"])
        out_tensors[rb + ".scales"] = _nvfp4_scales(idx[b + ".weight_scale"])
        globals_map[rb] = mx.array(_f32(_raw(idx[b + ".weight_scale_2"], 4))[0])   # 0-d scalar
        quant_cfg[rb] = dict(_NVFP4); flush()
    # 4) experts -> stacked switch_mlp [E, out, in]
    for (head, proj), emap in sorted(experts.items()):
        if not keep(head):
            continue
        E = max(emap) + 1
        ws = [_nvfp4_weight(idx[emap[e] + ".weight"]) for e in range(E)]
        ss = [_nvfp4_scales(idx[emap[e] + ".weight_scale"]) for e in range(E)]
        gs = [_f32(_raw(idx[emap[e] + ".weight_scale_2"], 4))[0] for e in range(E)]
        rb = _remap(f"{head}.mlp.switch_mlp.{proj}")
        out_tensors[rb + ".weight"] = mx.stack(ws)
        out_tensors[rb + ".scales"] = mx.stack(ss)
        globals_map[rb] = mx.array(np.array(gs, np.float32))      # [E]
        quant_cfg[rb] = dict(_NVFP4); flush()
    flush(force=True)

    # index.json (fix the PLACEHOLDER -> NNNNN-of-MMMMM)
    total = shard_i
    fixed = {}
    rename = {}
    for k, fn in weight_map.items():
        i = int(fn.split("-")[1])
        nf = f"model-{i:05d}-of-{total:05d}.safetensors"
        rename[fn] = nf; fixed[k] = nf
    for old, new in set((o, n) for o, n in rename.items()):
        src_f = dst_p / old
        if src_f.exists():
            src_f.rename(dst_p / new)
    index = {"metadata": {"total_size": sum(_p.stat().st_size for _p in dst_p.glob("*.safetensors"))},
             "weight_map": fixed}
    (dst_p / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # global_scales side-car -- in a subdir so mlx-vlm's top-level *.safetensors
    # glob does NOT load it as model weights.
    meta_dir = dst_p / "omlx_meta"; meta_dir.mkdir(exist_ok=True)
    mx.save_safetensors(str(meta_dir / "global_scales.safetensors"), globals_map,
                        metadata={"format": "mlx", "omlx_global_scales": "1"})

    # config.json: base nvfp4 + per-attention mxfp8 overrides; drop modelopt block
    cfg = json.loads((Path(src) / "config.json").read_text())
    cfg.pop("quantization_config", None)
    # modelopt names the vision tower "<arch>_vision"; mlx-vlm's VisionModel
    # only accepts the base arch string (qwen3_vl / qwen3_5 / qwen3_5_moe).
    vc = cfg.get("vision_config")
    if isinstance(vc, dict) and isinstance(vc.get("model_type"), str) \
            and vc["model_type"].endswith("_vision"):
        vc["model_type"] = vc["model_type"][: -len("_vision")]
    if not keep_mtp:
        for c in (cfg, cfg.get("text_config", {})):
            if isinstance(c, dict) and "mtp_num_hidden_layers" in c:
                c["mtp_num_hidden_layers"] = 0
    quantization = dict(_NVFP4)               # base = nvfp4 g16
    quantization.update(quant_cfg)            # per-layer (mxfp8 attn, explicit nvfp4 too)
    cfg["quantization"] = quantization
    (dst_p / "config.json").write_text(json.dumps(cfg, indent=2))

    # aux files (tokenizer / processor / templates) -- skip ones we generate
    skip = {"config.json", "model.safetensors.index.json"}
    for f in Path(src).iterdir():
        if f.is_file() and f.suffix in {".json", ".txt", ".jinja", ".model"} \
                and f.name not in skip and not f.name.endswith(".safetensors"):
            shutil.copy(f, dst_p / f.name)

    dt = time.monotonic() - t0
    print(f"wrote {total} shards + global_scales.safetensors + config.json in {dt:.0f}s")
    print(f"  quant layers: {len(quant_cfg)} per-layer entries | globals: {len(globals_map)}")


def _verify_written(dst: str, src: str, n=4) -> bool:
    """Reload a few WRITTEN tensors and confirm dequant(view) * global == modelopt ref."""
    import mlx.core as mx
    sidx = _index(src)
    didx = _index(dst)
    gmap = mx.load(str(Path(dst) / "omlx_meta" / "global_scales.safetensors"))
    snv, sfp, _ = classify(sidx)
    ok = True
    checked = 0
    for b in sorted(sfp):
        rb = _remap(b)
        if rb + ".weight" not in didx:
            continue
        o, i = sidx[b + ".weight"][1]["shape"]
        Wsrc = np.frombuffer(_raw(sidx[b + ".weight"]), np.uint8).reshape(o, i)
        ps = _f32(_raw(sidx[b + ".weight_scale"], 4))[0]
        ref = _E4M3_LUT[Wsrc] * ps
        wq = np.frombuffer(_raw(didx[rb + ".weight"]), np.uint32).reshape(o, i // 4)
        sc = np.frombuffer(_raw(didx[rb + ".scales"]), np.uint8).reshape(o, i // 32)
        deq = np.array(mx.dequantize(mx.array(wq), mx.array(sc), group_size=32, bits=8, mode="mxfp8").astype(mx.float32))
        g = float(np.array(gmap[rb]).reshape(-1)[0])
        r = np.array_equal(deq * g, ref.astype(np.float32))
        ok &= r; checked += 1
        print(f"  [written mxfp8] {rb.split('.')[-1]:10s} bit-exact={r}")
        if checked >= n:
            break
    return ok


def main():
    ap = argparse.ArgumentParser(description="Transcode modelopt NVFP4/FP8 -> MLX (lossless re-pack)")
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--limit-layers", type=int, default=None, help="only transcode layers < N (logic test)")
    ap.add_argument("--keep-mtp", action="store_true", help="keep the MTP head (default: drop it)")
    ap.add_argument("--mtp-quant", choices=["mxfp8", "nvfp4", "bf16"], default="mxfp8",
                    help="how to write the MTP head with --keep-mtp (default: mxfp8; "
                         "draft head -> quantizing it is the MTP speedup)")
    args = ap.parse_args()

    idx = _index(args.src)
    nvfp4, fp8, passthrough = classify(idx)
    experts, standalone = group_experts(nvfp4)
    print(f"source tensors: {len(idx)}")
    print(f"  nvfp4 bases:  {len(nvfp4)}  -> {len(experts)} switch_mlp groups + {len(standalone)} standalone")
    print(f"  fp8 bases:    {len(fp8)} | passthrough: {len(passthrough)}")

    print("\nverifying re-pack is bit-exact (sample)...")
    if not _verify_sample(idx, nvfp4, fp8):
        raise SystemExit("re-pack verification FAILED")
    if args.verify_only or not args.dst:
        return

    print(f"\ntranscoding -> {args.dst}")
    transcode(args.src, args.dst, limit_layers=args.limit_layers,
              keep_mtp=args.keep_mtp, mtp_quant=args.mtp_quant)
    print("\nverifying WRITTEN checkpoint round-trips...")
    print("  ALL WRITTEN BIT-EXACT:", _verify_written(args.dst, args.src))


if __name__ == "__main__":
    main()
