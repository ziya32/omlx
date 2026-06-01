"""End-to-end-ish tests for the NVFP4 transcode path (review issue #7).

These are fast CPU unit tests -- no real model, no GPU. A synthetic modelopt
checkpoint (random bytes in the exact nvfp4 / fp8 / bf16 layout modelopt emits)
is transcoded, then we assert:

  * every written quant format round-trips bit-exact (re-pack correctness),
  * experts stack into switch_mlp, passthrough sanitize transforms apply,
  * the global-scale side-car + config quantization entries are correct,
  * --keep-mtp quantizes the draft head (switch_mlp gets .scales),
  * the global-scale output wrapper scales QuantizedLinear / QuantizedSwitchLinear
    forwards correctly, and attach/no-op behave.

The full mlx-vlm engine load+infer on a *real* transcoded checkpoint is covered
by manual live tests (it needs a multi-GB source model + GPU); here we cover the
transcode logic + the scale mechanism that those live runs exercise.
"""

import json
import os
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

H = 32          # hidden
I = 32          # moe intermediate (divisible by 16 and 32)
V = 64          # vocab
E = 2           # experts


# ---------------------------------------------------------------------------
# Synthetic modelopt checkpoint builder
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors):
    """tensors: dict name -> (dtype_str, shape, raw_bytes). Manual writer so we
    can emit modelopt's F8_E4M3 / U8 dtypes that numpy can't represent."""
    header, off = {}, 0
    for name, (dt, shape, data) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [off, off + len(data)]}
        off += len(data)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _name, (_dt, _shape, data) in tensors.items():
            f.write(data)


def _bf16_bytes(arr):
    """float32 ndarray -> truncated-bf16 little-endian bytes (valid BF16)."""
    u32 = np.ascontiguousarray(arr, dtype="<f4").view(np.uint32)
    return (u32 >> 16).astype("<u2").tobytes()


def _e4m3(n, rng):
    """n finite E4M3 bytes (0..0x77 avoids the e=0xF NaN region entirely)."""
    return rng.integers(0, 0x78, size=n, dtype=np.uint8).tobytes()


def _u8(n, rng):
    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()


def _f32_pos(rng):
    return np.array([rng.uniform(0.1, 2.0)], dtype="<f4").tobytes()


def _bf16(shape, rng):
    return _bf16_bytes(rng.standard_normal(int(np.prod(shape))).astype("f4") * 0.05)


def _make_modelopt_checkpoint(path, keep_mtp=True):
    """Write a tiny but structurally-faithful modelopt NVFP4 checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    quant = {}   # shard 1: nvfp4 + fp8 (read via _raw, never mx.load'd)
    bf16 = {}    # shard 2: bf16 passthrough (read via mx.load)

    def add_nvfp4(base, out, in_):
        quant[f"{base}.weight"] = ("U8", (out, in_ // 2), _u8(out * in_ // 2, rng))
        quant[f"{base}.weight_scale"] = ("F8_E4M3", (out, in_ // 16), _e4m3(out * in_ // 16, rng))
        quant[f"{base}.weight_scale_2"] = ("F32", (), _f32_pos(rng))

    def add_fp8(base, out, in_):
        quant[f"{base}.weight"] = ("F8_E4M3", (out, in_), _e4m3(out * in_, rng))
        quant[f"{base}.weight_scale"] = ("F32", (), _f32_pos(rng))

    LP = "model.language_model.layers.0"
    # backbone MoE experts (nvfp4)
    for e in range(E):
        add_nvfp4(f"{LP}.mlp.experts.{e}.gate_proj", I, H)
        add_nvfp4(f"{LP}.mlp.experts.{e}.up_proj", I, H)
        add_nvfp4(f"{LP}.mlp.experts.{e}.down_proj", H, I)
    # shared_expert (nvfp4 standalone) + lm_head (nvfp4 standalone)
    add_nvfp4(f"{LP}.mlp.shared_expert.gate_proj", I, H)
    add_nvfp4(f"{LP}.mlp.shared_expert.up_proj", I, H)
    add_nvfp4(f"{LP}.mlp.shared_expert.down_proj", H, I)
    add_nvfp4("lm_head", V, H)
    # attention (fp8)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        add_fp8(f"{LP}.self_attn.{p}", H, H)

    # bf16 passthrough
    bf16[f"{LP}.input_layernorm.weight"] = ("BF16", (H,), _bf16((H,), rng))
    bf16[f"{LP}.post_attention_layernorm.weight"] = ("BF16", (H,), _bf16((H,), rng))
    bf16[f"{LP}.self_attn.q_norm.weight"] = ("BF16", (H,), _bf16((H,), rng))
    bf16[f"{LP}.self_attn.k_norm.weight"] = ("BF16", (H,), _bf16((H,), rng))
    bf16["model.language_model.norm.weight"] = ("BF16", (H,), _bf16((H,), rng))
    bf16["model.language_model.embed_tokens.weight"] = ("BF16", (V, H), _bf16((V, H), rng))
    bf16[f"{LP}.mlp.gate.weight"] = ("BF16", (E, H), _bf16((E, H), rng))  # router (stays bf16)
    # linear_attn conv1d: shape[-1] != 1 -> moveaxis(2, 1)
    bf16[f"{LP}.linear_attn.conv1d.weight"] = ("BF16", (H, 1, 4), _bf16((H, 1, 4), rng))
    # vision patch_embed conv3d: [out, in=3, T, Hk, Wk] -> transpose(0,2,3,4,1)
    bf16["model.visual.patch_embed.proj.weight"] = ("BF16", (8, 3, 2, 4, 4), _bf16((8, 3, 2, 4, 4), rng))
    bf16["model.visual.patch_embed.position_ids"] = ("BF16", (4,), _bf16((4,), rng))  # dropped

    if keep_mtp:
        bf16["mtp.fc.weight"] = ("BF16", (H, 2 * H), _bf16((H, 2 * H), rng))
        bf16["mtp.norm.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.pre_fc_norm_hidden.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.pre_fc_norm_embedding.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.layers.0.input_layernorm.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.layers.0.post_attention_layernorm.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.layers.0.self_attn.q_proj.weight"] = ("BF16", (H, H), _bf16((H, H), rng))
        bf16["mtp.layers.0.self_attn.k_proj.weight"] = ("BF16", (H, H), _bf16((H, H), rng))
        bf16["mtp.layers.0.self_attn.v_proj.weight"] = ("BF16", (H, H), _bf16((H, H), rng))
        bf16["mtp.layers.0.self_attn.o_proj.weight"] = ("BF16", (H, H), _bf16((H, H), rng))
        bf16["mtp.layers.0.self_attn.q_norm.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.layers.0.self_attn.k_norm.weight"] = ("BF16", (H,), _bf16((H,), rng))
        bf16["mtp.layers.0.mlp.gate.weight"] = ("BF16", (E, H), _bf16((E, H), rng))
        # MoE experts: fused + stacked (gate_up_proj [E, 2I, H], down_proj [E, H, I])
        bf16["mtp.layers.0.mlp.experts.gate_up_proj"] = ("BF16", (E, 2 * I, H), _bf16((E, 2 * I, H), rng))
        bf16["mtp.layers.0.mlp.experts.down_proj"] = ("BF16", (E, H, I), _bf16((E, H, I), rng))
        bf16["mtp.layers.0.mlp.shared_expert.gate_proj.weight"] = ("BF16", (I, H), _bf16((I, H), rng))
        bf16["mtp.layers.0.mlp.shared_expert.up_proj.weight"] = ("BF16", (I, H), _bf16((I, H), rng))
        bf16["mtp.layers.0.mlp.shared_expert.down_proj.weight"] = ("BF16", (H, I), _bf16((H, I), rng))
        bf16["mtp.layers.0.mlp.shared_expert_gate.weight"] = ("BF16", (1, H), _bf16((1, H), rng))

    _write_safetensors(path / "model-00001-of-00002.safetensors", quant)
    _write_safetensors(path / "model-00002-of-00002.safetensors", bf16)
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {"num_hidden_layers": 1, "num_experts": E,
                        "mtp_num_hidden_layers": 1 if keep_mtp else 0},
        "vision_config": {"model_type": "qwen3_5_moe_vision"},  # exercises the _vision strip
        "quantization_config": {"quant_method": "modelopt"},     # transcode pops this
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


# ---------------------------------------------------------------------------
# Transcode tests
# ---------------------------------------------------------------------------

def test_transcode_repack_bit_exact(tmp_path):
    from omlx.transcode_nvfp4 import transcode, _verify_written
    src = _make_modelopt_checkpoint(tmp_path / "src", keep_mtp=True)
    dst = tmp_path / "dst"
    transcode(str(src), str(dst), keep_mtp=True, mtp_quant="mxfp8")
    # every written quant format (mxfp8 + nvfp4 standalone + nvfp4 experts) round-trips
    assert _verify_written(str(dst), str(src), n=8) is True


def test_transcode_structure(tmp_path):
    from omlx.transcode_nvfp4 import transcode
    src = _make_modelopt_checkpoint(tmp_path / "src", keep_mtp=True)
    dst = tmp_path / "dst"
    transcode(str(src), str(dst), keep_mtp=True, mtp_quant="mxfp8")

    wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((dst / "config.json").read_text())

    # experts stacked into switch_mlp [E, out, in]
    gp = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    assert f"{gp}.weight" in wm and f"{gp}.scales" in wm
    # global-scale side-car present + has the expert vector + standalone scalars
    g = mx.load(str(dst / "omlx_meta" / "global_scales.safetensors"))
    assert g[gp].shape == (E,)                      # per-expert vector
    assert g["language_model.lm_head"].ndim == 0    # 0-d scalar
    # config: base nvfp4 + per-attn mxfp8 overrides, modelopt block dropped
    assert cfg["quantization"]["mode"] == "nvfp4"
    assert "quantization_config" not in cfg
    qa = cfg["quantization"]["language_model.model.layers.0.self_attn.q_proj"]
    assert qa["mode"] == "mxfp8"
    # vision model_type "_vision" suffix stripped
    assert cfg["vision_config"]["model_type"] == "qwen3_5_moe"
    # format=mlx metadata on the shards (so mlx-vlm skips sanitize)
    shard = sorted(dst.glob("model-*.safetensors"))[0]
    with open(shard, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    assert hdr["__metadata__"]["format"] == "mlx"


def test_passthrough_sanitize_transforms(tmp_path):
    from omlx.transcode_nvfp4 import transcode
    src = _make_modelopt_checkpoint(tmp_path / "src", keep_mtp=True)
    dst = tmp_path / "dst"
    transcode(str(src), str(dst), keep_mtp=True, mtp_quant="mxfp8")
    wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]

    def load(name):
        return mx.load(str(dst / wm[name]))[name]

    # vision conv3d transposed [O,in,T,Hk,Wk] -> [O,T,Hk,Wk,in]
    pe = load("vision_tower.patch_embed.proj.weight")
    assert tuple(pe.shape) == (8, 2, 4, 4, 3)
    # position_ids buffer dropped
    assert not any("position_ids" in k for k in wm)
    # conv1d moveaxis(2,1): [H,1,4] -> [H,4,1]
    cv = load("language_model.model.layers.0.linear_attn.conv1d.weight")
    assert tuple(cv.shape) == (H, 4, 1)
    # RMSNorm +1.0: written == source + 1
    src_norm = mx.load(str(src / "model-00002-of-00002.safetensors"))["model.language_model.norm.weight"]
    dst_norm = load("language_model.model.norm.weight")
    assert float(mx.max(mx.abs((dst_norm.astype(mx.float32) - src_norm.astype(mx.float32)) - 1.0))) < 1e-2
    # MTP norm also +1.0
    src_mnorm = mx.load(str(src / "model-00002-of-00002.safetensors"))["mtp.norm.weight"]
    dst_mnorm = load("language_model.mtp.norm.weight")
    assert float(mx.max(mx.abs((dst_mnorm.astype(mx.float32) - src_mnorm.astype(mx.float32)) - 1.0))) < 1e-2


def test_mtp_head_quantized(tmp_path):
    from omlx.transcode_nvfp4 import transcode
    src = _make_modelopt_checkpoint(tmp_path / "src", keep_mtp=True)

    # mxfp8: the fused gate_up splits to switch_mlp.{gate,up}, all quantized
    dst = tmp_path / "dst_mxfp8"
    transcode(str(src), str(dst), keep_mtp=True, mtp_quant="mxfp8")
    wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((dst / "config.json").read_text())
    base = "language_model.mtp.layers.0.mlp.switch_mlp"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert f"{base}.{proj}.weight" in wm
        assert f"{base}.{proj}.scales" in wm                  # quantized
        assert cfg["quantization"][f"{base}.{proj}"]["mode"] == "mxfp8"
    # MTP head carries NO global scales (mx.quantize is self-contained)
    g = mx.load(str(dst / "omlx_meta" / "global_scales.safetensors"))
    assert not any(k.startswith("language_model.mtp.") for k in g)
    # router gate stays bf16 (no scales)
    assert "language_model.mtp.layers.0.mlp.gate.scales" not in wm

    # bf16: MTP head left un-quantized (lossless), no scales
    dst2 = tmp_path / "dst_bf16"
    transcode(str(src), str(dst2), keep_mtp=True, mtp_quant="bf16")
    wm2 = json.loads((dst2 / "model.safetensors.index.json").read_text())["weight_map"]
    assert f"{base}.gate_proj.weight" in wm2
    assert f"{base}.gate_proj.scales" not in wm2


# ---------------------------------------------------------------------------
# Global-scale output-wrapper tests
# ---------------------------------------------------------------------------

def test_global_scale_wrapper_quantized_linear():
    from omlx.patches.global_scale_runtime import apply
    apply()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(H, 16, bias=False)

    m = M()
    nn.quantize(m, group_size=32, bits=4)        # m.fc -> QuantizedLinear
    x = mx.random.normal((4, H))
    y0 = m.fc(x)                                  # no global attr -> wrapper no-op
    g = mx.array(2.5)
    m.fc._omlx_global_scale = g
    y1 = m.fc(x)                                  # wrapper multiplies
    assert mx.allclose(y1, y0 * g, atol=1e-5)


def test_global_scale_wrapper_quantized_switch():
    from mlx_lm.models.switch_layers import SwitchLinear
    from omlx.patches.global_scale_runtime import apply
    apply()

    sl = SwitchLinear(H, 16, num_experts=4, bias=False).to_quantized(group_size=32, bits=4)
    x = mx.random.normal((3, 1, H))
    indices = mx.array([[0], [2], [3]])           # [tokens, top_k=1]
    y0 = sl(x, indices)
    g = mx.array([1.5, 2.0, 2.5, 3.0])
    sl._omlx_global_scale = g
    y1 = sl(x, indices)
    # each token's output scaled by its routed expert's global
    for t, e in enumerate([0, 2, 3]):
        assert mx.allclose(y1[t], y0[t] * g[e], atol=1e-5)


def test_maybe_attach_global_scales_noop_and_attach(tmp_path):
    from omlx.patches.global_scale_runtime import apply, attach_global_scales
    from omlx.utils.model_loading import maybe_attach_global_scales
    apply()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(H, 16, bias=False)

    # no side-car -> no-op, returns 0, model untouched
    m = M()
    nn.quantize(m, group_size=32, bits=4)
    assert maybe_attach_global_scales(m, str(tmp_path)) == 0
    assert getattr(m.fc, "_omlx_global_scale", None) is None

    # with a side-car keyed on the module path -> attaches
    meta = tmp_path / "omlx_meta"
    meta.mkdir()
    mx.save_safetensors(str(meta / "global_scales.safetensors"), {"fc": mx.array(2.0)})
    n = attach_global_scales(m, str(tmp_path))
    assert n == 1
    assert float(m.fc._omlx_global_scale) == 2.0


# ---------------------------------------------------------------------------
# Full real-model load+infer (the part a synthetic checkpoint can't cover).
# Gated: needs a real transcoded NVFP4 checkpoint + GPU, so it's marked slow
# AND skipped unless OMLX_NVFP4_TEST_MODEL points at one. Run with:
#   OMLX_NVFP4_TEST_MODEL=~/tmp/Qwen3.6-35B-A3B-NVFP4-mtp pytest -m slow \
#     tests/test_nvfp4_transcode.py::test_transcoded_model_loads_and_generates
# Mirrors the manual live validation: pre-load patches -> vlm_load -> shared
# attach hook (asserts scales attach) -> coherent generation.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_transcoded_model_loads_and_generates():
    model_dir = os.environ.get("OMLX_NVFP4_TEST_MODEL", "")
    if not model_dir or not os.path.isdir(os.path.expanduser(model_dir)):
        pytest.skip("set OMLX_NVFP4_TEST_MODEL to a transcoded NVFP4 checkpoint dir")
    model_dir = os.path.expanduser(model_dir)

    # Real-model load+infer needs the GPU (the module pins CPU for the fast unit
    # tests; mlx-vlm's generate reads Metal-only device_info). This test runs
    # last, so flipping the default device here doesn't affect the others.
    mx.set_default_device(mx.gpu)

    from omlx.utils.model_loading import (
        maybe_apply_pre_load_patches,
        maybe_attach_global_scales,
    )

    class _MS:
        mtp_enabled = False

    # pre-load hook applies the global-scale class patch (gated on the side-car)
    maybe_apply_pre_load_patches(model_dir, model_settings=_MS(), for_vlm=True)

    from mlx_vlm import generate
    from mlx_vlm import load as vlm_load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = vlm_load(model_dir)
    # shared post-load hook attaches the global output-scales
    n = maybe_attach_global_scales(model, model_dir)
    assert n > 0, "transcoded NVFP4 model should attach global output-scales"

    try:
        prompt = apply_chat_template(
            processor, model.config, "What is the capital of France?", num_images=0
        )
    except Exception:
        prompt = "What is the capital of France?"
    res = generate(model, processor, prompt, max_tokens=16, verbose=False)
    text = getattr(res, "text", res)
    assert isinstance(text, str) and text.strip(), "expected non-empty generation"
