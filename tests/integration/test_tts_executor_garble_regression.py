# SPDX-License-Identifier: Apache-2.0
"""Regression test: Qwen3-TTS audio must not garble when generated on the MLX executor thread.

Guards against (fixed upstream in mlx-audio branch fix/qwen): the TTS sampler
``categorical_sampling`` was ``@mx.compile(inputs=mx.random.state, ...)``. The compiled
graph captures MLX's THREAD-LOCAL global RNG and binds it to the thread that first traced
it (the main thread, at import). omlx runs ALL TTS generation on a single background MLX
executor thread (``engine_core.get_mlx_executor``, to serialise Metal access — issue #85).
There the compiled sampler can no longer advance that thread's RNG, so sampling degenerates
and Qwen3-TTS drifts and rambles → garbled audio.

What this test does — it tests the ACTUAL AUDIO: generate the same text on the executor
thread, transcribe each clip with Qwen3-ASR, and assert every clip is intelligible (word
overlap >= ``PASS_THRESHOLD``). Buggy clips collapse (overlap ~0.04-0.43 historically);
with the fix they are ~0.86-1.00.

Why SUBPROCESSES: the buggy frozen-RNG snapshot is captured once *per process*, so every
generation within one process is byte-identical — a single process reproduces the garble
only when its frozen key happens to be a bad one (~1/3 of fresh processes). To RELIABLY
catch a regression we therefore generate each clip in a FRESH subprocess (this file is its
own worker: ``python test_..._garble_regression.py <out.wav>``), giving an independent
frozen-RNG draw per clip. Across N clips a regression fails ~1-0.67**N of the time.

Speed: the buggy path rambles to max length (~327 s of audio / ~4 min per clip uncapped),
which is what made naive reproductions take many minutes. We cap ``max_tokens`` above a
healthy clip (~450 codec tokens) but far below the runaway ramble, so each clip stays a
few seconds and the whole test is ~2 min (and a regression stays bounded, not 14 min).

    pytest -m "slow and integration" tests/integration/test_tts_executor_garble_regression.py -v -s

Requires, in ~/.myemee/models (or $OMLX_MODEL_DIR): a Qwen3-TTS *CustomVoice* model and a
Qwen3-ASR model; plus mlx-audio (omlx[audio]).
"""

from __future__ import annotations

import gc
import os
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="MLX executor-thread TTS regression requires macOS with Apple Silicon",
    ),
]

_env_dir = os.environ.get("OMLX_MODEL_DIR")
MODEL_DIR = Path(_env_dir) if _env_dir else Path.home() / ".myemee" / "models"

# Fresh processes, so N is the reliability knob: a regression fails ~1-0.67**N of runs.
N_RUNS = int(os.environ.get("TTS_GARBLE_RUNS", "6"))
# Cap generation length: a healthy clip of TEXT is ~450 codec tokens; the buggy path
# otherwise rambles to thousands (~327 s of audio). This bounds a regressed clip to a few
# seconds without ever truncating a healthy one.
MAX_TOKENS = int(os.environ.get("TTS_GARBLE_MAX_TOKENS", "600"))
PASS_THRESHOLD = 0.60  # healthy ~0.86-1.00; the executor-thread garble collapses to ~0.04-0.43
VOICE = "ryan"
TEXT = (
    "Apple Silicon represents a paradigm shift in computer architecture. By "
    "integrating the CPU, GPU, Neural Engine, and unified memory onto a single "
    "chip, Apple has eliminated the traditional bottleneck of copying data "
    "between discrete components. The Metal framework provides direct access to "
    "the GPU's compute capabilities, enabling efficient parallel processing for "
    "machine learning inference and graphics rendering."
)


def _find_model(patterns: list[str], max_gb: float = 8.0) -> Optional[str]:
    """Smallest model dir whose name contains *all* of *patterns* (≤ max_gb)."""
    if not MODEL_DIR.exists():
        return None
    candidates = []
    for d in MODEL_DIR.iterdir():
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        name = d.name.lower()
        if all(p.lower() in name for p in patterns):
            size_gb = sum(f.stat().st_size for f in d.glob("*.safetensors")) / (1024**3)
            if size_gb <= max_gb:
                candidates.append((size_gb, d.name))
    candidates.sort()
    return candidates[0][1] if candidates else None


def _words(s: str) -> set[str]:
    return {w for w in s.lower().translate(str.maketrans("", "", string.punctuation)).split() if w}


def _overlap(reference: str, hypothesis: str) -> float:
    ref = _words(reference)
    return len(ref & _words(hypothesis)) / len(ref) if ref else 0.0


def _generate_one_clip(out_wav_path: str) -> None:
    """SUBPROCESS WORKER: generate one clip ON the MLX executor thread, write a WAV.

    Run as a fresh process (``python <thisfile> <out.wav>``) so the buggy compiled
    sampler captures a fresh per-process frozen-RNG snapshot. Generating on
    ``get_mlx_executor`` (not the main thread) is what reproduces the bug.
    """
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    from omlx.engine_core import get_mlx_executor
    from mlx_audio.tts.utils import load_model as load_tts

    tts_model = _find_model(["tts", "customvoice"], max_gb=6.0)
    exe = get_mlx_executor()
    tts = exe.submit(lambda: load_tts(str(MODEL_DIR / tts_model))).result()

    def _gen():
        results = list(tts.generate(text=TEXT, voice=VOICE, max_tokens=MAX_TOKENS))
        audio = mx.concatenate([r.audio.reshape(-1) for r in results])
        mx.eval(audio)
        return np.asarray(audio).reshape(-1), int(results[0].sample_rate)

    audio_np, sr = exe.submit(_gen).result()
    sf.write(out_wav_path, audio_np, sr, subtype="PCM_16")


def test_tts_executor_thread_audio_not_garbled():
    tts_model = _find_model(["tts", "customvoice"], max_gb=6.0)
    asr_model = _find_model(["asr"], max_gb=4.0)
    missing = [n for n, m in [("Qwen3-TTS CustomVoice", tts_model),
                              ("Qwen3-ASR", asr_model)] if not m]
    if missing:
        pytest.skip(f"Missing models in {MODEL_DIR}: {', '.join(missing)}")

    try:
        import mlx.core as mx  # noqa: F401
        import soundfile as sf  # noqa: F401
        from mlx_audio.stt.utils import load_model as load_stt  # noqa: F401
    except ImportError:
        pytest.skip("mlx-audio not installed (omlx[audio]) — needed to exercise TTS")

    import mlx.core as mx
    import soundfile as sf

    from mlx_audio.stt.utils import load_model as load_stt

    asr = load_stt(str(MODEL_DIR / asr_model))  # parent scores on the main thread (greedy, unaffected)

    overlaps: list[float] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            for i in range(N_RUNS):
                out = os.path.join(td, f"clip{i}.wav")
                # FRESH process => fresh per-process frozen-RNG draw (see module docstring).
                proc = subprocess.run(
                    [sys.executable, __file__, out],
                    capture_output=True, text=True, timeout=300,
                )
                assert proc.returncode == 0 and os.path.exists(out), (
                    f"TTS generation subprocess {i + 1}/{N_RUNS} failed "
                    f"(rc={proc.returncode}):\n{proc.stderr[-1500:]}"
                )
                text = asr.generate(out, language="English",
                                    temperature=0.0, verbose=False).text.strip()
                ov = _overlap(TEXT, text)
                overlaps.append(ov)
                info = sf.info(out)
                print(f"[TTS-GARBLE] clip {i + 1}/{N_RUNS} (fresh proc) overlap={ov:.0%} "
                      f"dur={info.frames / info.samplerate:.1f}s asr={text[:50]!r}", flush=True)
    finally:
        del asr
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    worst = min(overlaps) if overlaps else 0.0
    print(f"\n[TTS-GARBLE] overlaps={[f'{o:.0%}' for o in overlaps]} worst={worst:.0%} "
          f"({N_RUNS} fresh processes, threshold {PASS_THRESHOLD:.0%})", flush=True)
    assert worst >= PASS_THRESHOLD, (
        f"Qwen3-TTS audio garbled on the MLX executor thread: word overlaps "
        f"{[f'{o:.0%}' for o in overlaps]} (worst {worst:.0%} < {PASS_THRESHOLD:.0%}). "
        "The compiled categorical_sampler likely captured thread-local mx.random.state "
        "and degenerated off its compile thread — the TTS sampler must thread an explicit "
        "RNG key (mlx-audio branch fix/qwen)."
    )


if __name__ == "__main__":
    # Subprocess worker entrypoint: generate exactly one clip to the given path.
    _generate_one_clip(sys.argv[1])
    raise SystemExit(0)
