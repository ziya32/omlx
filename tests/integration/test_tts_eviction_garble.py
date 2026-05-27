# SPDX-License-Identifier: Apache-2.0
"""Eviction-safety test: TTS audio must stay intelligible when a model is
evicted during concurrent TTS generation.

Reproduces the nanobot voice/stress scenario (test_omlx_stress_e2e
``TestPriorityEviction``): under a low ``--max-model-memory`` a pre-warmed
embedding model is evicted the moment the TTS model loads, freeing the victim's
MLX arrays while TTS generation runs on the single MLX executor.

HISTORY (corrected): this test was originally written believing the eviction
free — when it ran on the asyncio event-loop thread (the old ``self._model =
None; gc.collect()`` in each engine ``stop()``) — was what corrupted the audio
(~18% word overlap vs ~84% clean). It was NOT. That garble was an mx.compile +
thread-local ``mx.random.state`` bug in mlx-audio's Qwen3-TTS sampler, fixed
separately (fork fix/qwen); this test now passes because of that mlx-audio fix.
It is retained as defense-in-depth that eviction frees — now serialized onto the
executor under the buffer-access lock (``mx_buffer_lock.locked_free_and_clear``,
issue #85) — don't corrupt concurrent TTS.

Boots a REAL ``omlx serve`` subprocess (so the memory enforcer / engine-pool
eviction actually run — the in-process ASGITransport fixtures do not start the
lifespan), then transcribes the omlx-produced WAVs with mlx-audio ASR directly
(most sensitive measure of audio quality).

Requires, in ~/.myemee/models (or $OMLX_MODEL_DIR): a Qwen3-TTS *CustomVoice*
model, a Qwen3-Embedding model, a Qwen3-ASR model; plus mlx-audio (omlx[audio]).

    pytest -m "slow and integration" tests/integration/test_tts_eviction_garble.py -v
"""

from __future__ import annotations

import asyncio
import gc
import os
import json
import signal
import socket
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="TTS eviction E2E requires macOS with Apple Silicon",
    ),
    pytest.mark.asyncio(loop_scope="session"),
]

_env_dir = os.environ.get("OMLX_MODEL_DIR")
MODEL_DIR = Path(_env_dir) if _env_dir else Path.home() / ".myemee" / "models"

# Cap forces the embedding model (resident from warmup) to be evicted when the
# TTS model loads: embedding (~7GB) + TTS (~4GB) > 8GB.
MAX_MODEL_MEMORY = os.environ.get("OMLX_TEST_MAX_MODEL_MEMORY", "8GB")
PASS_THRESHOLD = 0.60  # clean baseline ~0.84; the bug collapses to ~0.18
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


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _wait_ready(client: httpx.AsyncClient, base: str, proc, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"omlx serve exited early (rc={proc.returncode})")
        try:
            if (await client.get(f"{base}/v1/models", timeout=5)).status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise TimeoutError("omlx serve did not become ready")


async def test_tts_audio_not_garbled_under_model_eviction():
    tts_model = _find_model(["tts", "customvoice"], max_gb=6.0)
    emb_model = _find_model(["embedding"], max_gb=8.0)
    asr_model = _find_model(["asr"], max_gb=4.0)
    missing = [n for n, m in [("Qwen3-TTS CustomVoice", tts_model),
                              ("Qwen3-Embedding", emb_model),
                              ("Qwen3-ASR", asr_model)] if not m]
    if missing:
        pytest.skip(f"Missing models in {MODEL_DIR}: {', '.join(missing)}")
    try:
        from mlx_audio.stt.utils import load_model as load_stt  # noqa: F401
    except ImportError:
        pytest.skip("mlx-audio not installed (omlx[audio]) — needed to score TTS audio")

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    wavs: list[bytes] = []

    with tempfile.TemporaryDirectory() as base_path:
        # v0.3.12 dropped the --max-model-memory serve flag (the static
        # max_*_memory settings were replaced by memory_guard_tier). Force a
        # low admission ceiling — so the idle embedding model is evicted when
        # the TTS model loads — via the tier model's "custom" ceiling, written
        # into the server's settings.json (loaded from --base-path).
        from omlx.config import parse_size
        _ceiling_gb = parse_size(MAX_MODEL_MEMORY) / (1024 ** 3)
        Path(base_path, "settings.json").write_text(json.dumps({
            "memory": {
                "memory_guard_tier": "custom",
                "memory_guard_custom_ceiling_gb": _ceiling_gb,
            }
        }))
        proc = subprocess.Popen(
            [sys.executable, "-m", "omlx", "serve",
             "--model-dir", str(MODEL_DIR),
             "--port", str(port), "--base-path", base_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        try:
            async with httpx.AsyncClient() as client:
                await _wait_ready(client, base, proc)

                # Warm the embedding model so it is resident + idle and becomes
                # the eviction victim when the TTS model loads.
                r = await client.post(
                    f"{base}/v1/embeddings",
                    json={"model": emb_model, "input": "warm up the embedding engine"},
                    timeout=120,
                )
                r.raise_for_status()

                # Fire several concurrent multi-segment TTS requests; the
                # embedding eviction overlaps their generation.
                async def _tts() -> bytes:
                    resp = await client.post(
                        f"{base}/v1/audio/speech",
                        json={"model": tts_model, "input": TEXT,
                              "voice": VOICE, "response_format": "wav"},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    return resp.content

                wavs = list(await asyncio.gather(*[_tts() for _ in range(3)]))
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    assert all(w[:4] == b"RIFF" for w in wavs), "TTS did not return valid WAVs"

    # Transcribe with mlx-audio ASR directly (server is down; clean MLX state).
    # Pass each WAV by path so ASR.generate loads + resamples to 16 kHz itself
    # (avoids depending on a specific mlx-audio resample helper).
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model as load_stt

    asr = load_stt(str(MODEL_DIR / asr_model))
    overlaps = []
    try:
        with tempfile.TemporaryDirectory() as td:
            for i, w in enumerate(wavs):
                wav_path = os.path.join(td, f"clip{i}.wav")
                with open(wav_path, "wb") as fh:
                    fh.write(w)
                text = asr.generate(wav_path, language="English",
                                    temperature=0.0, verbose=False).text.strip()
                overlaps.append(_overlap(TEXT, text))
    finally:
        del asr
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    worst = min(overlaps) if overlaps else 0.0
    assert worst >= PASS_THRESHOLD, (
        f"TTS audio garbled under model eviction: word overlaps {[f'{o:.0%}' for o in overlaps]} "
        f"(worst {worst:.0%} < {PASS_THRESHOLD:.0%}; clean baseline ~84%). "
        "Most likely the mlx-audio Qwen3-TTS sampler regressed (mx.compile + thread-local "
        "mx.random.state off the executor thread; see fork fix/qwen). Less likely: an eviction "
        "buffer free raced generation on the executor (mx_buffer_lock.locked_free_and_clear / #85)."
    )
