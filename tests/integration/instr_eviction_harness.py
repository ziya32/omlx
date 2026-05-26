#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Instrumented loop to CATCH the omlx TTS eviction garble live and trace it.

Boots a REAL `omlx serve` per iteration with OMLX_GTRACE=1 (stderr -> per-iter log),
warms an embedding model (eviction victim), fires 3 concurrent TTS while the embedding
is evicted, kills the server, then ASR-scores each clip. Stops as soon as a clip's word
overlap drops below CATCH (0.60) and points at that iteration's GTRACE log so we can see
exactly which free/reclaim/load ran on which thread relative to the TTS generation
window. Uses the INSTALLED mlx-audio (what omlx serve uses) — no PYTHONPATH override.

    .venv/bin/python tests/integration/instr_eviction_harness.py
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

MODEL_DIR = Path(os.environ.get("OMLX_MODEL_DIR", str(Path.home() / ".myemee" / "models")))
MAX_MEM = os.environ.get("OMLX_TEST_MAX_MODEL_MEMORY", "8GB")
N_ITERS = int(os.environ.get("ITERS", "20"))
CATCH = 0.60
VOICE = "ryan"
TEXT = (
    "Apple Silicon represents a paradigm shift in computer architecture. By "
    "integrating the CPU, GPU, Neural Engine, and unified memory onto a single "
    "chip, Apple has eliminated the traditional bottleneck of copying data "
    "between discrete components. The Metal framework provides direct access to "
    "the GPU's compute capabilities, enabling efficient parallel processing for "
    "machine learning inference and graphics rendering."
)


def _find_model(patterns, max_gb=8.0):
    if not MODEL_DIR.exists():
        return None
    cands = []
    for d in MODEL_DIR.iterdir():
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        n = d.name.lower()
        if all(p.lower() in n for p in patterns):
            g = sum(f.stat().st_size for f in d.glob("*.safetensors")) / (1024**3)
            if g <= max_gb:
                cands.append((g, d.name))
    cands.sort()
    return cands[0][1] if cands else None


def _words(s):
    return {w for w in s.lower().translate(str.maketrans("", "", string.punctuation)).split() if w}


def _overlap(ref, hyp):
    r = _words(ref)
    return len(r & _words(hyp)) / len(r) if r else 0.0


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


async def _wait_ready(client, base, proc, timeout=180):
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
    raise TimeoutError("omlx serve not ready")


async def _one_iter(i, emb, tts_model, base_path):
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    logf = f"/tmp/gtrace_iter{i}.log"
    env = dict(os.environ)
    env["OMLX_GTRACE"] = "1"
    wavs = []
    t_fire = None
    with open(logf, "wb") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "omlx", "serve", "--model-dir", str(MODEL_DIR),
             "--max-model-memory", MAX_MEM, "--port", str(port), "--base-path", base_path],
            stdout=lf, stderr=subprocess.STDOUT, env=env,
        )
        try:
            async with httpx.AsyncClient() as client:
                await _wait_ready(client, base, proc)
                if not os.environ.get("NO_EMBED"):
                    r = await client.post(f"{base}/v1/embeddings",
                                          json={"model": emb, "input": "warm up the embedding engine"},
                                          timeout=120)
                    r.raise_for_status()
                t_fire = time.time()

                async def _tts():
                    resp = await client.post(f"{base}/v1/audio/speech",
                                             json={"model": tts_model, "input": TEXT,
                                                   "voice": VOICE, "response_format": "wav"},
                                             timeout=300)
                    resp.raise_for_status()
                    return resp.content
                wavs = list(await asyncio.gather(*[_tts() for _ in range(3)]))
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    return wavs, t_fire, logf


async def main():
    tts_model = _find_model(["tts", "customvoice"], 6.0)
    emb = _find_model(["embedding"], 8.0)
    asr_model = _find_model(["asr"], 4.0)
    print(f"[HARNESS] tts={tts_model} emb={emb} asr={asr_model} iters={N_ITERS} catch<{CATCH:.0%}", flush=True)
    if not (tts_model and emb and asr_model):
        print("[HARNESS] missing models", flush=True); return

    import gc
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model as load_stt
    asr = load_stt(str(MODEL_DIR / asr_model))  # persistent across iters; server is dead during scoring

    worst_seen = (1.0, -1, [])
    with tempfile.TemporaryDirectory() as base_path:
        for i in range(N_ITERS):
            try:
                wavs, t_fire, logf = await _one_iter(i, emb, tts_model, base_path)
            except Exception as e:
                print(f"[ITER {i}] ERROR booting/generating: {e!r}", flush=True)
                continue
            ok = all(w[:4] == b"RIFF" for w in wavs) if wavs else False
            overlaps = []
            with tempfile.TemporaryDirectory() as td:
                for j, w in enumerate(wavs):
                    p = os.path.join(td, f"c{j}.wav")
                    with open(p, "wb") as fh:
                        fh.write(w)
                    txt = asr.generate(p, language="English", temperature=0.0, verbose=False).text.strip()
                    overlaps.append(_overlap(TEXT, txt))
            worst = min(overlaps) if overlaps else 0.0
            print(f"[ITER {i}] valid_wav={ok} overlaps={[f'{o:.0%}' for o in overlaps]} "
                  f"worst={worst:.0%} log={logf}", flush=True)
            if worst < worst_seen[0]:
                worst_seen = (worst, i, overlaps)
            if worst < CATCH:
                print(f"[CAUGHT] garble at iter {i}: worst={worst:.0%}. "
                      f"GTRACE log -> {logf} (t_fire={t_fire:.4f})", flush=True)
                break
        else:
            w, wi, wo = worst_seen
            print(f"[NOCATCH] no clip < {CATCH:.0%} in {N_ITERS} iters. "
                  f"worst was iter {wi} worst={w:.0%} log=/tmp/gtrace_iter{wi}.log", flush=True)
    try:
        del asr; gc.collect(); mx.clear_cache()
    except Exception:
        pass
    print("[DONE]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
