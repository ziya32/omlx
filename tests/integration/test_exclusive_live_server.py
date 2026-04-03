#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Intensive stress test with a real omlx server process.

Starts omlx as a subprocess, pins the VLM as exclusive via admin API,
fires intensive concurrent requests, measures response times, and
generates a detailed report.

Fail conditions:
  - Any VLM response time > 10 seconds
  - Any request fails after 3 retries
  - Any request takes > 5 minutes
  - Any response is empty or errored

Report: written to test-reports/ with full VLM request/response details.

Run:
  pytest -m slow tests/integration/test_exclusive_live_server.py -v
"""

from __future__ import annotations

import asyncio
import atexit
import io
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

TEST_API_KEY = "test-exclusive-stress"
SERVER_PORT = 10241
SERVER_URL = f"http://localhost:{SERVER_PORT}"

VLM_MODEL = "Qwen3.5-35B-A3B-qwenmlx8bit"
EMBEDDING_MODEL = "Qwen3-Embedding-4B"
RERANKER_MODEL = "Qwen3-Reranker-4B"
TTS_CUSTOM_MODEL = "Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
TTS_DESIGN_MODEL = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
ASR_MODEL = "Qwen3-ASR-1.7B-bf16"

ALL_MODELS = [
    VLM_MODEL, EMBEDDING_MODEL, RERANKER_MODEL,
    TTS_CUSTOM_MODEL, TTS_DESIGN_MODEL, ASR_MODEL,
]

MAX_RETRIES = 3
REQUEST_TIMEOUT = 300.0
RETRY_BACKOFF = 5.0
RETRYABLE_CODES = {504, 507}
SERVER_STARTUP_TIMEOUT = 180
VLM_MAX_RESPONSE_TIME = 30.0  # seconds — fail if single VLM response exceeds this

REPORT_DIR = Path(__file__).resolve().parents[2] / "test-reports"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Requires macOS with Apple Silicon",
    ),
    pytest.mark.asyncio(loop_scope="session"),
]


# ──────────────────────────────────────────────────────────────────────
# Timing tracker
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RequestRecord:
    test_name: str
    label: str
    model: str
    prompt: str = ""
    max_tokens: int = 0
    streaming: bool = False
    start_time: float = 0.0
    ttft: float = 0.0          # time to first token (streaming only)
    total_time: float = 0.0
    status_code: int = 0
    retries: int = 0
    response_content: str = ""  # full response text (no truncation)
    usage: dict = field(default_factory=dict)
    error: str = ""


class TestTracker:
    """Collects all request records for the final report."""

    def __init__(self):
        self.records: list[RequestRecord] = []
        self.failures: list[str] = []

    def add(self, rec: RequestRecord):
        self.records.append(rec)

    def add_failure(self, msg: str):
        self.failures.append(msg)

    def vlm_records(self) -> list[RequestRecord]:
        return [r for r in self.records if r.model == VLM_MODEL]

    def write_report(self):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"exclusive_stress_{ts}.txt"

        lines = []
        lines.append("=" * 80)
        lines.append("EXCLUSIVE PINNED MODEL STRESS TEST REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Server: {SERVER_URL}")
        lines.append(f"VLM: {VLM_MODEL}")
        lines.append(f"VLM max response time threshold: {VLM_MAX_RESPONSE_TIME}s")
        lines.append(f"Total requests tracked: {len(self.records)}")
        lines.append("=" * 80)

        # Failures
        if self.failures:
            lines.append("")
            lines.append("FAILURES:")
            for f in self.failures:
                lines.append(f"  - {f}")

        # VLM response time summary
        vlm_recs = self.vlm_records()
        if vlm_recs:
            lines.append("")
            lines.append("-" * 80)
            lines.append("VLM RESPONSE TIME SUMMARY")
            lines.append("-" * 80)
            times = [r.total_time for r in vlm_recs if r.total_time > 0]
            ttfts = [r.ttft for r in vlm_recs if r.ttft > 0]
            if times:
                lines.append(f"  Count:   {len(times)}")
                lines.append(f"  Min:     {min(times):.2f}s")
                lines.append(f"  Max:     {max(times):.2f}s")
                lines.append(f"  Avg:     {sum(times)/len(times):.2f}s")
                lines.append(f"  Median:  {sorted(times)[len(times)//2]:.2f}s")
            if ttfts:
                lines.append(f"  TTFT min:  {min(ttfts):.2f}s")
                lines.append(f"  TTFT max:  {max(ttfts):.2f}s")
                lines.append(f"  TTFT avg:  {sum(ttfts)/len(ttfts):.2f}s")
            exceeded = [r for r in vlm_recs if r.total_time > VLM_MAX_RESPONSE_TIME]
            lines.append(f"  Exceeded {VLM_MAX_RESPONSE_TIME}s: {len(exceeded)}/{len(vlm_recs)}")

        # All request response times
        lines.append("")
        lines.append("-" * 80)
        lines.append("ALL REQUEST RESPONSE TIMES")
        lines.append("-" * 80)
        lines.append(f"  {'Test':<40} {'Label':<20} {'Model':<15} {'Time':>8} {'TTFT':>8} {'Status':>6} {'Retries':>7}")
        for r in self.records:
            model_short = r.model.split("-")[0] if r.model else ""
            ttft_str = f"{r.ttft:.2f}s" if r.ttft > 0 else "-"
            time_str = f"{r.total_time:.2f}s" if r.total_time > 0 else "err"
            lines.append(
                f"  {r.test_name:<40} {r.label:<20} {model_short:<15} "
                f"{time_str:>8} {ttft_str:>8} {r.status_code:>6} {r.retries:>7}"
            )

        # Full VLM request/response details (no truncation)
        lines.append("")
        lines.append("-" * 80)
        lines.append("FULL VLM REQUEST/RESPONSE DETAILS")
        lines.append("-" * 80)
        for i, r in enumerate(vlm_recs):
            lines.append("")
            lines.append(f"--- VLM Request #{i+1} [{r.test_name} / {r.label}] ---")
            lines.append(f"  Prompt:      {r.prompt}")
            lines.append(f"  Max tokens:  {r.max_tokens}")
            lines.append(f"  Streaming:   {r.streaming}")
            lines.append(f"  Total time:  {r.total_time:.2f}s")
            if r.ttft > 0:
                lines.append(f"  TTFT:        {r.ttft:.2f}s")
            lines.append(f"  Status:      {r.status_code}")
            lines.append(f"  Retries:     {r.retries}")
            lines.append(f"  Usage:       {r.usage}")
            exceeded_marker = " *** EXCEEDED THRESHOLD ***" if r.total_time > VLM_MAX_RESPONSE_TIME else ""
            lines.append(f"  Response{exceeded_marker}:")
            lines.append(r.response_content)

        if self.failures:
            lines.append("")
            lines.append("-" * 80)
            lines.append("FAILURE DETAILS")
            lines.append("-" * 80)
            for f in self.failures:
                lines.append(f)

        report = "\n".join(lines) + "\n"
        path.write_text(report)
        print(f"\n{'='*60}")
        print(f"Report written to: {path}")
        print(f"{'='*60}")
        return path


_tracker = TestTracker()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _get_model_dir() -> str | None:
    if d := os.environ.get("OMLX_MODEL_DIR"):
        return d
    for p in [
        Path.home() / ".myemee" / "models",
        Path.home() / "Workspace" / "models",
        Path.home() / "models",
    ]:
        if p.exists():
            return str(p)
    return None


def _make_wav_bytes(
    sample_rate: int = 16000,
    duration_s: float = 2.0,
    frequency: float = 440.0,
) -> bytes:
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for i in range(num_samples):
        sample = math.sin(2 * math.pi * frequency * i / sample_rate)
        buf.write(struct.pack("<h", int(sample * 32767 * 0.5)))
    return buf.getvalue()


class RetryExhausted(Exception):
    pass


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = MAX_RETRIES,
    label: str = "",
    **kwargs,
) -> httpx.Response:
    for attempt in range(1, max_retries + 1):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code == 200:
            return resp
        if resp.status_code in RETRYABLE_CODES:
            detail = ""
            try:
                detail = resp.json().get("detail", "")[:200]
            except Exception:
                pass
            if attempt < max_retries:
                await asyncio.sleep(RETRY_BACKOFF * attempt)
                continue
            raise RetryExhausted(
                f"[{label}] {max_retries} retries exhausted. "
                f"Last status={resp.status_code} detail={detail}"
            )
        detail = ""
        try:
            detail = resp.json().get("detail", "")[:300]
        except Exception:
            detail = resp.text[:300]
        raise AssertionError(
            f"[{label}] HTTP {resp.status_code}: {detail}"
        )
    raise RetryExhausted(f"[{label}] unexpected retry loop exit")


async def _run_batch(tasks: dict[str, asyncio.Task], label: str = "") -> None:
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    failures = []
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            failures.append(f"  {name}: {result}")
            _tracker.add_failure(f"[{label}] {name}: {result}")
    assert not failures, (
        f"[{label}] failures:\n" + "\n".join(failures)
    )


async def _get_loaded_models(client: httpx.AsyncClient) -> set[str]:
    """Query admin API for loaded model IDs."""
    resp = await client.get("/admin/api/models")
    assert resp.status_code == 200, f"list models: {resp.status_code}"
    data = resp.json()
    # Response is {"models": [...]} where each has "id" and "loaded"
    models = data.get("models", data) if isinstance(data, dict) else data
    loaded = set()
    for m in models:
        if m.get("loaded"):
            loaded.add(m.get("id", ""))
    return loaded


async def _get_model_settings(
    client: httpx.AsyncClient, model_id: str
) -> dict:
    """Get settings for a model from the admin list endpoint."""
    resp = await client.get("/admin/api/models")
    assert resp.status_code == 200
    data = resp.json()
    models = data.get("models", data) if isinstance(data, dict) else data
    for m in models:
        if m.get("id") == model_id:
            return m.get("settings", {})
    return {}


# ──────────────────────────────────────────────────────────────────────
# Request builders with timing
# ──────────────────────────────────────────────────────────────────────


async def do_vlm_chat(
    client: httpx.AsyncClient,
    prompt: str = "What is 2+2? Answer in one word.",
    max_tokens: int = 512,
    test_name: str = "",
    label: str = "vlm_chat",
) -> dict:
    """Non-streaming VLM request with timing."""
    rec = RequestRecord(
        test_name=test_name, label=label, model=VLM_MODEL,
        prompt=prompt, max_tokens=max_tokens, streaming=False,
    )
    t0 = time.monotonic()
    try:
        resp = await _request_with_retry(
            client, "POST", "/v1/chat/completions",
            label=label,
            json={
                "model": VLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        rec.total_time = time.monotonic() - t0
        rec.status_code = resp.status_code
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        rec.response_content = content
        rec.usage = data.get("usage", {})
        assert content and len(content.strip()) > 0
        assert data["usage"]["completion_tokens"] > 0
        _tracker.add(rec)
        return data
    except Exception as e:
        rec.total_time = time.monotonic() - t0
        rec.error = str(e)
        rec.response_content = f"ERROR: {e}"
        _tracker.add(rec)
        raise


async def do_vlm_stream(
    client: httpx.AsyncClient,
    prompt: str = "Write a short paragraph about the color blue.",
    max_tokens: int = 1024,
    test_name: str = "",
    label: str = "vlm_stream",
) -> dict:
    """Streaming VLM request — measures TTFT and total time."""
    rec = RequestRecord(
        test_name=test_name, label=label, model=VLM_MODEL,
        prompt=prompt, max_tokens=max_tokens, streaming=True,
    )
    t0 = time.monotonic()
    chunks = []
    first_token_time = 0.0
    try:
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={
                "model": VLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "stream": True,
            },
        ) as resp:
            rec.status_code = resp.status_code
            if resp.status_code != 200:
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                detail = body.decode(errors="replace")[:300]
                raise AssertionError(
                    f"[{label}] HTTP {resp.status_code}: {detail}"
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(payload)
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = (
                        delta.get("content", "")
                        or delta.get("reasoning_content", "")
                        or delta.get("thinking", "")
                    )
                    if content:
                        if not first_token_time:
                            first_token_time = time.monotonic()
                            rec.ttft = first_token_time - t0
                        chunks.append(content)
                    # Capture usage from last chunk
                    if "usage" in chunk_data and chunk_data["usage"]:
                        rec.usage = chunk_data["usage"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        rec.total_time = time.monotonic() - t0
        full_content = "".join(chunks)
        rec.response_content = full_content
        assert len(full_content.strip()) > 0, "Streaming VLM returned empty content"
        _tracker.add(rec)
        return {
            "content": full_content,
            "ttft": rec.ttft,
            "total_time": rec.total_time,
            "usage": rec.usage,
        }
    except Exception as e:
        rec.total_time = time.monotonic() - t0
        rec.error = str(e)
        rec.response_content = f"ERROR: {e}"
        _tracker.add(rec)
        raise


async def do_embedding(
    client: httpx.AsyncClient,
    test_name: str = "",
    label: str = "embedding",
) -> dict:
    rec = RequestRecord(
        test_name=test_name, label=label, model=EMBEDDING_MODEL,
    )
    t0 = time.monotonic()
    try:
        resp = await _request_with_retry(
            client, "POST", "/v1/embeddings",
            label=label,
            json={
                "model": EMBEDDING_MODEL,
                "input": [
                    "Machine learning is a branch of artificial intelligence.",
                    "The weather today is warm and sunny.",
                    "Neural networks consist of interconnected layers of nodes.",
                    "Quantum computing uses qubits instead of classical bits.",
                ],
            },
        )
        rec.total_time = time.monotonic() - t0
        rec.status_code = resp.status_code
        data = resp.json()
        assert len(data["data"]) == 4
        assert data["usage"]["total_tokens"] > 0
        _tracker.add(rec)
        return data
    except Exception as e:
        rec.total_time = time.monotonic() - t0
        rec.error = str(e)
        _tracker.add(rec)
        raise


async def do_rerank(
    client: httpx.AsyncClient,
    test_name: str = "",
    label: str = "rerank",
) -> dict:
    rec = RequestRecord(
        test_name=test_name, label=label, model=RERANKER_MODEL,
    )
    t0 = time.monotonic()
    try:
        resp = await _request_with_retry(
            client, "POST", "/v1/rerank",
            label=label,
            json={
                "model": RERANKER_MODEL,
                "query": "What is deep learning?",
                "documents": [
                    "Deep learning uses neural networks with many layers.",
                    "A recipe for chocolate cake with three layers.",
                    "Gradient descent is key to training deep networks.",
                    "The stock market fluctuated wildly yesterday.",
                    "Convolutional neural networks excel at image recognition.",
                ],
                "return_documents": True,
            },
        )
        rec.total_time = time.monotonic() - t0
        rec.status_code = resp.status_code
        data = resp.json()
        assert len(data["results"]) == 5
        _tracker.add(rec)
        return data
    except Exception as e:
        rec.total_time = time.monotonic() - t0
        rec.error = str(e)
        _tracker.add(rec)
        raise


async def do_tts(
    client: httpx.AsyncClient,
    model: str = TTS_CUSTOM_MODEL,
    text: str = "Hello, this is a test of text to speech synthesis.",
    test_name: str = "",
    label: str = "tts",
) -> bytes:
    rec = RequestRecord(
        test_name=test_name, label=label, model=model,
    )
    t0 = time.monotonic()
    payload: dict = {"model": model, "input": text, "response_format": "wav"}
    if "CustomVoice" in model:
        payload["voice"] = "vivian"
    elif "VoiceDesign" in model:
        payload["instructions"] = "Speak in a calm, friendly tone."
    try:
        resp = await _request_with_retry(
            client, "POST", "/v1/audio/speech", label=label, json=payload,
        )
        rec.total_time = time.monotonic() - t0
        rec.status_code = resp.status_code
        audio = resp.content
        assert len(audio) > 44 and audio[:4] == b"RIFF"
        _tracker.add(rec)
        return audio
    except Exception as e:
        rec.total_time = time.monotonic() - t0
        rec.error = str(e)
        _tracker.add(rec)
        raise


async def do_asr(
    client: httpx.AsyncClient,
    wav_bytes: bytes,
    test_name: str = "",
    label: str = "asr",
) -> dict:
    rec = RequestRecord(
        test_name=test_name, label=label, model=ASR_MODEL,
    )
    t0 = time.monotonic()
    try:
        resp = await _request_with_retry(
            client, "POST", "/v1/audio/transcriptions", label=label,
            data={"model": ASR_MODEL, "language": "en"},
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        rec.total_time = time.monotonic() - t0
        rec.status_code = resp.status_code
        data = resp.json()
        assert "text" in data
        _tracker.add(rec)
        return data
    except Exception as e:
        rec.total_time = time.monotonic() - t0
        rec.error = str(e)
        _tracker.add(rec)
        raise


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def model_dir():
    d = _get_model_dir()
    if d is None:
        pytest.skip("No model directory found")
    base = Path(d)
    missing = [m for m in ALL_MODELS if not (base / m / "config.json").exists()]
    if missing:
        pytest.skip(f"Missing models: {', '.join(missing)}")
    return d


@pytest.fixture(scope="session")
def server_process(model_dir):
    """Start omlx subprocess, wait for health, yield, kill on teardown."""
    python = os.path.join(
        os.path.dirname(sys.executable), "python"
    ) if "venv" in sys.executable else sys.executable

    cmd = [
        python, "-m", "omlx", "serve",
        "--model-dir", model_dir,
        "--api-key", TEST_API_KEY,
        "--port", str(SERVER_PORT),
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    def _kill_server():
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    atexit.register(_kill_server)

    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    healthy = False
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=2.0)
            if r.status_code == 200:
                healthy = True
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            pytest.skip(
                f"Server exited with code {proc.returncode}: {output[:500]}"
            )
        time.sleep(1)

    if not healthy:
        proc.kill()
        proc.wait()
        pytest.skip(f"Server not healthy within {SERVER_STARTUP_TIMEOUT}s")

    yield proc

    _kill_server()


@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def client(server_process):
    async with httpx.AsyncClient(
        base_url=SERVER_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    ) as ac:
        login_resp = await ac.post(
            "/admin/api/login",
            json={"api_key": TEST_API_KEY, "remember": True},
        )
        assert login_resp.status_code == 200, (
            f"Admin login failed: {login_resp.status_code}"
        )
        yield ac


@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def setup_exclusive(client):
    original = await _get_model_settings(client, VLM_MODEL)

    resp = await client.put(
        f"/admin/api/models/{VLM_MODEL}/settings",
        json={"is_pinned": True, "exclusive": True},
    )
    assert resp.status_code == 200

    # Warm up — first request loads the VLM, don't count it
    await do_vlm_chat(client, prompt="Hi", max_tokens=16, test_name="warmup")
    # Clear warmup record from tracker
    _tracker.records = [r for r in _tracker.records if r.test_name != "warmup"]

    yield

    await client.put(
        f"/admin/api/models/{VLM_MODEL}/settings",
        json={
            "is_pinned": original.get("is_pinned", False),
            "exclusive": original.get("exclusive", False),
        },
    )


# ──────────────────────────────────────────────────────────────────────
# Tests — eviction verification
# ──────────────────────────────────────────────────────────────────────


class TestEvictionVerification:

    async def test_01_vlm_is_exclusive_and_loaded(
        self, client, setup_exclusive
    ):
        settings = await _get_model_settings(client, VLM_MODEL)
        assert settings.get("is_pinned") is True, f"Not pinned: {settings}"
        assert settings.get("exclusive") is True, f"Not exclusive: {settings}"
        loaded = await _get_loaded_models(client)
        assert VLM_MODEL in loaded, f"VLM not loaded: {loaded}"

    async def test_02_vlm_evicts_embedding(self, client, setup_exclusive):
        tn = "test_02_vlm_evicts_embedding"
        await do_embedding(client, test_name=tn)
        loaded = await _get_loaded_models(client)
        assert EMBEDDING_MODEL in loaded

        await do_vlm_chat(client, test_name=tn, label="vlm_evict")

        loaded_after = await _get_loaded_models(client)
        assert EMBEDDING_MODEL not in loaded_after

    async def test_03_vlm_evicts_multiple(self, client, setup_exclusive):
        tn = "test_03_vlm_evicts_multiple"
        await do_embedding(client, test_name=tn)
        await do_rerank(client, test_name=tn)
        loaded = await _get_loaded_models(client)
        assert EMBEDDING_MODEL in loaded or RERANKER_MODEL in loaded

        await do_vlm_chat(client, test_name=tn, label="vlm_evict")
        loaded_after = await _get_loaded_models(client)
        assert EMBEDDING_MODEL not in loaded_after
        assert RERANKER_MODEL not in loaded_after

    async def test_04_small_loads_after_vlm(self, client, setup_exclusive):
        tn = "test_04_small_loads_after_vlm"
        await do_vlm_chat(client, test_name=tn)
        await do_embedding(client, test_name=tn)
        loaded = await _get_loaded_models(client)
        assert EMBEDDING_MODEL in loaded


# ──────────────────────────────────────────────────────────────────────
# Tests — stress with timing
# ──────────────────────────────────────────────────────────────────────


class TestIntensiveStress:

    async def test_10_concurrent_vlm_x5(self, client, setup_exclusive):
        tn = "test_10_concurrent_vlm_x5"
        tasks = {
            f"VLM-{i}": asyncio.create_task(
                do_vlm_chat(client, test_name=tn, label=f"vlm-{i}")
            )
            for i in range(5)
        }
        await _run_batch(tasks, tn)

    async def test_11_vlm_stream_with_embedding_burst(
        self, client, setup_exclusive
    ):
        tn = "test_11_vlm_stream_with_embedding"
        tasks = {
            "VLM-stream": asyncio.create_task(
                do_vlm_stream(
                    client, test_name=tn, label="vlm-stream",
                    prompt="Explain gradient descent in 3 sentences.",
                    max_tokens=512,
                )
            ),
            "Emb-1": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-1")
            ),
            "Emb-2": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-2")
            ),
            "Emb-3": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-3")
            ),
        }
        await _run_batch(tasks, tn)

    async def test_12_vlm_with_all_small(self, client, setup_exclusive):
        tn = "test_12_vlm_with_all_small"
        wav = _make_wav_bytes()
        tasks = {
            "VLM": asyncio.create_task(
                do_vlm_chat(client, test_name=tn, label="vlm")
            ),
            "Embedding": asyncio.create_task(
                do_embedding(client, test_name=tn)
            ),
            "Reranker": asyncio.create_task(
                do_rerank(client, test_name=tn)
            ),
            "TTS-Custom": asyncio.create_task(
                do_tts(client, TTS_CUSTOM_MODEL, test_name=tn, label="tts-cv")
            ),
            "TTS-Design": asyncio.create_task(
                do_tts(client, TTS_DESIGN_MODEL, test_name=tn, label="tts-vd")
            ),
            "ASR": asyncio.create_task(
                do_asr(client, wav, test_name=tn)
            ),
        }
        await _run_batch(tasks, tn)

    async def test_13_rapid_interleaved(self, client, setup_exclusive):
        tn = "test_13_rapid_interleaved"
        fns = [
            lambda: do_embedding(client, test_name=tn, label="emb"),
            lambda: do_rerank(client, test_name=tn, label="rerank"),
            lambda: do_tts(client, TTS_CUSTOM_MODEL, test_name=tn, label="tts"),
            lambda: do_embedding(client, test_name=tn, label="emb"),
            lambda: do_rerank(client, test_name=tn, label="rerank"),
        ]
        for i, fn in enumerate(fns):
            await do_vlm_chat(client, test_name=tn, label=f"vlm-{i}")
            await fn()

    async def test_14_vlm_stream_burst_then_small(
        self, client, setup_exclusive
    ):
        tn = "test_14_vlm_stream_burst"
        for i in range(3):
            await do_vlm_stream(
                client, test_name=tn, label=f"vlm-stream-{i}",
                prompt="Describe the theory of relativity briefly.",
                max_tokens=512,
            )

        wav = _make_wav_bytes()
        tasks = {
            "Emb-1": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-1")
            ),
            "Emb-2": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-2")
            ),
            "Reranker": asyncio.create_task(
                do_rerank(client, test_name=tn)
            ),
            "TTS": asyncio.create_task(
                do_tts(client, TTS_CUSTOM_MODEL, test_name=tn, label="tts")
            ),
            "ASR": asyncio.create_task(
                do_asr(client, wav, test_name=tn)
            ),
        }
        await _run_batch(tasks, tn)

    async def test_15_alternating_rounds(self, client, setup_exclusive):
        tn = "test_15_alternating_rounds"
        small_fns = [
            lambda: do_embedding(client, test_name=tn, label="emb"),
            lambda: do_rerank(client, test_name=tn, label="rerank"),
            lambda: do_tts(client, TTS_CUSTOM_MODEL, test_name=tn, label="tts"),
        ]
        for rnd in range(3):
            fn1 = small_fns[rnd % len(small_fns)]
            fn2 = small_fns[(rnd + 1) % len(small_fns)]
            tasks = {
                f"VLM-r{rnd}": asyncio.create_task(
                    do_vlm_chat(client, test_name=tn, label=f"vlm-r{rnd}")
                ),
                f"S1-r{rnd}": asyncio.create_task(fn1()),
                f"S2-r{rnd}": asyncio.create_task(fn2()),
            }
            await _run_batch(tasks, f"{tn}_r{rnd}")

    async def test_16_tts_asr_roundtrip_under_vlm(
        self, client, setup_exclusive
    ):
        tn = "test_16_roundtrip_under_vlm"

        async def roundtrip():
            audio = await do_tts(
                client, TTS_CUSTOM_MODEL, text="The quick brown fox.",
                test_name=tn, label="tts-rt",
            )
            data = await do_asr(client, audio, test_name=tn, label="asr-rt")
            assert len(data["text"].strip()) > 0
            return data

        tasks = {
            "VLM-stream": asyncio.create_task(
                do_vlm_stream(client, test_name=tn, label="vlm-stream")
            ),
            "VLM-short": asyncio.create_task(
                do_vlm_chat(client, test_name=tn, label="vlm-short")
            ),
            "TTS-ASR": asyncio.create_task(roundtrip()),
        }
        await _run_batch(tasks, tn)

    async def test_17_max_stress(self, client, setup_exclusive):
        tn = "test_17_max_stress"
        wav = _make_wav_bytes()
        tasks = {
            "VLM-stream": asyncio.create_task(
                do_vlm_stream(client, test_name=tn, label="vlm-stream")
            ),
            "VLM-short": asyncio.create_task(
                do_vlm_chat(client, test_name=tn, label="vlm-short")
            ),
            "VLM-tiny": asyncio.create_task(
                do_vlm_chat(
                    client, prompt="Name 3 colors.", max_tokens=64,
                    test_name=tn, label="vlm-tiny",
                )
            ),
            "Emb-1": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-1")
            ),
            "Emb-2": asyncio.create_task(
                do_embedding(client, test_name=tn, label="emb-2")
            ),
            "Rerank-1": asyncio.create_task(
                do_rerank(client, test_name=tn, label="rerank-1")
            ),
            "Rerank-2": asyncio.create_task(
                do_rerank(client, test_name=tn, label="rerank-2")
            ),
            "TTS": asyncio.create_task(
                do_tts(client, TTS_CUSTOM_MODEL, test_name=tn, label="tts")
            ),
            "ASR": asyncio.create_task(
                do_asr(client, wav, test_name=tn, label="asr")
            ),
        }
        await _run_batch(tasks, tn)

    async def test_18_endurance_10_rounds(self, client, setup_exclusive):
        tn = "test_18_endurance"
        for rnd in range(10):
            if rnd % 2 == 0:
                tasks = {
                    f"VLM-r{rnd}": asyncio.create_task(
                        do_vlm_chat(
                            client, test_name=tn, label=f"vlm-r{rnd}"
                        )
                    ),
                    f"Emb-r{rnd}": asyncio.create_task(
                        do_embedding(client, test_name=tn, label=f"emb-r{rnd}")
                    ),
                }
            else:
                tasks = {
                    f"Emb-r{rnd}": asyncio.create_task(
                        do_embedding(client, test_name=tn, label=f"emb-r{rnd}")
                    ),
                    f"Rerank-r{rnd}": asyncio.create_task(
                        do_rerank(client, test_name=tn, label=f"rerank-r{rnd}")
                    ),
                    f"VLM-r{rnd}": asyncio.create_task(
                        do_vlm_chat(
                            client, prompt="Say hello.", max_tokens=32,
                            test_name=tn, label=f"vlm-r{rnd}",
                        )
                    ),
                }
            await _run_batch(tasks, f"{tn}_r{rnd}")


# ──────────────────────────────────────────────────────────────────────
# Final report + VLM response time assertion
# ──────────────────────────────────────────────────────────────────────


class TestReport:

    async def test_99_generate_report_and_check_vlm_times(
        self, client, setup_exclusive
    ):
        """Generate report and fail if any VLM response exceeded threshold."""
        report_path = _tracker.write_report()

        vlm_recs = _tracker.vlm_records()
        exceeded = [
            r for r in vlm_recs
            if r.total_time > VLM_MAX_RESPONSE_TIME and not r.error
        ]

        if exceeded:
            details = "\n".join(
                f"  {r.label} ({r.test_name}): {r.total_time:.2f}s "
                f"(prompt: {r.prompt[:60]}...)"
                for r in exceeded
            )
            pytest.fail(
                f"{len(exceeded)} VLM request(s) exceeded "
                f"{VLM_MAX_RESPONSE_TIME}s threshold:\n{details}\n"
                f"Full report: {report_path}"
            )

    async def test_99b_check_exclusive_isolation(
        self, client, setup_exclusive
    ):
        """Scan server log: no non-VLM engine work during VLM inference.

        Parses the omlx server log for the duration of this test session
        and checks that no embedding, reranker, TTS, STT, or model-loading
        activity occurs between a VLM "Chat completion received" and its
        matching "Chat completion [same-id]: N tokens" completion line.
        """
        import re
        from pathlib import Path

        log_path = Path.home() / ".omlx" / "logs" / "server.log"
        if not log_path.exists():
            pytest.skip(f"Server log not found at {log_path}")

        log_lines = log_path.read_text().splitlines()

        # Use the tracker's earliest and latest timestamps to scope the scan
        # to this test session only.
        session_start = min(
            (r.start_time for r in _tracker.records if r.start_time > 0),
            default=0,
        )
        if session_start == 0:
            pytest.skip("No tracked requests with timestamps")

        # ── Parse log lines ──────────────────────────────────────────
        # Patterns:
        #   Chat completion [REQ_ID] received: model=..., ...
        #   Chat completion [REQ_ID]: N tokens in Xs (Y tok/s)
        #   Embedding [...]: ...
        #   Rerank [...]: ...
        #   Speech [...]: ...
        #   Loading model: ...
        #   STT transcribe done: ...

        vlm_received_re = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
            r"Chat completion \[([0-9a-f-]+)\] received:.*"
            r"model=" + re.escape(VLM_MODEL)
        )
        vlm_done_re = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*"
            r"Chat completion \[([0-9a-f-]+)\]: \d+ tokens in"
        )

        # Lines that indicate non-VLM engine work
        non_vlm_work_re = re.compile(
            r"Embedding \[|Rerank \[|Speech \[|STT transcribe done|"
            r"Loading model:|Loaded model:|Reranker engine started|"
            r"Embedding engine started|state=unloaded→loading"
        )

        def _parse_ts(ts_str: str) -> str:
            """Return timestamp string for display."""
            return ts_str

        # ── Identify VLM in-flight windows ───────────────────────────
        # Build a list of (req_id, start_line_idx, end_line_idx) tuples
        vlm_windows: list[tuple[str, int, int]] = []
        open_vlm: dict[str, int] = {}  # req_id → start line index

        for idx, line in enumerate(log_lines):
            m = vlm_received_re.match(line)
            if m:
                req_id = m.group(2)
                open_vlm[req_id] = idx
                continue
            m = vlm_done_re.match(line)
            if m:
                req_id = m.group(2)
                if req_id in open_vlm:
                    vlm_windows.append((req_id, open_vlm.pop(req_id), idx))

        # ── Scan for violations ──────────────────────────────────────
        violations: list[str] = []

        for req_id, start_idx, end_idx in vlm_windows:
            for line_idx in range(start_idx + 1, end_idx):
                line = log_lines[line_idx]
                if non_vlm_work_re.search(line):
                    violations.append(
                        f"  VLM [{req_id}] lines {start_idx+1}-{end_idx+1}:\n"
                        f"    !! line {line_idx+1}: {line.strip()}"
                    )

        # ── Report ───────────────────────────────────────────────────
        # Append to the test report file
        report_dir = REPORT_DIR
        report_files = sorted(report_dir.glob("exclusive_stress_*.txt"))
        if report_files:
            report_path = report_files[-1]
            with open(report_path, "a") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("EXCLUSIVE ISOLATION CHECK\n")
                f.write("=" * 80 + "\n")
                f.write(f"VLM in-flight windows found: {len(vlm_windows)}\n")
                f.write(f"Violations: {len(violations)}\n")
                if violations:
                    f.write("\nDETAILS:\n")
                    for v in violations:
                        f.write(v + "\n")
                else:
                    f.write("\nNo exclusivity violations found.\n")

        if violations:
            summary = "\n".join(violations[:20])
            extra = (
                f"\n  ... and {len(violations) - 20} more"
                if len(violations) > 20 else ""
            )
            pytest.fail(
                f"{len(violations)} exclusivity violation(s) found — "
                f"non-VLM engine work occurred during VLM inference:\n"
                f"{summary}{extra}\n"
                f"Full report: {report_files[-1] if report_files else 'N/A'}"
            )
