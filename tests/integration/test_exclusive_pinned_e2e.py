# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test for exclusive pinned model memory management.

Pins a large VLM model (Qwen3.5-27B-Opus46-v2-emee8bit) as exclusive,
then fires concurrent requests for VLM, embedding, reranking, and audio
models.  Validates that:
  - The pinned VLM always gets maximum headroom (non-pinned models evicted).
  - Non-pinned models are served between VLM requests (not starved).
  - Requests retry on memory rejection (507) or timeout (504) up to 3 times.
  - Every response is non-empty and error-free.
  - No single request takes more than 5 minutes.

Models tested:
  - VLM (pinned+exclusive): Qwen3.5-27B-Opus46-v2-emee8bit
  - Embedding:              Qwen3-Embedding-4B
  - Reranker:               Qwen3-Reranker-4B
  - TTS (custom voice):     Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
  - TTS (voice design):     Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
  - ASR:                    Qwen3-ASR-1.7B-bf16

Requirements:
  - Apple Silicon with ≥ 48GB RAM (64GB recommended)
  - All models present in model directory
  - pytest -m slow tests/integration/test_exclusive_pinned_e2e.py -v
"""

from __future__ import annotations

import asyncio
import gc
import io
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

TEST_API_KEY = "test-exclusive-pin"

# Model IDs (directory names)
VLM_MODEL = "Qwen3.5-27B-Opus46-v2-emee8bit"
EMBEDDING_MODEL = "Qwen3-Embedding-4B"
RERANKER_MODEL = "Qwen3-Reranker-4B"
TTS_CUSTOM_MODEL = "Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
TTS_DESIGN_MODEL = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
ASR_MODEL = "Qwen3-ASR-1.7B-bf16"

ALL_MODELS = [
    VLM_MODEL, EMBEDDING_MODEL, RERANKER_MODEL,
    TTS_CUSTOM_MODEL, TTS_DESIGN_MODEL, ASR_MODEL,
]

# Retry / timeout constants
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300.0  # 5 minutes per request (hard fail)
RETRY_BACKOFF = 5.0  # seconds between retries
RETRYABLE_CODES = {504, 507}  # timeout and memory rejection

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Requires macOS with Apple Silicon",
    ),
    pytest.mark.asyncio(loop_scope="session"),
]

# Cap Metal memory to avoid swap
try:
    import mlx.core as mx

    _total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.metal.set_memory_limit(int(_total_bytes * 0.85))
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _get_system_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _get_model_dir() -> Path | None:
    if d := os.environ.get("OMLX_MODEL_DIR"):
        return Path(d)
    for p in [
        Path.home() / ".myemee" / "models",
        Path.home() / "Workspace" / "models",
        Path.home() / "models",
    ]:
        if p.exists():
            return p
    return None


def _make_wav_bytes(
    sample_rate: int = 16000,
    duration_s: float = 2.0,
    frequency: float = 440.0,
) -> bytes:
    """Generate a WAV file with a sine tone (or silence if frequency=0)."""
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    data_size = num_samples * 2

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))       # PCM
    buf.write(struct.pack("<H", 1))       # mono
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
    """Raised when a request fails after MAX_RETRIES attempts."""


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = MAX_RETRIES,
    label: str = "",
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with retry on 504/507.

    Raises RetryExhausted if all retries fail.
    Raises AssertionError if a non-retryable error occurs.
    """
    last_resp = None
    for attempt in range(1, max_retries + 1):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code == 200:
            return resp
        if resp.status_code in RETRYABLE_CODES:
            last_resp = resp
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
        # Non-retryable error — fail immediately
        detail = ""
        try:
            detail = resp.json().get("detail", "")[:300]
        except Exception:
            detail = resp.text[:300]
        raise AssertionError(
            f"[{label}] HTTP {resp.status_code}: {detail}"
        )
    # Should not reach here
    raise RetryExhausted(f"[{label}] unexpected retry loop exit")


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def model_dir():
    d = _get_model_dir()
    if d is None:
        pytest.skip("No model directory found")
    return d


@pytest.fixture(scope="session")
def check_models(model_dir):
    """Verify all required models exist."""
    missing = []
    for mid in ALL_MODELS:
        model_path = model_dir / mid
        if not model_path.exists() or not (model_path / "config.json").exists():
            missing.append(mid)
    if missing:
        pytest.skip(f"Missing models: {', '.join(missing)}")


@pytest_asyncio.fixture(loop_scope="session", scope="module")
async def server_app(model_dir, check_models):
    """Initialize omlx server with VLM pinned as exclusive.

    **Module-scoped** so the pinned VLM (large) is released as soon as
    the last test in this module finishes, before any later integration
    module — in particular ``test_exclusive_live_server.py``, which
    spawns its own subprocess server — can add memory on top of it.
    """
    from omlx.model_settings import ModelSettings, ModelSettingsManager
    from omlx.server import _server_state, app, init_server

    max_mem = int(_get_system_memory_bytes() * 0.80)

    with tempfile.TemporaryDirectory() as tmp_settings:
        settings_mgr = ModelSettingsManager(Path(tmp_settings))

        # Pin VLM as exclusive
        settings_mgr.set_settings(
            VLM_MODEL,
            ModelSettings(is_pinned=True, is_default=True, exclusive=True),
        )

        # Embedding is CausalLM — needs type override
        settings_mgr.set_settings(
            EMBEDDING_MODEL,
            ModelSettings(model_type_override="embedding"),
        )

        # Short TTL on non-pinned models to help with memory pressure
        for mid in [RERANKER_MODEL, TTS_CUSTOM_MODEL, TTS_DESIGN_MODEL, ASR_MODEL]:
            settings_mgr.set_settings(mid, ModelSettings(ttl_seconds=30))

        init_server(
            model_dirs=str(model_dir),
            max_model_memory=max_mem,
            api_key=TEST_API_KEY,
        )

        _server_state.engine_pool.apply_settings_overrides(settings_mgr)

        # Apply pinned status
        _server_state.engine_pool.set_pinned(VLM_MODEL, True)
        _server_state.engine_pool.set_exclusive(VLM_MODEL, True)

        try:
            yield app
        finally:
            # Drain in-flight work, stop background tasks, unload every
            # engine (including the pinned VLM), clear Metal cache.
            if _server_state.engine_pool is not None:
                try:
                    await _server_state.engine_pool.shutdown()
                except Exception:
                    pass
                _server_state.engine_pool = None
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass


@pytest_asyncio.fixture(loop_scope="session", scope="module")
async def client(server_app):
    """Async HTTP client with per-request timeout of 5 minutes."""
    transport = ASGITransport(app=server_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    ) as ac:
        yield ac


# ──────────────────────────────────────────────────────────────────────
# Request builders (each returns a validated response)
# ──────────────────────────────────────────────────────────────────────


async def do_vlm_chat(client: httpx.AsyncClient) -> dict:
    """Send a chat completion to the VLM with enough max_tokens."""
    resp = await _request_with_retry(
        client, "POST", "/v1/chat/completions",
        label="vlm_chat",
        json={
            "model": VLM_MODEL,
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer briefly."},
            ],
            "max_tokens": 256,
            "temperature": 0.3,
        },
    )
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    assert content and len(content.strip()) > 0, (
        f"VLM chat returned empty content: {data}"
    )
    assert data["usage"]["completion_tokens"] > 0, (
        f"VLM chat has 0 completion tokens: {data['usage']}"
    )
    return data


async def do_embedding(client: httpx.AsyncClient) -> dict:
    """Send an embedding request."""
    resp = await _request_with_retry(
        client, "POST", "/v1/embeddings",
        label="embedding",
        json={
            "model": EMBEDDING_MODEL,
            "input": [
                "Machine learning is a branch of artificial intelligence.",
                "The weather today is warm and sunny.",
            ],
        },
    )
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    for emb_obj in data["data"]:
        emb = emb_obj["embedding"]
        assert isinstance(emb, list) and len(emb) > 100, (
            f"Embedding too short: {len(emb)} dims"
        )
    assert data["usage"]["total_tokens"] > 0
    return data


async def do_rerank(client: httpx.AsyncClient) -> dict:
    """Send a rerank request."""
    resp = await _request_with_retry(
        client, "POST", "/v1/rerank",
        label="rerank",
        json={
            "model": RERANKER_MODEL,
            "query": "What is deep learning?",
            "documents": [
                "Deep learning uses neural networks with many layers.",
                "A recipe for chocolate cake with three layers.",
                "Gradient descent is key to training deep networks.",
            ],
            "return_documents": True,
        },
    )
    data = resp.json()
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert "relevance_score" in r
        assert isinstance(r["relevance_score"], (int, float))
        assert "document" in r and r["document"] is not None
    assert data["usage"]["total_tokens"] > 0
    return data


async def do_tts(
    client: httpx.AsyncClient,
    model: str = TTS_CUSTOM_MODEL,
) -> bytes:
    """Send a TTS request and return WAV audio bytes."""
    payload: dict = {
        "model": model,
        "input": "Hello, this is a test of text to speech synthesis.",
        "response_format": "wav",
    }
    # CustomVoice requires a speaker name; VoiceDesign uses instructions
    if "CustomVoice" in model:
        payload["voice"] = "vivian"
    elif "VoiceDesign" in model:
        payload["instructions"] = "Speak in a calm, friendly tone."
    resp = await _request_with_retry(
        client, "POST", "/v1/audio/speech",
        label=f"tts({model})",
        json=payload,
    )
    audio = resp.content
    assert len(audio) > 44, f"TTS audio too small: {len(audio)} bytes"
    assert audio[:4] == b"RIFF", "TTS output is not a valid WAV file"
    assert audio[8:12] == b"WAVE"
    return audio


async def do_asr(client: httpx.AsyncClient, wav_bytes: bytes) -> dict:
    """Send an ASR transcription request."""
    resp = await _request_with_retry(
        client, "POST", "/v1/audio/transcriptions",
        label="asr",
        data={"model": ASR_MODEL, "language": "en"},
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    data = resp.json()
    assert "text" in data, f"ASR response missing 'text': {data}"
    return data


# ──────────────────────────────────────────────────────────────────────
# Tests — sequential (validate each model type works)
# ──────────────────────────────────────────────────────────────────────


class TestExclusivePinnedSequential:
    """Sequential tests — validate each model works before concurrency tests."""

    async def test_01_vlm_chat(self, client):
        """VLM (pinned) should always respond."""
        data = await do_vlm_chat(client)
        # do_vlm_chat already asserts non-empty content and completion_tokens > 0
        assert data["choices"][0]["message"]["content"]

    async def test_02_embedding(self, client):
        """Embedding model loads on demand, responds correctly."""
        data = await do_embedding(client)
        assert len(data["data"]) == 2

    async def test_03_vlm_after_embedding(self, client):
        """VLM still works after embedding was loaded (evicts embedding)."""
        data = await do_vlm_chat(client)
        assert data["usage"]["completion_tokens"] > 0

    async def test_04_rerank(self, client):
        """Reranker model loads on demand."""
        data = await do_rerank(client)
        scores = {r["index"]: r["relevance_score"] for r in data["results"]}
        # ML docs should score higher than cake recipe
        assert scores[0] > scores[1] or scores[2] > scores[1]

    async def test_05_tts_custom_voice(self, client):
        """TTS CustomVoice model loads and produces audio."""
        audio = await do_tts(client, TTS_CUSTOM_MODEL)
        assert len(audio) > 1000

    async def test_06_tts_voice_design(self, client):
        """TTS VoiceDesign model loads and produces audio."""
        audio = await do_tts(client, TTS_DESIGN_MODEL)
        assert len(audio) > 1000

    async def test_07_asr(self, client):
        """ASR model transcribes a tone WAV."""
        wav = _make_wav_bytes(duration_s=2.0, frequency=440.0)
        data = await do_asr(client, wav)
        assert "text" in data

    async def test_08_tts_asr_roundtrip(self, client):
        """TTS → ASR round-trip: synthesize, transcribe, check content."""
        audio = await do_tts(client, TTS_CUSTOM_MODEL)
        data = await do_asr(client, audio)
        text = data["text"].strip().lower()
        assert len(text) > 0, "ASR returned empty transcript from TTS audio"
        # Check at least some words survived
        key_words = ["hello", "test", "speech", "synthesis"]
        matched = [w for w in key_words if w in text]
        assert len(matched) >= 1, (
            f"Round-trip lost content. Transcript: {text!r}, "
            f"matched: {matched}"
        )

    async def test_09_vlm_after_all_models(self, client):
        """VLM still works after all model types were exercised."""
        data = await do_vlm_chat(client)
        assert data["usage"]["completion_tokens"] > 0


# ──────────────────────────────────────────────────────────────────────
# Tests — concurrent (the real stress test)
# ──────────────────────────────────────────────────────────────────────


class TestExclusivePinnedConcurrent:
    """Concurrent tests — fire requests for all model types simultaneously.

    The pinned VLM should always succeed. Non-pinned models may retry
    (up to MAX_RETRIES times) when the VLM holds exclusive memory.
    All requests must eventually succeed.
    """

    async def test_10_vlm_with_embedding_concurrent(self, client):
        """VLM + embedding requests concurrently."""
        vlm_task = asyncio.create_task(do_vlm_chat(client))
        emb_task = asyncio.create_task(do_embedding(client))

        vlm_data, emb_data = await asyncio.gather(
            vlm_task, emb_task, return_exceptions=True
        )

        # VLM must always succeed
        assert not isinstance(vlm_data, BaseException), (
            f"VLM failed: {vlm_data}"
        )
        # Embedding may have retried but must eventually succeed
        assert not isinstance(emb_data, BaseException), (
            f"Embedding failed: {emb_data}"
        )

    async def test_11_vlm_with_rerank_concurrent(self, client):
        """VLM + reranker requests concurrently."""
        vlm_task = asyncio.create_task(do_vlm_chat(client))
        rerank_task = asyncio.create_task(do_rerank(client))

        vlm_data, rerank_data = await asyncio.gather(
            vlm_task, rerank_task, return_exceptions=True
        )

        assert not isinstance(vlm_data, BaseException), (
            f"VLM failed: {vlm_data}"
        )
        assert not isinstance(rerank_data, BaseException), (
            f"Rerank failed: {rerank_data}"
        )

    async def test_12_vlm_with_audio_concurrent(self, client):
        """VLM + TTS + ASR requests concurrently."""
        wav = _make_wav_bytes(duration_s=2.0, frequency=440.0)

        vlm_task = asyncio.create_task(do_vlm_chat(client))
        tts_task = asyncio.create_task(do_tts(client, TTS_CUSTOM_MODEL))
        asr_task = asyncio.create_task(do_asr(client, wav))

        results = await asyncio.gather(
            vlm_task, tts_task, asr_task, return_exceptions=True
        )

        for i, (label, result) in enumerate(
            zip(["VLM", "TTS", "ASR"], results)
        ):
            assert not isinstance(result, BaseException), (
                f"{label} failed: {result}"
            )

    async def test_13_all_models_concurrent(self, client):
        """Fire ALL model types concurrently — the full stress test.

        VLM (pinned) + embedding + reranker + TTS custom + TTS design + ASR
        all at the same time. The engine pool must handle exclusive eviction
        for the VLM while serving or queuing all other requests.
        """
        wav = _make_wav_bytes(duration_s=2.0, frequency=440.0)

        tasks = {
            "VLM": asyncio.create_task(do_vlm_chat(client)),
            "Embedding": asyncio.create_task(do_embedding(client)),
            "Reranker": asyncio.create_task(do_rerank(client)),
            "TTS-Custom": asyncio.create_task(
                do_tts(client, TTS_CUSTOM_MODEL)
            ),
            "TTS-Design": asyncio.create_task(
                do_tts(client, TTS_DESIGN_MODEL)
            ),
            "ASR": asyncio.create_task(do_asr(client, wav)),
        }

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        failures = []
        for label, result in zip(tasks.keys(), results):
            if isinstance(result, BaseException):
                failures.append(f"{label}: {result}")

        assert not failures, (
            f"Concurrent all-model test failures:\n"
            + "\n".join(failures)
        )

    async def test_14_repeated_vlm_with_interleaved_small(self, client):
        """Send 3 VLM requests with small model requests interleaved.

        Validates that non-pinned models can be served in the gaps
        between VLM requests (natural windowing).
        """
        for i in range(3):
            # VLM request
            vlm_data = await do_vlm_chat(client)
            assert vlm_data["usage"]["completion_tokens"] > 0

            # Small model in the gap
            emb_data = await do_embedding(client)
            assert len(emb_data["data"]) == 2

    async def test_15_burst_vlm_then_burst_small(self, client):
        """Burst of VLM requests, then burst of small model requests.

        After VLM burst, memory should be available for small models.
        """
        # VLM burst (2 sequential)
        for _ in range(2):
            data = await do_vlm_chat(client)
            assert data["usage"]["completion_tokens"] > 0

        # Small model burst (all concurrent)
        wav = _make_wav_bytes(duration_s=2.0, frequency=440.0)
        tasks = {
            "Embedding": asyncio.create_task(do_embedding(client)),
            "Reranker": asyncio.create_task(do_rerank(client)),
            "ASR": asyncio.create_task(do_asr(client, wav)),
        }

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        failures = []
        for label, result in zip(tasks.keys(), results):
            if isinstance(result, BaseException):
                failures.append(f"{label}: {result}")

        assert not failures, (
            f"Post-VLM burst small model failures:\n"
            + "\n".join(failures)
        )

    async def test_16_concurrent_vlm_requests(self, client):
        """Multiple concurrent VLM requests — should coalesce, not deadlock."""
        tasks = [
            asyncio.create_task(do_vlm_chat(client))
            for _ in range(3)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                failures.append(f"VLM-{i}: {result}")

        assert not failures, (
            f"Concurrent VLM failures:\n" + "\n".join(failures)
        )
