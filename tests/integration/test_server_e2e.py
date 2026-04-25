# SPDX-License-Identifier: Apache-2.0
"""
End-to-end server tests with real models via HTTP endpoints.

Tests start a real omlx server (via init_server) and exercise the full
request path: HTTP → FastAPI → EnginePool → real Engine → real model.

Models are loaded on demand by the server's EnginePool with LRU eviction.
The test order exercises LRU management as different model types are needed:
  1. Embedding tests  → EnginePool loads embedding model
  2. Reranker tests   → EnginePool loads reranker (may evict embedding)
  3. ASR tests        → EnginePool loads ASR (may evict previous)
  4. TTS tests        → EnginePool loads TTS (may evict previous)
  5. Round-trip test   → EnginePool manages both TTS and ASR via LRU

Uses httpx.AsyncClient with ASGITransport for proper async execution.
All engine background tasks (scheduler loops) share the session event loop,
avoiding deadlocks that occur with sync TestClient bridges.

Requirements:
- Apple Silicon (M1/M2/M3/M4)
- Models in ~/.myemee/models/ or OMLX_MODEL_DIR
- pytest -m slow tests/integration/test_server_e2e.py -v

Memory: max_model_memory is set to 75% of system RAM.
"""

import gc
import io
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

TEST_API_KEY = "test-api-key"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Server E2E tests require macOS with Apple Silicon",
    ),
    pytest.mark.asyncio(loop_scope="session"),
]

# Cap Metal memory
try:
    import mlx.core as mx

    _total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.metal.set_memory_limit(int(_total_bytes * 0.75))
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _get_system_memory_bytes() -> int:
    """Return total physical memory in bytes."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _get_model_dir() -> Optional[Path]:
    """Get model directory from env or default locations."""
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


def _find_model(base: Path, patterns: list[str], min_gb: float = 0.0, max_gb: float = 8.0) -> Optional[str]:
    """Find a model_id (directory name) matching patterns within size bounds."""
    candidates = []
    for subdir in base.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        if not (subdir / "config.json").exists():
            continue
        name = subdir.name.lower()
        if any(p.lower() in name for p in patterns):
            size = sum(f.stat().st_size for f in subdir.glob("*.safetensors"))
            size_gb = size / (1024**3)
            if min_gb < size_gb <= max_gb:
                candidates.append((size_gb, subdir.name))
    if not candidates:
        return None
    candidates.sort()  # smallest first
    return candidates[0][1]


def _make_wav_bytes(sample_rate: int = 16000, duration_s: float = 1.0, frequency: float = 0.0) -> bytes:
    """Generate WAV bytes. frequency=0 for silence, >0 for a tone."""
    import math

    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    data_size = num_samples * 2

    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))       # PCM
    buf.write(struct.pack("<H", 1))       # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))       # block align
    buf.write(struct.pack("<H", 16))      # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))

    for i in range(num_samples):
        if frequency > 0:
            sample = math.sin(2 * math.pi * frequency * i / sample_rate)
        else:
            sample = 0.0
        buf.write(struct.pack("<h", int(sample * 32767 * 0.5)))

    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# Fixtures (session-scoped — one server for all tests)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def model_dir():
    d = _get_model_dir()
    if d is None:
        pytest.skip("No model directory found")
    return d


@pytest.fixture(scope="session")
def model_ids(model_dir):
    """Discover model IDs for each engine type; skip if any are missing."""
    embedding_id = _find_model(model_dir, ["Embedding"], min_gb=3.0, max_gb=8.0)
    reranker_id = _find_model(model_dir, ["Reranker"], min_gb=4.0, max_gb=8.0)
    asr_id = _find_model(model_dir, ["ASR", "asr", "Whisper", "whisper"], max_gb=4.0)
    tts_id = _find_model(model_dir, ["TTS", "tts"], max_gb=6.0)

    missing = []
    if not embedding_id:
        missing.append("embedding (Qwen3-Embedding)")
    if not reranker_id:
        missing.append("reranker (Qwen3-Reranker ≥4B)")
    if not asr_id:
        missing.append("ASR")
    if not tts_id:
        missing.append("TTS")
    if missing:
        pytest.skip(f"Missing models: {', '.join(missing)}")

    return {
        "embedding": embedding_id,
        "reranker": reranker_id,
        "asr": asr_id,
        "tts": tts_id,
    }


@pytest_asyncio.fixture(loop_scope="session", scope="module")
async def server_app(model_dir, model_ids):
    """Initialize the real omlx server and return the FastAPI app.

    Models are NOT pre-loaded — the EnginePool loads them on demand
    via get_engine() and manages memory with LRU eviction.

    **Module-scoped** so that loaded engines, the tempfile settings
    directory, and Metal buffers are all released as soon as the last
    test in this module finishes. Session scope would keep this pool
    alive concurrently with other integration modules' fixtures — and
    with the subprocess server spawned by test_exclusive_live_server.py
    — causing double (or triple) memory usage.
    """
    from omlx.model_settings import ModelSettings, ModelSettingsManager
    from omlx.server import _server_state, app, init_server

    max_mem = int(_get_system_memory_bytes() * 0.75)

    with tempfile.TemporaryDirectory() as tmp_settings:
        settings_mgr = ModelSettingsManager(Path(tmp_settings))

        # Qwen3-Embedding is CausalLM — override to embedding
        settings_mgr.set_settings(
            model_ids["embedding"],
            ModelSettings(model_type_override="embedding"),
        )
        # Qwen3-Reranker is auto-detected via CausalLM arch + directory name

        init_server(
            model_dirs=str(model_dir),
            max_model_memory=max_mem,
            api_key=TEST_API_KEY,
        )

        # Apply our overrides after init_server's own apply_settings_overrides
        _server_state.engine_pool.apply_settings_overrides(settings_mgr)

        try:
            yield app
        finally:
            # Drain in-flight work, stop background tasks, unload every
            # engine, clear Metal cache — so the next integration module
            # starts with a clean Metal state and doesn't accumulate
            # memory on top of this one.
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
    """Async HTTP client sharing the session event loop.

    Uses httpx.AsyncClient with ASGITransport so that all async operations
    (endpoint handlers, engine scheduler loops, event waits) run on the
    same event loop — no sync-to-async bridge that could starve background
    tasks and cause deadlocks.
    """
    transport = ASGITransport(app=server_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=httpx.Timeout(600.0),  # 10min per request for large models
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    ) as ac:
        yield ac


# ──────────────────────────────────────────────────────────────────────
# Embedding endpoint
# ──────────────────────────────────────────────────────────────────────


class TestEmbeddingEndpoint:
    """Test /v1/embeddings with a real embedding model."""

    async def test_embed_single_text(self, client, model_ids):
        resp = await client.post("/v1/embeddings", json={
            "model": model_ids["embedding"],
            "input": "Hello world",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        emb = data["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) > 100  # real embeddings have many dimensions

    async def test_embed_multiple_texts(self, client, model_ids):
        texts = [
            "Machine learning is great",
            "The weather is sunny",
            "Deep learning uses neural networks",
        ]
        resp = await client.post("/v1/embeddings", json={
            "model": model_ids["embedding"],
            "input": texts,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        # All embeddings should have the same dimensions
        dims = [len(d["embedding"]) for d in data["data"]]
        assert dims[0] == dims[1] == dims[2]

    async def test_embed_usage_tokens(self, client, model_ids):
        resp = await client.post("/v1/embeddings", json={
            "model": model_ids["embedding"],
            "input": "Test token counting",
        })
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] > 0

    async def test_embed_cosine_similarity(self, client, model_ids):
        """Similar texts should have higher cosine similarity than dissimilar ones."""
        resp = await client.post("/v1/embeddings", json={
            "model": model_ids["embedding"],
            "input": [
                "Python is a programming language",
                "Python supports object-oriented programming",
                "The recipe calls for two cups of flour",
            ],
        })
        assert resp.status_code == 200
        embs = [d["embedding"] for d in resp.json()["data"]]

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_related = cosine_sim(embs[0], embs[1])
        sim_unrelated = cosine_sim(embs[0], embs[2])
        assert sim_related > sim_unrelated, (
            f"Related texts similarity ({sim_related:.3f}) should exceed "
            f"unrelated ({sim_unrelated:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────
# Rerank endpoint
# ──────────────────────────────────────────────────────────────────────


class TestRerankEndpoint:
    """Test /v1/rerank with a real LLM reranker model."""

    async def test_rerank_basic(self, client, model_ids):
        resp = await client.post("/v1/rerank", json={
            "model": model_ids["reranker"],
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "The weather forecast for tomorrow predicts rain in the afternoon.",
                "Deep learning uses neural networks with many layers to learn complex patterns.",
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        # ML-related docs should rank above weather
        scores = {r["index"]: r["relevance_score"] for r in data["results"]}
        assert scores[0] > scores[1], (
            f"ML doc ({scores[0]:.3f}) should outscore weather doc ({scores[1]:.3f})"
        )
        assert scores[2] > scores[1], (
            f"DL doc ({scores[2]:.3f}) should outscore weather doc ({scores[1]:.3f})"
        )

    async def test_rerank_top_n(self, client, model_ids):
        resp = await client.post("/v1/rerank", json={
            "model": model_ids["reranker"],
            "query": "Python programming",
            "documents": [
                "Python is a high-level programming language.",
                "Java is another popular language.",
                "Cooking recipes for dinner.",
                "Python supports multiple paradigms.",
            ],
            "top_n": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        # Top results should be Python-related (index 0 or 3)
        top_indices = {r["index"] for r in data["results"]}
        assert top_indices & {0, 3}, (
            f"Top-2 should include Python docs, got indices {top_indices}"
        )

    async def test_rerank_return_documents(self, client, model_ids):
        docs = ["First document", "Second document"]
        resp = await client.post("/v1/rerank", json={
            "model": model_ids["reranker"],
            "query": "test",
            "documents": docs,
            "return_documents": True,
        })
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert "document" in r
            assert r["document"]["text"] in docs

    async def test_rerank_usage(self, client, model_ids):
        resp = await client.post("/v1/rerank", json={
            "model": model_ids["reranker"],
            "query": "test",
            "documents": ["doc one", "doc two"],
        })
        assert resp.status_code == 200
        assert resp.json()["usage"]["total_tokens"] > 0


# ──────────────────────────────────────────────────────────────────────
# ASR endpoint
# ──────────────────────────────────────────────────────────────────────


class TestASREndpoint:
    """Test /v1/audio/transcriptions with a real ASR model."""

    async def test_transcribe_silent_audio(self, client, model_ids):
        """Silent audio should return a valid (possibly empty) transcription."""
        wav = _make_wav_bytes(duration_s=1.0)
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": model_ids["asr"], "language": "auto"},
            files={"file": ("silence.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data


# ──────────────────────────────────────────────────────────────────────
# TTS endpoint
# ──────────────────────────────────────────────────────────────────────


class TestTTSEndpoint:
    """Test /v1/audio/speech and /v1/audio/speakers with a real TTS model."""

    async def test_synthesize_speech(self, client, model_ids):
        resp = await client.post("/v1/audio/speech", json={
            "model": model_ids["tts"],
            "input": "Hello, this is a test.",
            "voice": "vivian",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        audio = resp.content
        assert len(audio) > 44  # more than just a WAV header
        assert audio[:4] == b"RIFF"
        assert audio[8:12] == b"WAVE"

    async def test_synthesize_with_voice(self, client, model_ids):
        resp = await client.post("/v1/audio/speech", json={
            "model": model_ids["tts"],
            "input": "Testing voice selection.",
            "voice": "vivian",
        })
        assert resp.status_code == 200
        assert len(resp.content) > 44

    async def test_speakers_list(self, client, model_ids):
        resp = await client.get(
            "/v1/audio/speakers",
            params={"model": model_ids["tts"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["speakers"], list)


# ──────────────────────────────────────────────────────────────────────
# TTS → ASR round-trip via HTTP endpoints
# ──────────────────────────────────────────────────────────────────────


class TestTTSASRRoundTripHTTP:
    """Synthesize speech via /v1/audio/speech, transcribe it via /v1/audio/transcriptions."""

    async def test_round_trip(self, client, model_ids):
        source_text = "The quick brown fox jumps over the lazy dog."

        # Step 1: TTS
        tts_resp = await client.post("/v1/audio/speech", json={
            "model": model_ids["tts"],
            "input": source_text,
            "voice": "vivian",
        })
        assert tts_resp.status_code == 200
        wav_bytes = tts_resp.content
        assert wav_bytes[:4] == b"RIFF"

        # Step 2: ASR
        asr_resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": model_ids["asr"], "language": "en"},
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        )
        assert asr_resp.status_code == 200
        transcript = asr_resp.json()["text"].strip().lower()
        assert len(transcript) > 0, "ASR produced empty transcript from TTS audio"

        # Check key words survived the round trip
        key_words = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
        matched = [w for w in key_words if w in transcript]
        assert len(matched) >= 4, (
            f"Round-trip lost too much content. "
            f"Source: {source_text!r}  Transcript: {transcript!r}  "
            f"Matched {len(matched)}/6 key words: {matched}"
        )
