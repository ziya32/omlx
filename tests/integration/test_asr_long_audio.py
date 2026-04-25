# SPDX-License-Identifier: Apache-2.0
"""
E2E test for ASR transcription of a long audio file with concurrent embedding.

Uses a 7-hour interview audio (tests/fixtures/sample_speech.wav) downloaded
from https://www.youtube.com/watch?v=rIwgZWzUKm8 — a marathon interview with
Saining Xie covering world models, AMI Labs, Yann LeCun, Fei-Fei Li, and more.

The test exercises:
1. The full /v1/audio/transcriptions endpoint via a real omlx server
2. Long audio chunking (> 20 min triggers per-chunk executor yielding)
3. Concurrent embedding requests during ASR to verify executor yielding
4. Transcript saved to tests/fixtures/sample_speech_transcript.json

Requirements:
- tests/fixtures/sample_speech.wav (download with yt-dlp, see docs)
- Qwen3-ASR and Qwen3-Embedding models in ~/.myemee/models/
- pytest -m slow tests/integration/test_asr_long_audio.py -v
"""

import asyncio
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

try:
    import mlx.core as mx

    _total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.metal.set_memory_limit(int(_total_bytes * 0.75))
except Exception:
    pass

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="ASR E2E tests require macOS with Apple Silicon",
    ),
    pytest.mark.asyncio(loop_scope="session"),
]

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
AUDIO_PATH = FIXTURE_DIR / "sample_speech.wav"
TRANSCRIPT_PATH = FIXTURE_DIR / "sample_speech_transcript.json"

_env_dir = os.environ.get("OMLX_MODEL_DIR")
MODEL_DIR = Path(_env_dir) if _env_dir else Path.home() / ".myemee" / "models"


def _find_model(patterns: list[str], max_gb: float = 8.0) -> Optional[str]:
    """Find smallest model matching any pattern."""
    if not MODEL_DIR.exists():
        return None
    candidates = []
    for d in MODEL_DIR.iterdir():
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        name = d.name.lower()
        if any(p.lower() in name for p in patterns):
            size = sum(f.stat().st_size for f in d.glob("*.safetensors"))
            size_gb = size / (1024**3)
            if size_gb <= max_gb:
                candidates.append((size_gb, d.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def audio_file():
    if not AUDIO_PATH.exists():
        pytest.skip(
            f"Audio fixture not found: {AUDIO_PATH}. "
            "Download with: yt-dlp -x --audio-format wav "
            "-o tests/fixtures/sample_speech.wav "
            "https://www.youtube.com/watch?v=rIwgZWzUKm8"
        )
    return AUDIO_PATH


@pytest.fixture(scope="session")
def model_ids():
    asr = _find_model(["ASR", "asr", "Whisper", "whisper"], max_gb=4.0)
    embedding = _find_model(["Embedding"], max_gb=8.0)
    missing = []
    if not asr:
        missing.append("ASR (Qwen3-ASR)")
    if not embedding:
        missing.append("Embedding (Qwen3-Embedding)")
    if missing:
        pytest.skip(f"Missing models: {', '.join(missing)}")
    return {"asr": asr, "embedding": embedding}


TEST_API_KEY = "test-asr-long-audio-key"


@pytest_asyncio.fixture(loop_scope="session", scope="module")
async def server_app(model_ids):
    """Start a real omlx server with ASR and embedding models.

    **Module-scoped** so the loaded ASR + embedding engines are torn
    down as soon as the last test in this module finishes, rather
    than persisting until the end of the whole pytest session and
    coexisting with other integration modules' engine pools.
    """
    from omlx.model_settings import ModelSettings, ModelSettingsManager
    from omlx.server import _server_state, app, init_server

    max_mem = int(
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.75
    )

    with tempfile.TemporaryDirectory() as tmp:
        settings_mgr = ModelSettingsManager(Path(tmp))
        # Qwen3-Embedding uses CausalLM architecture — override to embedding
        settings_mgr.set_settings(
            model_ids["embedding"],
            ModelSettings(model_type_override="embedding"),
        )

        init_server(model_dirs=str(MODEL_DIR), max_model_memory=max_mem)
        _server_state.api_key = TEST_API_KEY
        _server_state.engine_pool.apply_settings_overrides(settings_mgr)

        try:
            yield app
        finally:
            # Drain in-flight work, stop background tasks, unload every
            # engine, clear Metal cache.
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
    """Async HTTP client for the real omlx server.

    Uses a 300s timeout — with SSE streaming each chunk resets the read
    timeout, so no single wait exceeds this even for multi-hour audio.
    """
    transport = ASGITransport(app=server_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=httpx.Timeout(300.0),
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    ) as ac:
        yield ac


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestASRWithConcurrentEmbedding:
    """
    Transcribe a long audio file while firing embedding requests concurrently.

    This validates that the ASR engine's per-chunk executor yielding allows
    other engine types (embedding) to make progress during long transcriptions.
    """

    async def test_transcribe_with_concurrent_embeddings(
        self, client, model_ids, audio_file
    ):
        """
        Start a long ASR transcription via SSE streaming, then send embedding
        requests while it's running. Both should succeed — proving executor
        yielding works and the streaming keepalive prevents timeout.
        """
        embedding_results: list[dict] = []
        progress_events: list[dict] = []
        embedding_model = model_ids["embedding"]
        asr_model = model_ids["asr"]

        async def _do_transcription():
            """Send the full 7-hour audio for streaming transcription."""
            asr_data = None
            with open(audio_file, "rb") as f:
                async with client.stream(
                    "POST",
                    "/v1/audio/transcriptions",
                    data={
                        "model": asr_model,
                        "response_format": "verbose_json",
                        "stream": "true",
                    },
                    files={"file": ("speech.wav", f, "audio/wav")},
                ) as resp:
                    assert resp.status_code == 200
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        event = json.loads(payload)
                        if event.get("type") == "progress":
                            progress_events.append(event)
                            chunk = event["chunk"]
                            total = event["total_chunks"]
                            print(f"  ASR chunk {chunk}/{total}")
                        elif event.get("type") == "transcription":
                            asr_data = event
                        elif event.get("type") == "error":
                            pytest.fail(f"ASR error: {event.get('error')}")
            return asr_data

        async def _do_embedding(text: str, idx: int):
            """Send an embedding request and record timing."""
            start = time.perf_counter()
            resp = await client.post(
                "/v1/embeddings",
                json={"model": embedding_model, "input": text},
            )
            elapsed = time.perf_counter() - start
            embedding_results.append({
                "idx": idx,
                "status": resp.status_code,
                "elapsed": elapsed,
                "dim": len(resp.json().get("data", [{}])[0].get("embedding", []))
                if resp.status_code == 200
                else 0,
            })
            return resp

        async def _fire_embeddings_during_asr():
            """Wait a bit for ASR to start, then fire embedding requests."""
            # Give ASR time to start processing
            await asyncio.sleep(3.0)

            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by biological neurons.",
                "Deep learning uses many layers of neural networks.",
                "Computer vision enables machines to interpret images.",
                "Natural language processing deals with human language.",
            ]

            for i, text in enumerate(texts):
                await _do_embedding(text, i)
                # Small gap between requests
                await asyncio.sleep(1.0)

        # Run transcription and embeddings concurrently
        asr_task = asyncio.create_task(_do_transcription())
        embed_task = asyncio.create_task(_fire_embeddings_during_asr())

        # Wait for both
        asr_data, _ = await asyncio.gather(asr_task, embed_task)

        # ── Verify streaming progress ──
        assert len(progress_events) > 0, "No progress events received"
        print(f"\nReceived {len(progress_events)} progress events")

        # ── Verify ASR results ──
        assert asr_data is not None, "No transcription result received"
        assert "text" in asr_data
        assert len(asr_data["text"]) > 100, "Transcription too short"
        assert "segments" in asr_data
        assert len(asr_data["segments"]) > 0

        # Check for expected AI/ML keywords
        text_lower = asr_data["text"].lower()
        assert any(
            kw in text_lower
            for kw in ["model", "learning", "ai", "research", "neural", "data"]
        ), f"Missing expected keywords in: {asr_data['text'][:300]}"

        # Save transcript to file
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            json.dump(asr_data, f, ensure_ascii=False, indent=2)
        print(f"\nTranscript saved to {TRANSCRIPT_PATH}")
        print(f"Transcript length: {len(asr_data['text'])} chars")
        print(f"Segments: {len(asr_data['segments'])}")
        if asr_data.get("duration"):
            hours = asr_data["duration"] / 3600
            print(f"Duration: {asr_data['duration']:.0f}s ({hours:.1f} hours)")

        # ── Verify embedding results ──
        assert len(embedding_results) == 5, (
            f"Expected 5 embedding results, got {len(embedding_results)}"
        )
        for r in embedding_results:
            assert r["status"] == 200, (
                f"Embedding #{r['idx']} failed with status {r['status']}"
            )
            assert r["dim"] > 0, f"Embedding #{r['idx']} returned empty vector"
            print(
                f"Embedding #{r['idx']}: {r['elapsed']:.2f}s, dim={r['dim']}"
            )

        # Embeddings should complete in reasonable time (< 60s each).
        # If executor yielding is broken, they'd be blocked for the
        # entire ASR duration (potentially minutes).
        max_embed_time = max(r["elapsed"] for r in embedding_results)
        assert max_embed_time < 60.0, (
            f"Slowest embedding took {max_embed_time:.1f}s — "
            "executor yielding may not be working"
        )
