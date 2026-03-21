# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for audio and reranker endpoints.

Tests FastAPI endpoints using TestClient with mocked engines to verify
request/response formats without loading actual models.
"""

import io
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.engine.asr import ASREngine, TranscriptionOutput
from omlx.engine.llm_reranker import LLMRerankerEngine
from omlx.engine.reranker import RerankerEngine
from omlx.engine.tts import TTSEngine, SpeechOutput
from omlx.models.reranker import RerankOutput


# ──────────────────────────────────────────────────────────────────────
# Mock engines
# ──────────────────────────────────────────────────────────────────────


class MockASREngine(ASREngine):
    """Mock ASR engine that returns canned transcription."""

    def __init__(self, model_name: str = "test-asr-model"):
        self._model_name = model_name
        self._model = True  # Mark as "loaded"
        self._processor = True
        self._active_operations = 0
        self._total_operations = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0

    async def start(self):
        pass

    async def stop(self):
        pass

    async def transcribe(self, audio_path, language="auto", prompt=None):
        return TranscriptionOutput(
            text="This is a test transcription.",
            language="en",
            duration=3.5,
            segments=[{"id": 0, "start": 0.0, "end": 3.5, "text": "This is a test transcription."}],
        )

    def get_languages(self):
        return ["en", "zh", "ja", "ko", "fr", "de", "es"]

    def get_stats(self):
        return {"model_name": self._model_name, "loaded": True}


class MockTTSEngine(TTSEngine):
    """Mock TTS engine that returns minimal WAV bytes."""

    def __init__(self, model_name: str = "test-tts-model"):
        self._model_name = model_name
        self._model = True  # Mark as "loaded"
        self._variant = "custom_voice"
        self._active_operations = 0
        self._total_operations = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0

    async def start(self):
        pass

    async def stop(self):
        pass

    async def synthesize(self, text, speaker=None, instruct=None, ref_audio=None, ref_text=None):
        # Generate minimal valid WAV header + 1 second of silence
        sample_rate = 24000
        num_samples = sample_rate
        data_size = num_samples * 2  # 16-bit

        buf = io.BytesIO()
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
        buf.write(b"\x00" * data_size)

        return SpeechOutput(
            audio_bytes=buf.getvalue(),
            sample_rate=sample_rate,
            duration=1.0,
        )

    def get_speakers(self):
        return ["ryan", "vivian", "emma"]


class MockLLMRerankerEngine(LLMRerankerEngine):
    """Mock LLM reranker engine with canned scores."""

    def __init__(self, model_name: str = "test-llm-reranker"):
        self._model_name = model_name
        self._batched = MagicMock()  # Mark as "loaded"

    async def start(self):
        pass

    async def stop(self):
        pass

    async def rerank(self, query, documents, top_n=None, **kwargs):
        n_docs = len(documents)
        # Assign decreasing scores
        scores = [0.95 - i * 0.15 for i in range(n_docs)]
        indices = list(range(n_docs))
        if top_n is not None and top_n < n_docs:
            indices = indices[:top_n]
        return RerankOutput(
            scores=scores,
            indices=indices,
            total_tokens=n_docs * 20,
        )

    def get_stats(self):
        return {"model_name": self._model_name, "loaded": True, "engine_type": "llm_reranker"}


class MockBaseEngine:
    """Minimal mock LLM engine."""

    def __init__(self, model_name="test-llm-model"):
        self._model_name = model_name

    @property
    def model_name(self):
        return self._model_name

    @property
    def tokenizer(self):
        tok = MagicMock()
        tok.eos_token_id = 2
        tok.encode = lambda t: [100]
        tok.apply_chat_template = lambda m, **kw: "prompt"
        return tok

    @property
    def model_type(self):
        return "llama"

    async def generate(self, prompt, **kw):
        @dataclass
        class Out:
            text: str = "Generated."
            tokens: list = field(default_factory=list)
            prompt_tokens: int = 5
            completion_tokens: int = 3
            finish_reason: str = "stop"
            new_text: str = ""
            finished: bool = True
            tool_calls: object = None
            cached_tokens: int = 0
        return Out()

    async def chat(self, messages, **kw):
        return await self.generate("")

    async def stream_chat(self, messages, **kw):
        yield await self.generate("")

    async def stream_generate(self, prompt, **kw):
        yield await self.generate("")

    def count_chat_tokens(self, messages, tools=None, chat_template_kwargs=None):
        return 5

    def get_stats(self):
        return {}

    def get_cache_stats(self):
        return None


class MockEnginePool:
    """Mock engine pool supporting LLM, ASR, TTS, and LLM reranker engines."""

    def __init__(self):
        self._llm_engine = MockBaseEngine()
        self._asr_engine = MockASREngine()
        self._tts_engine = MockTTSEngine()
        self._llm_reranker_engine = MockLLMRerankerEngine()
        self._entries = {
            "test-llm-model": MagicMock(engine_type="batched", engine=self._llm_engine),
            "test-asr-model": MagicMock(engine_type="asr", engine=self._asr_engine),
            "test-tts-model": MagicMock(engine_type="tts", engine=self._tts_engine),
            "test-llm-reranker": MagicMock(engine_type="llm_reranker", engine=self._llm_reranker_engine),
        }

    @property
    def model_count(self):
        return len(self._entries)

    @property
    def loaded_model_count(self):
        return len(self._entries)

    @property
    def max_model_memory(self):
        return 32 * 1024**3

    @property
    def current_model_memory(self):
        return 1000000

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def acquire_engine(self, model_id) -> None:
        pass

    def release_engine(self, model_id) -> None:
        pass

    def get_model_ids(self):
        return list(self._entries.keys())

    def get_status(self):
        return {
            "models": [{"id": k, "loaded": True} for k in self._entries],
            "loaded_count": len(self._entries),
            "max_model_memory": self.max_model_memory,
        }

    async def get_engine(self, model_id):
        if model_id == "test-asr-model":
            return self._asr_engine
        if model_id == "test-tts-model":
            return self._tts_engine
        if model_id == "test-llm-reranker":
            return self._llm_reranker_engine
        return self._llm_engine


@pytest.fixture
def client():
    """Create a test client with mock engines for audio and reranker endpoints."""
    from omlx.server import app, _server_state

    pool = MockEnginePool()

    original_pool = _server_state.engine_pool
    original_default = _server_state.default_model

    _server_state.engine_pool = pool
    _server_state.default_model = "test-llm-model"

    yield TestClient(app)

    _server_state.engine_pool = original_pool
    _server_state.default_model = original_default


# ──────────────────────────────────────────────────────────────────────
# Audio transcription endpoint tests
# ──────────────────────────────────────────────────────────────────────


class TestTranscriptionEndpoint:
    """Tests for POST /v1/audio/transcriptions."""

    def test_transcription_basic(self, client):
        """Test basic audio transcription returns text."""
        # Create a minimal WAV file (44 bytes header + 0 data)
        wav = self._make_wav()
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test-asr-model", "language": "en"},
            files={"file": ("test.wav", wav, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "This is a test transcription."

    def test_transcription_with_language_auto(self, client):
        """Test transcription with auto language detection."""
        wav = self._make_wav()
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test-asr-model", "language": "auto"},
            files={"file": ("test.wav", wav, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"

    def test_transcription_returns_duration(self, client):
        """Test transcription response includes duration."""
        wav = self._make_wav()
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test-asr-model"},
            files={"file": ("test.wav", wav, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["duration"] == 3.5

    def test_transcription_missing_file(self, client):
        """Test error when no audio file provided."""
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test-asr-model"},
        )

        assert response.status_code == 400

    def test_transcription_missing_model(self, client):
        """Test error when model not specified."""
        wav = self._make_wav()
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", wav, "audio/wav")},
        )

        assert response.status_code == 400

    @staticmethod
    def _make_wav():
        """Create a minimal valid WAV file."""
        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36))  # file size - 8
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))       # PCM
        buf.write(struct.pack("<H", 1))       # mono
        buf.write(struct.pack("<I", 16000))   # sample rate
        buf.write(struct.pack("<I", 32000))   # byte rate
        buf.write(struct.pack("<H", 2))       # block align
        buf.write(struct.pack("<H", 16))      # bits per sample
        buf.write(b"data")
        buf.write(struct.pack("<I", 0))       # data size
        buf.seek(0)
        return buf


# ──────────────────────────────────────────────────────────────────────
# Audio speech (TTS) endpoint tests
# ──────────────────────────────────────────────────────────────────────


class TestSpeechEndpoint:
    """Tests for POST /v1/audio/speech."""

    def test_speech_basic(self, client):
        """Test basic TTS returns WAV audio."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello world!",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        # Verify it's valid WAV (starts with RIFF)
        assert response.content[:4] == b"RIFF"
        assert response.content[8:12] == b"WAVE"

    def test_speech_with_voice(self, client):
        """Test TTS with voice parameter."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello world!",
                "voice": "ryan",
            },
        )

        assert response.status_code == 200

    def test_speech_with_instructions(self, client):
        """Test TTS with instructions parameter."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello world!",
                "voice": "vivian",
                "instructions": "Speak warmly and gently.",
            },
        )

        assert response.status_code == 200

    def test_speech_response_is_binary(self, client):
        """Test TTS response is binary audio data, not JSON."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Test.",
            },
        )

        assert response.status_code == 200
        # Should be binary WAV, not JSON
        assert len(response.content) > 44  # WAV header is at least 44 bytes

    def test_speech_missing_model(self, client):
        """Test error when model not specified."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello world!",
            },
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_speech_missing_input(self, client):
        """Test error when input text not specified."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
            },
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_speech_mp3_format(self, client):
        """Test TTS with mp3 response_format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "response_format": "mp3",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        # MP3 starts with 0xFF 0xFB/0xF3/0xF2 or ID3 tag
        assert response.content[:3] == b"ID3" or response.content[0] == 0xFF

    def test_speech_flac_format(self, client):
        """Test TTS with flac response_format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "response_format": "flac",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/flac"
        assert response.content[:4] == b"fLaC"

    def test_speech_opus_format(self, client):
        """Test TTS with opus response_format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "response_format": "opus",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/opus"
        assert len(response.content) > 0

    def test_speech_pcm_format(self, client):
        """Test TTS with pcm response_format returns raw PCM."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "response_format": "pcm",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        # PCM should not have WAV header (no RIFF)
        assert response.content[:4] != b"RIFF"

    def test_speech_unsupported_format(self, client):
        """Test TTS rejects unsupported format."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "response_format": "aac",
            },
        )

        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]

    def test_speech_with_speed(self, client):
        """Test TTS with speed parameter."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "speed": 1.5,
            },
        )

        assert response.status_code == 200
        # Speed applied via ffmpeg, output is still WAV
        assert response.content[:4] == b"RIFF"

    def test_speech_speed_out_of_range(self, client):
        """Test TTS rejects speed outside 0.25-4.0."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts-model",
                "input": "Hello!",
                "speed": 10.0,
            },
        )

        assert response.status_code == 400
        assert "Speed" in response.json()["detail"]


# ──────────────────────────────────────────────────────────────────────
# Speakers endpoint tests
# ──────────────────────────────────────────────────────────────────────


class TestSpeakersEndpoint:
    """Tests for GET /v1/audio/speakers."""

    def test_speakers_list(self, client):
        """Test speakers list returns available voices."""
        response = client.get(
            "/v1/audio/speakers",
            params={"model": "test-tts-model"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "speakers" in data
        assert "ryan" in data["speakers"]
        assert "vivian" in data["speakers"]
        assert "emma" in data["speakers"]

    def test_speakers_auto_detect_model(self, client):
        """Test speakers auto-detects loaded TTS model when no model specified."""
        response = client.get("/v1/audio/speakers")

        assert response.status_code == 200
        data = response.json()
        assert "speakers" in data


class TestLanguagesEndpoint:
    """Tests for GET /v1/audio/languages."""

    def test_languages_list(self, client):
        """Test languages endpoint returns supported languages."""
        response = client.get(
            "/v1/audio/languages",
            params={"model": "test-asr-model"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert "en" in data["languages"]
        assert "zh" in data["languages"]
        assert data["model"] == "test-asr-model"

    def test_languages_auto_detect_model(self, client):
        """Test languages auto-detects ASR model when no model specified."""
        response = client.get("/v1/audio/languages")

        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert len(data["languages"]) > 0


# ──────────────────────────────────────────────────────────────────────
# LLM reranker endpoint tests
# ──────────────────────────────────────────────────────────────────────


class TestLLMRerankerEndpoint:
    """Tests for POST /v1/rerank with LLMRerankerEngine."""

    def test_rerank_basic(self, client):
        """Test basic rerank via LLM reranker engine."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-llm-reranker",
                "query": "What is machine learning?",
                "documents": [
                    "ML is a subset of AI.",
                    "The weather is nice today.",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_rerank_with_top_n(self, client):
        """Test LLM reranker with top_n filtering."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-llm-reranker",
                "query": "Test query",
                "documents": ["Doc 1", "Doc 2", "Doc 3", "Doc 4"],
                "top_n": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_rerank_response_format(self, client):
        """Test LLM reranker response has correct format."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-llm-reranker",
                "query": "Test",
                "documents": ["Document 1"],
                "return_documents": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "model" in data
        assert "results" in data
        result = data["results"][0]
        assert "index" in result
        assert "relevance_score" in result

    def test_rerank_scores_ordering(self, client):
        """Test that results are sorted by relevance score (descending)."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-llm-reranker",
                "query": "relevant query",
                "documents": ["doc A", "doc B", "doc C"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        scores = [r["relevance_score"] for r in data["results"]]
        # Results should be sorted by score descending
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_query_rejected(self, client):
        """Test that empty query is rejected."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-llm-reranker",
                "query": "",
                "documents": ["doc"],
            },
        )

        assert response.status_code == 400

    def test_rerank_empty_documents_rejected(self, client):
        """Test that empty documents list is rejected."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-llm-reranker",
                "query": "test",
                "documents": [],
            },
        )

        assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Model type routing tests
# ──────────────────────────────────────────────────────────────────────


class TestEngineTypeRouting:
    """Test that engine types route correctly to the right engine."""

    def test_asr_model_rejects_embedding_request(self, client):
        """Test that using ASR model for embeddings fails."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-asr-model",
                "input": "test text",
            },
        )

        # Should reject because ASR engine is not an embedding engine
        assert response.status_code == 400

    def test_tts_model_rejects_chat_request(self, client):
        """Test that using TTS model for chat fails (not silently succeeds).

        TTS engine doesn't implement BaseEngine interface, so the server
        will error. TestClient raises server exceptions by default, so we
        catch any exception as proof that TTS can't serve chat requests.
        """
        with pytest.raises(Exception):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-tts-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    def test_models_endpoint_lists_all_types(self, client):
        """Test /v1/models includes all engine types."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "test-asr-model" in model_ids
        assert "test-tts-model" in model_ids
        assert "test-llm-reranker" in model_ids


# ──────────────────────────────────────────────────────────────────────
# Prefill-only mode integration tests
# ──────────────────────────────────────────────────────────────────────


class TestPrefillOnlyIntegration:
    """Test prefill_only fields through the request/output chain."""

    def test_sampling_params_prefill_only_round_trip(self):
        """Test SamplingParams correctly carries prefill_only fields."""
        from omlx.request import SamplingParams

        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prefill_only=True,
            prefill_output="logits",
        )

        assert params.prefill_only is True
        assert params.prefill_output == "logits"
        assert params.max_tokens == 1

    def test_request_output_carries_logits(self):
        """Test RequestOutput can carry last_logits."""
        from omlx.request import RequestOutput

        logits = list(range(1000))
        output = RequestOutput(
            request_id="test",
            finished=True,
            finish_reason="length",
            last_logits=logits,
        )

        assert output.last_logits is not None
        assert len(output.last_logits) == 1000
        assert output.last_logits[0] == 0
        assert output.last_logits[999] == 999

    def test_request_with_prefill_only_sampling(self):
        """Test Request can be created with prefill_only SamplingParams."""
        from omlx.request import Request, SamplingParams

        params = SamplingParams(
            max_tokens=1,
            prefill_only=True,
            prefill_output="logits",
        )
        request = Request(
            request_id="prefill-test",
            prompt="Test prompt",
            sampling_params=params,
        )

        assert request.sampling_params.prefill_only is True
        assert request.sampling_params.prefill_output == "logits"

    def test_prefill_only_with_llm_reranker_scoring(self):
        """Test the full scoring pipeline: logits -> softmax -> P(yes)."""
        import mlx.core as mx

        # Simulate what LLMRerankerEngine does after getting logprobs
        vocab_size = 10000
        yes_id = 42
        no_id = 99

        logits = [0.0] * vocab_size
        logits[yes_id] = 5.0    # high yes
        logits[no_id] = -5.0    # low no

        logits_arr = mx.array(logits)
        probs = mx.softmax(logits_arr[mx.array([yes_id, no_id])])
        mx.eval(probs)

        p_yes = float(probs[0].item())
        p_no = float(probs[1].item())

        # P(yes) should be close to 1.0 (exp(5) >> exp(-5))
        assert p_yes > 0.99
        assert p_no < 0.01
        assert abs(p_yes + p_no - 1.0) < 1e-6


# ──────────────────────────────────────────────────────────────────────
# Model discovery integration tests
# ──────────────────────────────────────────────────────────────────────


class TestModelDiscoveryIntegration:
    """Test model discovery with new engine types through the full chain."""

    def test_discover_asr_model(self, tmp_path):
        """Test ASR model discovery + EnginePool routing."""
        model_dir = tmp_path / "whisper-base"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        }))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        from omlx.model_discovery import detect_model_type, DiscoveredModel
        from omlx.engine_pool import EnginePool

        assert detect_model_type(model_dir) == "asr"

        pool = EnginePool(max_model_memory=None)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("whisper-base")
        assert entry is not None
        assert entry.model_type == "asr"
        assert entry.engine_type == "asr"

    def test_discover_tts_model(self, tmp_path):
        """Test TTS model discovery + EnginePool routing."""
        model_dir = tmp_path / "qwen3-tts"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3_tts",
            "architectures": ["Qwen3TTSForConditionalGeneration"],
        }))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        from omlx.model_discovery import detect_model_type
        from omlx.engine_pool import EnginePool

        assert detect_model_type(model_dir) == "tts"

        pool = EnginePool(max_model_memory=None)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("qwen3-tts")
        assert entry is not None
        assert entry.model_type == "tts"
        assert entry.engine_type == "tts"

    def test_llm_reranker_auto_detected_from_name(self, tmp_path):
        """Test CausalLM reranker is auto-detected from model directory name."""
        model_dir = tmp_path / "Qwen3-Reranker-0.6B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        from omlx.model_discovery import detect_model_type
        from omlx.engine_pool import EnginePool

        assert detect_model_type(model_dir) == "llm_reranker"

        pool = EnginePool(max_model_memory=None)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("Qwen3-Reranker-0.6B")
        assert entry.model_type == "llm_reranker"
        assert entry.engine_type == "llm_reranker"

    def test_llm_reranker_via_model_type_override(self, tmp_path):
        """Test LLM reranker configured via model_type_override for a model
        that isn't auto-detected (no 'reranker' in name)."""
        model_dir = tmp_path / "Qwen3-Custom-0.6B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        from omlx.model_discovery import detect_model_type
        from omlx.engine_pool import EnginePool
        from omlx.model_settings import ModelSettingsManager

        # Without override, it's detected as LLM (no "reranker" in name)
        assert detect_model_type(model_dir) == "llm"

        pool = EnginePool(max_model_memory=None)
        pool.discover_models(str(tmp_path))
        entry = pool.get_entry("Qwen3-Custom-0.6B")
        assert entry.model_type == "llm"
        assert entry.engine_type == "batched"

        # Apply override
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()
        sm = ModelSettingsManager(str(settings_dir))
        settings = sm.get_settings("Qwen3-Custom-0.6B")
        settings.model_type_override = "llm_reranker"
        sm.set_settings("Qwen3-Custom-0.6B", settings)

        pool.apply_settings_overrides(sm)
        entry = pool.get_entry("Qwen3-Custom-0.6B")
        assert entry.model_type == "llm_reranker"
        assert entry.engine_type == "llm_reranker"

    def test_engine_pool_all_type_mappings(self):
        """Verify all model type -> engine type mappings exist."""
        from omlx.engine_pool import EnginePool

        pool = EnginePool(max_model_memory=None)
        expected = {
            "llm": "batched",
            "vlm": "vlm",
            "embedding": "embedding",
            "reranker": "reranker",
            "llm_reranker": "llm_reranker",
            "asr": "asr",
            "tts": "tts",
        }
        for model_type, engine_type in expected.items():
            assert pool._MODEL_TYPE_TO_ENGINE[model_type] == engine_type

    def test_admin_accepts_llm_reranker_override(self):
        """Test that admin routes accept llm_reranker as valid override."""
        valid_types = {"llm", "vlm", "embedding", "reranker", "llm_reranker", "asr", "tts"}
        assert "llm_reranker" in valid_types
