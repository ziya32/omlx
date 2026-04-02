# SPDX-License-Identifier: Apache-2.0
"""
End-to-end tests for audio and reranker engines with real models.

Tests:
- LLM-based reranker via /v1/rerank with a real Qwen3-Reranker model
- Prefill-only mode through the full engine stack
- Audio endpoints with real ASR/TTS models (if available)

These tests are marked with @pytest.mark.slow and are skipped by default.
Run with: pytest -m slow tests/integration/test_audio_reranker_e2e.py -v

Requirements:
- Apple Silicon (M1/M2/M3/M4)
- Model files in ~/myemee/models/ or OMLX_MODEL_DIR
- For reranker tests: Qwen3-Reranker model (auto-detected)
- For audio tests: Whisper/Qwen3-ASR and/or Qwen3-TTS models
"""

import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="E2E tests require macOS with Apple Silicon",
    ),
]

# Cap Metal memory
try:
    import mlx.core as mx

    _total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.metal.set_memory_limit(int(_total_bytes * 0.75))
except Exception:
    pass


def _find_model(patterns: list[str], max_size_gb: float = 6.0, min_size_gb: float = 0.0) -> Optional[Path]:
    """Find a model matching any of the patterns, preferring smallest."""
    model_dir = os.environ.get("OMLX_MODEL_DIR")
    if model_dir:
        search_dirs = [Path(model_dir)]
    else:
        search_dirs = [
            Path.home() / ".myemee" / "models",
            Path.home() / "Workspace" / "models",
            Path.home() / "models",
        ]

    candidates = []
    for base in search_dirs:
        if not base.exists():
            continue
        # Two-level scan (flat + organization folders)
        for subdir in base.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            dirs_to_check = []
            if (subdir / "config.json").exists():
                dirs_to_check.append(subdir)
            else:
                for child in subdir.iterdir():
                    if child.is_dir() and (child / "config.json").exists():
                        dirs_to_check.append(child)

            for d in dirs_to_check:
                name = d.name.lower()
                if any(p.lower() in name for p in patterns):
                    size = sum(f.stat().st_size for f in d.glob("*.safetensors"))
                    size_gb = size / (1024**3)
                    if min_size_gb < size_gb <= max_size_gb:
                        candidates.append((size_gb, d))

    if not candidates:
        return None
    candidates.sort()  # smallest first
    return candidates[0][1]


def _get_model_type(model_path: Path) -> str:
    """Read model_type from config.json."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("model_type", "")
    return ""


# ──────────────────────────────────────────────────────────────────────
# LLM Reranker E2E
# ──────────────────────────────────────────────────────────────────────


class TestRerankerE2E:
    """E2E tests for MLXRerankerModel with a real Qwen3-Reranker model."""

    @pytest.fixture(scope="class")
    def reranker_model(self):
        model = _find_model(["reranker", "Reranker"], min_size_gb=4.0, max_size_gb=20.0)
        if model is None:
            pytest.skip("No reranker model found (need Qwen3-Reranker-8B or similar)")
        return model

    @pytest.fixture(scope="class")
    def model(self, reranker_model):
        """Load the MLXRerankerModel once for all tests in this class."""
        from omlx.models.reranker import MLXRerankerModel

        model = MLXRerankerModel(str(reranker_model))
        model.load()
        yield model
        del model
        gc.collect()
        mx.clear_cache()

    def test_rerank_basic(self, model):
        """Test basic reranking produces valid scores."""
        output = model.rerank(
            query="What is machine learning?",
            documents=[
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "The weather forecast for tomorrow predicts rain in the afternoon.",
                "Deep learning uses neural networks with many layers to learn complex patterns.",
            ],
        )

        assert len(output.scores) == 3
        assert len(output.indices) == 3
        for score in output.scores:
            assert 0.0 <= score <= 1.0
        assert output.scores[0] > output.scores[1], (
            f"ML doc ({output.scores[0]:.3f}) should score higher than "
            f"weather doc ({output.scores[1]:.3f})"
        )
        assert output.scores[2] > output.scores[1], (
            f"DL doc ({output.scores[2]:.3f}) should score higher than "
            f"weather doc ({output.scores[1]:.3f})"
        )

    def test_rerank_top_n(self, model):
        """Test top_n filtering returns correct number of results."""
        output = model.rerank(
            query="Python programming",
            documents=[
                "Python is a high-level programming language.",
                "Java is another popular language.",
                "Cooking recipes for dinner.",
                "Python supports multiple paradigms.",
            ],
        )

        # top_n is handled by the engine layer, not the model — all scored
        assert len(output.scores) == 4

    def test_rerank_single_document(self, model):
        """Test reranking with a single document."""
        output = model.rerank(
            query="What is AI?",
            documents=["Artificial intelligence is the simulation of human intelligence."],
        )

        assert len(output.scores) == 1
        assert output.scores[0] > 0.5  # Should be relevant

    def test_rerank_token_count(self, model):
        """Test that token count is tracked."""
        output = model.rerank(
            query="test",
            documents=["doc one", "doc two"],
        )

        assert output.total_tokens > 0

    def test_rerank_with_instruction(self, model):
        """Test that custom instruction works."""
        output = model.rerank(
            query="A cartoon duck on a green background",
            documents=["A red stone monument with Chinese calligraphy"],
            instruction="Given a description of a photo, determine if the candidate describes the same visual content",
        )

        assert output.scores[0] < 0.1  # Should be very low — unrelated


# ──────────────────────────────────────────────────────────────────────
# ASR E2E (requires mlx-audio + ASR model)
# ──────────────────────────────────────────────────────────────────────


class TestASRE2E:
    """E2E tests for ASR engine with a real model."""

    @pytest.fixture(scope="class")
    def asr_model(self):
        model = _find_model(["whisper", "Whisper", "ASR", "asr"], max_size_gb=4.0)
        if model is None:
            pytest.skip("No ASR model found")
        return model

    @pytest.fixture(scope="class")
    def asr_engine(self, asr_model):
        """Load ASR engine (skips if mlx-audio not installed)."""
        import asyncio

        try:
            from omlx.engine.stt import STTEngine
        except ImportError:
            pytest.skip("mlx-audio not installed")

        engine = STTEngine(model_name=str(asr_model))
        try:
            asyncio.run(engine.start())
        except ImportError:
            pytest.skip("mlx-audio not installed")
        except Exception as e:
            pytest.skip(f"Failed to load ASR model: {e}")
        yield engine
        asyncio.run(engine.stop())
        gc.collect()
        mx.clear_cache()

    def test_asr_engine_loaded(self, asr_engine):
        """Test ASR engine loads successfully."""
        stats = asr_engine.get_stats()
        assert stats["loaded"] is True

    def test_asr_transcribe(self, asr_engine, tmp_path):
        """Test transcription with a generated silent audio file."""
        import asyncio
        import struct

        # Generate 1 second of silence as WAV
        wav_path = tmp_path / "silence.wav"
        sample_rate = 16000
        num_samples = sample_rate
        data_size = num_samples * 2

        with open(wav_path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))
            f.write(struct.pack("<H", 2))
            f.write(struct.pack("<H", 16))
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(b"\x00" * data_size)

        output = asyncio.run(
            asr_engine.transcribe(str(wav_path))
        )

        assert isinstance(output.text, str)
        # Silent audio should produce empty or very short transcription
        # (exact output depends on model)


# ──────────────────────────────────────────────────────────────────────
# TTS E2E (requires mlx-audio + TTS model)
# ──────────────────────────────────────────────────────────────────────


class TestTTSE2E:
    """E2E tests for TTS engine with a real model."""

    @pytest.fixture(scope="class")
    def tts_model(self):
        model = _find_model(["TTS", "tts", "Qwen3-TTS"], max_size_gb=6.0)
        if model is None:
            pytest.skip("No TTS model found")
        return model

    @pytest.fixture(scope="class")
    def tts_engine(self, tts_model):
        """Load TTS engine (skips if mlx-audio not installed)."""
        import asyncio

        try:
            from omlx.engine.tts import TTSEngine
        except ImportError:
            pytest.skip("mlx-audio not installed")

        engine = TTSEngine(model_name=str(tts_model))
        try:
            asyncio.run(engine.start())
        except ImportError:
            pytest.skip("mlx-audio not installed")
        except Exception as e:
            pytest.skip(f"Failed to load TTS model: {e}")
        yield engine
        asyncio.run(engine.stop())
        gc.collect()
        mx.clear_cache()

    def test_tts_engine_loaded(self, tts_engine):
        """Test TTS engine loads successfully."""
        stats = tts_engine.get_stats()
        assert stats["loaded"] is True

    def test_tts_synthesize(self, tts_engine):
        """Test speech synthesis produces valid WAV audio."""
        import asyncio

        output = asyncio.run(
            tts_engine.synthesize(text="Hello world.")
        )

        assert output.audio_bytes is not None
        assert len(output.audio_bytes) > 44  # At least a WAV header
        assert output.audio_bytes[:4] == b"RIFF"
        assert output.audio_bytes[8:12] == b"WAVE"
        assert output.sample_rate > 0
        assert output.duration > 0.0

    def test_tts_speakers(self, tts_engine):
        """Test speaker listing (CustomVoice models only)."""
        speakers = tts_engine.get_speakers()
        # May be empty for VoiceDesign models
        assert isinstance(speakers, list)


# ──────────────────────────────────────────────────────────────────────
# TTS → ASR round-trip (requires both TTS and ASR models)
# ──────────────────────────────────────────────────────────────────────


class TestTTSASRRoundTrip:
    """Round-trip test: synthesize speech with TTS, transcribe it back with ASR."""

    @pytest.fixture(scope="class")
    def tts_engine(self):
        import asyncio

        model = _find_model(["TTS", "tts", "Qwen3-TTS"], max_size_gb=6.0)
        if model is None:
            pytest.skip("No TTS model found")

        try:
            from omlx.engine.tts import TTSEngine
        except ImportError:
            pytest.skip("mlx-audio not installed")

        engine = TTSEngine(model_name=str(model))
        try:
            asyncio.run(engine.start())
        except Exception as e:
            pytest.skip(f"Failed to load TTS model: {e}")
        yield engine
        asyncio.run(engine.stop())
        gc.collect()
        mx.clear_cache()

    @pytest.fixture(scope="class")
    def asr_engine(self):
        import asyncio

        model = _find_model(["whisper", "Whisper", "ASR", "asr"], max_size_gb=4.0)
        if model is None:
            pytest.skip("No ASR model found")

        try:
            from omlx.engine.stt import STTEngine
        except ImportError:
            pytest.skip("mlx-audio not installed")

        engine = STTEngine(model_name=str(model))
        try:
            asyncio.run(engine.start())
        except Exception as e:
            pytest.skip(f"Failed to load ASR model: {e}")
        yield engine
        asyncio.run(engine.stop())
        gc.collect()
        mx.clear_cache()

    def test_tts_asr_round_trip(self, tts_engine, asr_engine, tmp_path):
        """Synthesize a sentence with TTS, then transcribe it with ASR."""
        import asyncio

        source_text = "The quick brown fox jumps over the lazy dog."

        # Step 1: TTS — text to audio
        tts_output = asyncio.run(
            tts_engine.synthesize(text=source_text)
        )
        assert tts_output.audio_bytes is not None
        assert tts_output.duration > 0.0

        # Write WAV to disk for ASR
        wav_path = tmp_path / "round_trip.wav"
        wav_path.write_bytes(tts_output.audio_bytes)

        # Step 2: ASR — audio back to text
        asr_output = asyncio.run(
            asr_engine.transcribe(str(wav_path))
        )

        transcript = asr_output.text.strip().lower()
        assert len(transcript) > 0, "ASR produced empty transcript from TTS audio"

        # Check that key words from the source appear in the transcript.
        # We don't require an exact match — models may differ in punctuation,
        # casing, or minor word variations — but the core content should survive.
        key_words = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
        matched = [w for w in key_words if w in transcript]
        assert len(matched) >= 4, (
            f"Round-trip lost too much content. "
            f"Source: {source_text!r}  Transcript: {transcript!r}  "
            f"Matched {len(matched)}/6 key words: {matched}"
        )
