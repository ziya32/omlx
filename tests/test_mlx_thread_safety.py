# SPDX-License-Identifier: Apache-2.0
"""
Tests that all engine MLX/Metal operations are serialized on the global
MLX executor thread.

The oMLX architecture requires ALL MLX GPU operations to run on a single
thread (the global MLX executor) to prevent Metal command buffer races.
See engine_core.py issue #85.

These tests verify this invariant by monkey-patching the executor and
checking that MLX operations in TTS, STT, Embedding, and other engines
actually dispatch to the executor rather than running inline on the
asyncio event loop thread.
"""

import asyncio
import threading
import wave
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def wav_path(tmp_path):
    """Create a minimal valid WAV file (0.1s of silence, 16kHz mono 16-bit)."""
    path = tmp_path / "test_audio.wav"
    sample_rate = 16000
    n_samples = 1600  # 0.1 seconds
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return str(path)


def _get_executor_thread_name() -> str:
    """Return the name of the MLX executor thread."""
    from omlx.engine_core import get_mlx_executor

    executor = get_mlx_executor()
    # Run a no-op to discover the thread name
    future = executor.submit(lambda: threading.current_thread().name)
    return future.result(timeout=5)


class TestTTSThreadSafety:
    """Verify TTSEngine operations run on the MLX executor thread."""

    async def test_start_loads_on_executor_thread(self, tmp_path):
        """TTS model loading must happen on the MLX executor thread."""
        from omlx.engine.tts import TTSEngine

        load_thread_name = None

        def _mock_load_model(path, **kwargs):
            nonlocal load_thread_name
            load_thread_name = threading.current_thread().name
            return MagicMock()

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        mock_tts_utils = MagicMock()
        mock_tts_utils.load_model = _mock_load_model

        with patch.dict("sys.modules", {
            "mlx_audio": MagicMock(),
            "mlx_audio.tts": MagicMock(),
            "mlx_audio.tts.utils": mock_tts_utils,
        }):
            engine = TTSEngine(model_name=str(model_dir))
            await engine.start()

        executor_thread = _get_executor_thread_name()
        assert load_thread_name == executor_thread, (
            f"TTS load_model ran on thread '{load_thread_name}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )

    async def test_synthesize_runs_on_executor_thread(self):
        """TTS synthesis (model.generate + mx.eval) must run on the executor thread."""
        import numpy as np
        from omlx.engine.tts import TTSEngine

        generate_thread_name = None

        mock_model = MagicMock()
        mock_model.sample_rate = 24000

        def _mock_generate(**kwargs):
            nonlocal generate_thread_name
            generate_thread_name = threading.current_thread().name
            seg = MagicMock()
            seg.audio = np.zeros(24000, dtype=np.float32)
            seg.sample_rate = 24000
            return iter([seg])

        mock_model.generate = _mock_generate

        engine = TTSEngine(model_name="test-tts")
        engine._model = mock_model
        engine._variant = "custom_voice"

        with patch("omlx.engine.tts._audio_to_wav_bytes", return_value=b"WAV"):
            await engine.synthesize("Hello")

        executor_thread = _get_executor_thread_name()
        assert generate_thread_name == executor_thread, (
            f"TTS model.generate ran on thread '{generate_thread_name}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )

    async def test_stop_clears_cache_on_executor_thread(self):
        """TTS mx.clear_cache must run on the executor thread."""
        from omlx.engine.tts import TTSEngine

        clear_cache_thread = None
        original_clear_cache = None

        def _track_clear_cache():
            nonlocal clear_cache_thread
            clear_cache_thread = threading.current_thread().name

        engine = TTSEngine(model_name="test-tts")
        engine._model = MagicMock()

        with patch("omlx.engine.tts.mx") as mock_mx, \
             patch("omlx.engine.tts.gc"):
            mock_mx.clear_cache = _track_clear_cache
            await engine.stop()

        executor_thread = _get_executor_thread_name()
        assert clear_cache_thread == executor_thread, (
            f"TTS mx.clear_cache ran on thread '{clear_cache_thread}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )


class TestSTTThreadSafety:
    """Verify STTEngine operations run on the MLX executor thread."""

    async def test_start_loads_on_executor_thread(self):
        """STT model loading must happen on the MLX executor thread."""
        from omlx.engine.stt import STTEngine

        load_thread_name = None

        def _mock_stt_load(model_name):
            nonlocal load_thread_name
            load_thread_name = threading.current_thread().name
            return MagicMock()

        mock_stt = MagicMock()
        mock_stt.load = _mock_stt_load

        with patch.dict("sys.modules", {
            "mlx_audio": MagicMock(stt=mock_stt),
            "mlx_audio.stt": mock_stt,
        }):
            engine = STTEngine(model_name="whisper-tiny")
            await engine.start()

        executor_thread = _get_executor_thread_name()
        assert load_thread_name == executor_thread, (
            f"STT stt.load ran on thread '{load_thread_name}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )

    async def test_transcribe_runs_on_executor_thread(self, wav_path):
        """STT transcription must run on the executor thread."""
        from omlx.engine.stt import STTEngine

        transcribe_thread_name = None

        def _mock_generate_transcription(**kwargs):
            nonlocal transcribe_thread_name
            transcribe_thread_name = threading.current_thread().name
            output = MagicMock()
            output.text = "hello"
            output.language = "en"
            output.segments = [{"start": 0.0, "end": 1.0}]
            return output

        mock_gen_module = MagicMock()
        mock_gen_module.generate_transcription = _mock_generate_transcription

        engine = STTEngine(model_name="whisper-tiny")
        engine._model = MagicMock(sample_rate=16000)

        with patch.dict("sys.modules", {
            "mlx_audio": MagicMock(),
            "mlx_audio.stt": MagicMock(),
            "mlx_audio.stt.generate": mock_gen_module,
        }):
            result = await engine.transcribe(wav_path)

        executor_thread = _get_executor_thread_name()
        assert transcribe_thread_name == executor_thread, (
            f"STT generate_transcription ran on thread '{transcribe_thread_name}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )

    async def test_stop_clears_cache_on_executor_thread(self):
        """STT mx.clear_cache must run on the executor thread."""
        from omlx.engine.stt import STTEngine

        clear_cache_thread = None

        def _track_clear_cache():
            nonlocal clear_cache_thread
            clear_cache_thread = threading.current_thread().name

        engine = STTEngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with patch("omlx.engine.stt.mx") as mock_mx, \
             patch("omlx.engine.stt.gc"):
            mock_mx.clear_cache = _track_clear_cache
            await engine.stop()

        executor_thread = _get_executor_thread_name()
        assert clear_cache_thread == executor_thread, (
            f"STT mx.clear_cache ran on thread '{clear_cache_thread}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )


class TestEmbeddingThreadSafety:
    """Verify EmbeddingEngine operations run on the MLX executor thread."""

    async def test_embed_runs_on_executor_thread(self):
        """Embedding forward pass must run on the executor thread."""
        from omlx.engine.embedding import EmbeddingEngine

        embed_thread_name = None

        mock_embed_model = MagicMock()

        def _mock_embed(**kwargs):
            nonlocal embed_thread_name
            embed_thread_name = threading.current_thread().name
            return MagicMock()

        mock_embed_model.embed = _mock_embed

        engine = EmbeddingEngine(model_name="test-embed")
        engine._model = mock_embed_model

        await engine.embed(["hello world"])

        executor_thread = _get_executor_thread_name()
        assert embed_thread_name == executor_thread, (
            f"Embedding.embed ran on thread '{embed_thread_name}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )

    async def test_stop_clears_cache_on_executor_thread(self):
        """Embedding mx.clear_cache must run on the executor thread."""
        from omlx.engine.embedding import EmbeddingEngine

        clear_cache_thread = None

        def _track_clear_cache():
            nonlocal clear_cache_thread
            clear_cache_thread = threading.current_thread().name

        engine = EmbeddingEngine(model_name="test-embed")
        engine._model = MagicMock()

        with patch("omlx.engine.embedding.mx") as mock_mx, \
             patch("omlx.engine.embedding.gc"):
            mock_mx.clear_cache = _track_clear_cache
            await engine.stop()

        executor_thread = _get_executor_thread_name()
        assert clear_cache_thread == executor_thread, (
            f"Embedding mx.clear_cache ran on thread '{clear_cache_thread}', "
            f"expected MLX executor thread '{executor_thread}'. "
            f"This would cause Metal command buffer races!"
        )
