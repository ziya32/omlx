# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.engine.asr and omlx.engine.tts modules."""

import struct
from unittest.mock import MagicMock, patch

import pytest


def _make_stt_module(**overrides):
    """Create a mock mlx_audio.stt module with sensible defaults."""
    mock_stt = MagicMock()
    mock_stt.load.return_value = MagicMock(name="model")
    for k, v in overrides.items():
        setattr(mock_stt, k, v)
    return mock_stt


def _make_stt_output(text="", language=None, segments=None):
    """Create a mock STTOutput."""
    output = MagicMock()
    output.text = text
    output.language = language
    output.segments = segments
    return output


def _patch_stt(mock_stt):
    """Context manager that patches sys.modules so ``from mlx_audio import stt`` works."""
    mlx_audio_mock = MagicMock()
    mlx_audio_mock.stt = mock_stt
    return patch.dict("sys.modules", {
        "mlx_audio": mlx_audio_mock,
        "mlx_audio.stt": mock_stt,
    })


def _patch_stt_generate(mock_gen):
    """Context manager that patches sys.modules so ``from mlx_audio.stt.generate import generate_transcription`` works."""
    mock_generate_module = MagicMock()
    mock_generate_module.generate_transcription = mock_gen
    mock_stt = MagicMock()
    mock_stt.generate = mock_generate_module
    mlx_audio_mock = MagicMock()
    mlx_audio_mock.stt = mock_stt
    mlx_audio_mock.stt.generate = mock_generate_module
    return patch.dict("sys.modules", {
        "mlx_audio": mlx_audio_mock,
        "mlx_audio.stt": mock_stt,
        "mlx_audio.stt.generate": mock_generate_module,
    })


# ---------------------------------------------------------------------------
# ASR Engine Tests
# ---------------------------------------------------------------------------

class TestASREngineInit:
    """Test ASREngine construction and properties."""

    def test_init(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="mlx-community/whisper-large-v3")
        assert engine.model_name == "mlx-community/whisper-large-v3"
        assert engine._model is None

    def test_repr_stopped(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        assert "stopped" in repr(engine)
        assert "whisper-tiny" in repr(engine)

    def test_repr_running(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()
        assert "running" in repr(engine)

    def test_get_stats_not_loaded(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        stats = engine.get_stats()
        assert stats["model_name"] == "whisper-tiny"
        assert stats["loaded"] is False

    def test_get_stats_loaded(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()
        stats = engine.get_stats()
        assert stats["loaded"] is True

    def test_get_model_info_not_loaded(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        info = engine.get_model_info()
        assert info["loaded"] is False
        assert info["model_name"] == "whisper-tiny"

    def test_get_model_info_loaded(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()
        info = engine.get_model_info()
        assert info["loaded"] is True
        assert info["model_name"] == "whisper-tiny"


class TestASREngineLifecycle:
    """Test ASREngine start/stop with mocked mlx_audio.stt."""

    async def test_start_loads_model(self):
        from omlx.engine.asr import ASREngine

        mock_model = MagicMock(name="model")
        mock_stt = _make_stt_module()
        mock_stt.load.return_value = mock_model

        engine = ASREngine(model_name="whisper-tiny")
        with _patch_stt(mock_stt):
            await engine.start()

        assert engine._model is mock_model
        mock_stt.load.assert_called_once_with("whisper-tiny")

    async def test_start_idempotent(self):
        """Calling start() twice should not re-load."""
        from omlx.engine.asr import ASREngine

        mock_stt = _make_stt_module()
        engine = ASREngine(model_name="whisper-tiny")
        with _patch_stt(mock_stt):
            await engine.start()
            await engine.start()

        mock_stt.load.assert_called_once()

    @patch("omlx.engine.asr.mx")
    @patch("omlx.engine.asr.gc")
    async def test_stop_clears_model(self, mock_gc, mock_mx):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        await engine.stop()

        assert engine._model is None
        mock_gc.collect.assert_called_once()
        # mx.clear_cache is called via the MLX executor (run_in_executor)
        mock_mx.clear_cache.assert_called_once()

    async def test_stop_when_not_started_is_noop(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        await engine.stop()
        assert engine._model is None


class TestASREngineTranscribe:
    """Test ASREngine.transcribe with mocked generate_transcription."""

    async def test_transcribe_not_started_raises(self):
        from omlx.engine.asr import ASREngine
        engine = ASREngine(model_name="whisper-tiny")
        with pytest.raises(RuntimeError, match="Engine not started"):
            await engine.transcribe("/tmp/audio.wav")

    async def test_transcribe_auto_language(self):
        from omlx.engine.asr import ASREngine, TranscriptionOutput

        mock_gen = MagicMock(return_value=_make_stt_output(
            text="Hello world",
            language="en",
            segments=[{"start": 0.0, "end": 3.5}],
        ))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock(name="model")

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav", language="auto")

        assert isinstance(result, TranscriptionOutput)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 3.5
        # language="auto" should not pass language kwarg
        mock_gen.assert_called_once_with(
            model=engine._model,
            audio="/tmp/audio.wav",
        )

    async def test_transcribe_explicit_language(self):
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(
            text="Bonjour",
            language="fr",
        ))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav", language="fr")

        assert result.text == "Bonjour"
        assert result.language == "fr"
        assert result.duration is None
        mock_gen.assert_called_once_with(
            model=engine._model,
            audio="/tmp/audio.wav",
            language="fr",
        )

    async def test_transcribe_no_segments(self):
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(text="test"))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav")

        assert result.text == "test"
        assert result.language is None
        assert result.duration is None

    async def test_transcribe_empty_segments(self):
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(text="test", segments=[]))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav")

        assert result.duration is None

    async def test_transcribe_empty_result(self):
        """Handles result with no text gracefully."""
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(text=None))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav")

        assert result.text == ""

    async def test_transcribe_multi_segment_duration(self):
        """Duration comes from the last segment's end time."""
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(
            text="one two three",
            language="en",
            segments=[
                {"start": 0.0, "end": 1.0},
                {"start": 1.0, "end": 2.5},
                {"start": 2.5, "end": 5.2},
            ],
        ))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav")

        assert result.duration == 5.2

    async def test_transcribe_language_as_list(self):
        """Handles language returned as list (newer mlx_audio versions)."""
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(
            text="Hello",
            language=["en"],
        ))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav")

        assert result.language == "en"

    async def test_transcribe_language_none_string(self):
        """Handles language='None' string from silent audio."""
        from omlx.engine.asr import ASREngine

        mock_gen = MagicMock(return_value=_make_stt_output(
            text="",
            language=["None"],
        ))

        engine = ASREngine(model_name="whisper-tiny")
        engine._model = MagicMock()

        with _patch_stt_generate(mock_gen):
            result = await engine.transcribe("/tmp/audio.wav")

        assert result.language is None


# ---------------------------------------------------------------------------
# TTS Engine Tests
# ---------------------------------------------------------------------------

class TestTTSEngineInit:
    """Test TTSEngine construction and properties."""

    def test_init(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="mlx-community/qwen3-tts")
        assert engine.model_name == "mlx-community/qwen3-tts"
        assert engine._model is None
        assert engine._variant == "custom_voice"

    def test_repr_stopped(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        assert "stopped" in repr(engine)
        assert "qwen3-tts" in repr(engine)
        assert "custom_voice" in repr(engine)

    def test_repr_running(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        engine._model = MagicMock()
        assert "running" in repr(engine)

    def test_get_stats_not_loaded(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        stats = engine.get_stats()
        assert stats["model_name"] == "qwen3-tts"
        assert stats["loaded"] is False
        assert stats["variant"] == "custom_voice"

    def test_get_stats_loaded(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        engine._model = MagicMock()
        stats = engine.get_stats()
        assert stats["loaded"] is True

    def test_get_model_info_not_loaded(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        info = engine.get_model_info()
        assert info["loaded"] is False
        assert info["model_name"] == "qwen3-tts"

    def test_get_model_info_loaded_custom_voice(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.talker_config = {"spk_id": {"ryan": 0, "vivian": 1}}
        mock_model.config = mock_config
        engine._model = mock_model
        engine._variant = "custom_voice"

        info = engine.get_model_info()
        assert info["loaded"] is True
        assert info["variant"] == "custom_voice"
        assert set(info["speakers"]) == {"ryan", "vivian"}

    def test_get_model_info_loaded_voice_design(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts-vd")
        engine._model = MagicMock()
        engine._variant = "voice_design"

        info = engine.get_model_info()
        assert info["loaded"] is True
        assert info["variant"] == "voice_design"
        assert info["speakers"] == []


class TestTTSEngineVariantDetection:
    """Test _detect_variant logic."""

    def test_detect_custom_voice_default(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="test")
        # Non-existent path returns default
        assert engine._detect_variant("/nonexistent/path") == "custom_voice"

    def test_detect_voice_design(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        config_path = tmp_path / "config.json"
        config_path.write_text('{"tts_model_type": "voice_design"}')

        engine = TTSEngine(model_name="test")
        assert engine._detect_variant(str(tmp_path)) == "voice_design"

    def test_detect_base(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        config_path = tmp_path / "config.json"
        config_path.write_text('{"tts_model_type": "base"}')

        engine = TTSEngine(model_name="test")
        assert engine._detect_variant(str(tmp_path)) == "base"

    def test_detect_custom_voice_from_config(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        config_path = tmp_path / "config.json"
        config_path.write_text('{"tts_model_type": "custom_voice"}')

        engine = TTSEngine(model_name="test")
        # tts_model_type == "custom_voice" doesn't match "voice_design" or "base",
        # so falls through to default
        assert engine._detect_variant(str(tmp_path)) == "custom_voice"

    def test_detect_invalid_json(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        config_path = tmp_path / "config.json"
        config_path.write_text("NOT JSON")

        engine = TTSEngine(model_name="test")
        assert engine._detect_variant(str(tmp_path)) == "custom_voice"

    def test_detect_no_tts_model_type_key(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "qwen2"}')

        engine = TTSEngine(model_name="test")
        assert engine._detect_variant(str(tmp_path)) == "custom_voice"


class TestTTSEngineLifecycle:
    """Test TTSEngine start/stop with mocked mlx_audio.tts."""

    async def test_start_loads_model(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        mock_model = MagicMock()

        mock_load_model = MagicMock(return_value=mock_model)
        mock_tts_utils = MagicMock()
        mock_tts_utils.load_model = mock_load_model

        # Write config.json so _detect_variant can read it
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"tts_model_type": "voice_design"}')

        with patch.dict("sys.modules", {
            "mlx_audio": MagicMock(),
            "mlx_audio.tts": MagicMock(),
            "mlx_audio.tts.utils": mock_tts_utils,
        }):
            engine = TTSEngine(model_name=str(model_dir))
            await engine.start()

        assert engine._model is mock_model
        assert engine._variant == "voice_design"

    async def test_start_idempotent(self, tmp_path):
        from omlx.engine.tts import TTSEngine
        mock_load_model = MagicMock(return_value=MagicMock())
        mock_tts_utils = MagicMock()
        mock_tts_utils.load_model = mock_load_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        with patch.dict("sys.modules", {
            "mlx_audio": MagicMock(),
            "mlx_audio.tts": MagicMock(),
            "mlx_audio.tts.utils": mock_tts_utils,
        }):
            engine = TTSEngine(model_name=str(model_dir))
            await engine.start()
            await engine.start()

        mock_load_model.assert_called_once()

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts.gc")
    async def test_stop_clears_model(self, mock_gc, mock_mx):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        engine._model = MagicMock()

        await engine.stop()

        assert engine._model is None
        mock_gc.collect.assert_called_once()
        mock_mx.clear_cache.assert_called_once()

    async def test_stop_when_not_started_is_noop(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        await engine.stop()
        assert engine._model is None


class TestTTSEngineSpeakers:
    """Test TTSEngine.get_speakers."""

    def test_get_speakers_not_loaded(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        assert engine.get_speakers() == []

    def test_get_speakers_voice_design_returns_empty(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts-vd")
        engine._model = MagicMock()
        engine._variant = "voice_design"
        assert engine.get_speakers() == []

    def test_get_speakers_custom_voice(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.talker_config = {"spk_id": {"ryan": 0, "vivian": 1, "alex": 2}}
        mock_model.config = mock_config
        engine._model = mock_model
        engine._variant = "custom_voice"

        speakers = engine.get_speakers()
        assert set(speakers) == {"ryan", "vivian", "alex"}

    def test_get_speakers_no_talker_config(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=[])  # no talker_config attr
        engine._model = mock_model
        engine._variant = "custom_voice"

        assert engine.get_speakers() == []

    def test_get_speakers_no_spk_id(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.talker_config = {"other_key": "val"}
        mock_model.config = mock_config
        engine._model = mock_model
        engine._variant = "custom_voice"

        assert engine.get_speakers() == []


class TestTTSEngineSynthesize:
    """Test TTSEngine.synthesize with mocked model."""

    async def test_synthesize_not_started_raises(self):
        from omlx.engine.tts import TTSEngine
        engine = TTSEngine(model_name="qwen3-tts")
        with pytest.raises(RuntimeError, match="Engine not started"):
            await engine.synthesize("Hello")

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_custom_voice_defaults(self, mock_audio_to_wav, mock_mx):
        from omlx.engine.tts import TTSEngine, SpeechOutput

        engine = TTSEngine(model_name="qwen3-tts")
        engine._variant = "custom_voice"

        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_audio = MagicMock()
        mock_audio.shape = (24000,)
        mock_segment = MagicMock()
        mock_segment.audio = mock_audio
        mock_model.generate.return_value = iter([mock_segment])
        engine._model = mock_model

        mock_audio_to_wav.return_value = b"RIFF....WAV"

        result = await engine.synthesize("Hello world")

        assert isinstance(result, SpeechOutput)
        assert result.audio_bytes == b"RIFF....WAV"
        assert result.sample_rate == 24000
        assert result.duration == 1.0  # 24000 samples / 24000 Hz
        # Check generate was called with voice="vivian" (default)
        mock_model.generate.assert_called_once_with(text="Hello world", voice="vivian")

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_custom_voice_with_speaker_and_instruct(self, mock_audio_to_wav, mock_mx):
        from omlx.engine.tts import TTSEngine

        engine = TTSEngine(model_name="qwen3-tts")
        engine._variant = "custom_voice"

        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_audio = MagicMock()
        mock_audio.shape = (48000,)
        mock_segment = MagicMock()
        mock_segment.audio = mock_audio
        mock_model.generate.return_value = iter([mock_segment])
        engine._model = mock_model

        mock_audio_to_wav.return_value = b"WAV"

        result = await engine.synthesize("Hi", speaker="ryan", instruct="Speak warmly")

        mock_model.generate.assert_called_once_with(
            text="Hi", voice="ryan", instruct="Speak warmly"
        )
        assert result.duration == 2.0  # 48000 / 24000

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_voice_design(self, mock_audio_to_wav, mock_mx):
        from omlx.engine.tts import TTSEngine

        engine = TTSEngine(model_name="qwen3-tts-vd")
        engine._variant = "voice_design"

        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_audio = MagicMock()
        mock_audio.shape = (12000,)
        mock_segment = MagicMock()
        mock_segment.audio = mock_audio
        mock_model.generate.return_value = iter([mock_segment])
        engine._model = mock_model

        mock_audio_to_wav.return_value = b"WAV"

        result = await engine.synthesize(
            "Goodbye",
            instruct="A warm female narrator with gentle pace",
        )

        mock_model.generate.assert_called_once_with(
            text="Goodbye",
            instruct="A warm female narrator with gentle pace",
        )
        assert result.duration == 0.5  # 12000 / 24000

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_voice_design_default_instruct(self, mock_audio_to_wav, mock_mx):
        from omlx.engine.tts import TTSEngine

        engine = TTSEngine(model_name="qwen3-tts-vd")
        engine._variant = "voice_design"

        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_audio = MagicMock()
        mock_audio.shape = (24000,)
        mock_segment = MagicMock()
        mock_segment.audio = mock_audio
        mock_model.generate.return_value = iter([mock_segment])
        engine._model = mock_model

        mock_audio_to_wav.return_value = b"WAV"

        await engine.synthesize("Test")

        # When instruct is None, "A neutral narrator" should be used
        mock_model.generate.assert_called_once_with(
            text="Test",
            instruct="A neutral narrator",
        )

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_multiple_segments(self, mock_audio_to_wav, mock_mx):
        from omlx.engine.tts import TTSEngine

        engine = TTSEngine(model_name="qwen3-tts")
        engine._variant = "custom_voice"

        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        seg1 = MagicMock()
        seg2 = MagicMock()
        seg3 = MagicMock()
        mock_model.generate.return_value = iter([seg1, seg2, seg3])
        engine._model = mock_model

        mock_audio = MagicMock()
        mock_audio.shape = (72000,)
        mock_mx.concatenate.return_value = mock_audio
        mock_audio_to_wav.return_value = b"WAV"

        result = await engine.synthesize("Long text")

        # Verify concatenation receives all 3 segment audios
        concat_args = mock_mx.concatenate.call_args[0][0]
        assert len(concat_args) == 3
        assert result.duration == 3.0  # 72000 / 24000

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_custom_sample_rate(self, mock_audio_to_wav, mock_mx):
        """If model reports a different sample_rate, it should be respected."""
        from omlx.engine.tts import TTSEngine

        engine = TTSEngine(model_name="qwen3-tts")
        engine._variant = "custom_voice"

        mock_model = MagicMock()
        mock_model.sample_rate = 48000  # custom rate
        mock_audio = MagicMock()
        mock_audio.shape = (48000,)
        mock_segment = MagicMock()
        mock_segment.audio = mock_audio
        mock_model.generate.return_value = iter([mock_segment])
        engine._model = mock_model

        mock_audio_to_wav.return_value = b"WAV"

        result = await engine.synthesize("Hi")

        assert result.sample_rate == 48000
        assert result.duration == 1.0  # 48000 / 48000
        mock_audio_to_wav.assert_called_once_with(mock_audio, 48000)

    @patch("omlx.engine.tts.mx")
    @patch("omlx.engine.tts._audio_to_wav")
    async def test_synthesize_custom_voice_no_instruct(self, mock_audio_to_wav, mock_mx):
        """When instruct is None for custom_voice, it should not be passed."""
        from omlx.engine.tts import TTSEngine

        engine = TTSEngine(model_name="qwen3-tts")
        engine._variant = "custom_voice"

        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_audio = MagicMock()
        mock_audio.shape = (24000,)
        mock_segment = MagicMock()
        mock_segment.audio = mock_audio
        mock_model.generate.return_value = iter([mock_segment])
        engine._model = mock_model

        mock_audio_to_wav.return_value = b"WAV"

        await engine.synthesize("Hi", speaker="ryan")

        # instruct not provided -> should NOT appear in generate kwargs
        mock_model.generate.assert_called_once_with(text="Hi", voice="ryan")


class TestAudioToWav:
    """Test the _audio_to_wav helper function."""

    def test_wav_header_structure(self):
        import numpy as np
        from omlx.engine.tts import _audio_to_wav

        samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav = _audio_to_wav(samples, 24000)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"

        # Verify data size
        data_size = struct.unpack("<I", wav[40:44])[0]
        assert data_size == 5 * 2  # 5 samples * 2 bytes (int16)

    def test_wav_total_size(self):
        import numpy as np
        from omlx.engine.tts import _audio_to_wav

        samples = np.array([0.0] * 100, dtype=np.float32)
        wav = _audio_to_wav(samples, 24000)

        # Total file = 44 byte header + data
        assert len(wav) == 44 + 100 * 2

    def test_wav_clamping(self):
        """Values outside [-1, 1] should be clamped."""
        import numpy as np
        from omlx.engine.tts import _audio_to_wav

        samples = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        wav = _audio_to_wav(samples, 24000)

        # Extract PCM data after 44-byte header
        pcm_data = wav[44:]
        pcm = np.frombuffer(pcm_data, dtype=np.int16)
        assert pcm[0] == 32767   # clamped to 1.0
        assert pcm[1] == -32767  # clamped to -1.0 -> -32767
        assert pcm[2] == 0

    def test_wav_sample_rate_in_header(self):
        """Sample rate should appear in the WAV header."""
        import numpy as np
        from omlx.engine.tts import _audio_to_wav

        samples = np.array([0.0], dtype=np.float32)
        wav = _audio_to_wav(samples, 44100)

        # Sample rate is at offset 24
        sr = struct.unpack("<I", wav[24:28])[0]
        assert sr == 44100

    def test_wav_mono_16bit(self):
        """Verify WAV format fields: mono, 16-bit PCM."""
        import numpy as np
        from omlx.engine.tts import _audio_to_wav

        samples = np.array([0.1], dtype=np.float32)
        wav = _audio_to_wav(samples, 24000)

        # PCM format = 1
        fmt = struct.unpack("<H", wav[20:22])[0]
        assert fmt == 1
        # num channels = 1
        channels = struct.unpack("<H", wav[22:24])[0]
        assert channels == 1
        # bits per sample = 16
        bps = struct.unpack("<H", wav[34:36])[0]
        assert bps == 16


class TestTranscriptionOutput:
    """Test TranscriptionOutput dataclass."""

    def test_defaults(self):
        from omlx.engine.asr import TranscriptionOutput
        out = TranscriptionOutput(text="hello")
        assert out.text == "hello"
        assert out.language is None
        assert out.duration is None

    def test_all_fields(self):
        from omlx.engine.asr import TranscriptionOutput
        out = TranscriptionOutput(text="bonjour", language="fr", duration=2.5)
        assert out.text == "bonjour"
        assert out.language == "fr"
        assert out.duration == 2.5


class TestSpeechOutput:
    """Test SpeechOutput dataclass."""

    def test_defaults(self):
        from omlx.engine.tts import SpeechOutput
        out = SpeechOutput(audio_bytes=b"wav")
        assert out.audio_bytes == b"wav"
        assert out.sample_rate == 24000
        assert out.duration == 0.0

    def test_all_fields(self):
        from omlx.engine.tts import SpeechOutput
        out = SpeechOutput(audio_bytes=b"data", sample_rate=48000, duration=3.0)
        assert out.audio_bytes == b"data"
        assert out.sample_rate == 48000
        assert out.duration == 3.0


def _make_wav_bytes(sample_rate=24000, duration_s=0.1) -> bytes:
    """Generate minimal valid WAV bytes for testing."""
    import io as _io
    num_samples = int(sample_rate * duration_s)
    data_size = num_samples * 2
    buf = _io.BytesIO()
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
    buf.write(b"\x00" * data_size)
    return buf.getvalue()


class TestConvertWav:
    """Tests for _convert_wav format conversion."""

    def test_wav_to_mp3(self):
        from omlx.engine.tts import _convert_wav
        wav = _make_wav_bytes()
        mp3 = _convert_wav(wav, "mp3")
        # MP3 starts with 0xFF 0xFB or ID3 tag
        assert mp3[:3] == b"ID3" or mp3[0] == 0xFF
        assert len(mp3) > 0

    def test_wav_to_flac(self):
        from omlx.engine.tts import _convert_wav
        wav = _make_wav_bytes()
        flac = _convert_wav(wav, "flac")
        assert flac[:4] == b"fLaC"

    def test_wav_to_opus(self):
        from omlx.engine.tts import _convert_wav
        wav = _make_wav_bytes()
        opus = _convert_wav(wav, "opus")
        assert len(opus) > 0

    def test_wav_to_wav_with_speed(self):
        from omlx.engine.tts import _convert_wav
        wav = _make_wav_bytes(duration_s=0.5)
        fast = _convert_wav(wav, "wav", speed=2.0)
        # Sped-up audio should be shorter (fewer bytes)
        assert fast[:4] == b"RIFF"
        assert len(fast) < len(wav)

    def test_wav_to_mp3_with_speed(self):
        from omlx.engine.tts import _convert_wav
        wav = _make_wav_bytes(duration_s=0.5)
        mp3 = _convert_wav(wav, "mp3", speed=0.5)
        assert mp3[:3] == b"ID3" or mp3[0] == 0xFF

    def test_invalid_input_raises(self):
        from omlx.engine.tts import _convert_wav
        from omlx.exceptions import AudioError
        with pytest.raises(AudioError, match="ffmpeg conversion failed"):
            _convert_wav(b"not audio data", "mp3")


class TestSupportedFormats:
    """Tests for format constants."""

    def test_all_formats_have_media_types(self):
        from omlx.engine.tts import _SUPPORTED_FORMATS, _FORMAT_MEDIA_TYPES
        for fmt in _SUPPORTED_FORMATS:
            assert fmt in _FORMAT_MEDIA_TYPES, f"Missing media type for {fmt}"

    def test_all_formats_have_extensions(self):
        from omlx.engine.tts import _SUPPORTED_FORMATS, _FORMAT_EXTENSIONS
        for fmt in _SUPPORTED_FORMATS:
            assert fmt in _FORMAT_EXTENSIONS, f"Missing extension for {fmt}"
