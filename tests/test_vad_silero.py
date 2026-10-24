from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice.engines.vad_silero import SileroVADEngine


@pytest.mark.asyncio
async def test_loads_configured_onnx_model(tmp_path):
    model_path = tmp_path / "silero_vad.onnx"
    model_path.touch()
    engine = SileroVADEngine(onnx_path=str(model_path))

    with patch("silero_vad.utils_vad.OnnxWrapper") as wrapper:
        await engine._load_impl()

    wrapper.assert_called_once_with(str(model_path), force_onnx_cpu=True)


def test_is_speech_uses_upstream_model_signature():
    engine = SileroVADEngine(threshold=0.5)
    engine._model = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.8)))
    audio = np.ones(512, dtype=np.float32)

    assert engine.is_speech(audio, sample_rate=16000)
    called_audio, called_rate = engine._model.call_args.args
    np.testing.assert_array_equal(called_audio, audio)
    assert called_rate == 16000


def test_segments_uses_official_timestamp_helper():
    engine = SileroVADEngine(threshold=0.6, min_speech_ms=300)
    engine._model = object()
    audio = np.zeros(16000, dtype=np.float32)

    with patch(
        "silero_vad.get_speech_timestamps",
        return_value=[{"start": 0.1, "end": 0.9}],
    ) as get_timestamps:
        result = engine.segments(
            audio,
            sample_rate=16000,
            min_silence_ms=500,
            speech_pad_ms=100,
        )

    assert [(segment.start, segment.end) for segment in result] == [(0.1, 0.9)]
    assert get_timestamps.call_args.kwargs == {
        "threshold": 0.6,
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 100,
        "sampling_rate": 16000,
        "return_seconds": True,
    }
