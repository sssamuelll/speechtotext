# tests/test_audio_io.py
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from speechtotext.audio.io import AudioDecodeError, decode_audio

class FakeFrame:
    def __init__(self, values):
        self._values = np.asarray([values], dtype=np.float32)

    def to_ndarray(self):
        return self._values


class FakeContainer:
    def __init__(self, frames, has_audio=True):
        self.frames = frames
        self.streams = SimpleNamespace(audio=[object()] if has_audio else [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def decode(self, stream):
        return iter(self.frames)


class FakeResampler:
    def __init__(self, format, layout, rate):
        assert (format, layout, rate) == ("flt", "mono", 16000)

    def resample(self, frame):
        return [] if frame is None else [frame]


def _av(frames, has_audio=True):
    opened = []

    def open_stream(stream):
        assert hasattr(stream, "read") and hasattr(stream, "seek")
        opened.append(stream)
        return FakeContainer(frames, has_audio)

    return SimpleNamespace(
        open=open_stream,
        AudioResampler=FakeResampler,
        opened=opened,
    )


def test_decode_audio_concatena_float32_mono():
    lease = io.BytesIO(b"leased-audio")
    av_module = _av([FakeFrame([0.1, 0.2]), FakeFrame([0.3])])
    view = decode_audio(
        lease,
        sample_rate=16000,
        av_module=av_module,
    )
    assert av_module.opened == [lease]
    assert view.sample_rate == 16000
    assert view.samples.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert view.provenance.steps[0].name == "pyav-decode"


def test_decode_audio_rechaza_archivo_sin_stream():
    with pytest.raises(AudioDecodeError, match="stream de audio"):
        decode_audio(
            io.BytesIO(b"leased-video"),
            sample_rate=16000,
            av_module=_av([], has_audio=False),
        )


def test_decode_audio_rechaza_decode_vacio():
    with pytest.raises(AudioDecodeError, match="sin muestras"):
        decode_audio(
            io.BytesIO(b"leased-empty"),
            sample_rate=16000,
            av_module=_av([]),
        )


@pytest.mark.parametrize(
    "invalid_stream",
    ["sample.wav", Path("sample.wav"), io.StringIO("not-binary")],
)
def test_decode_audio_rechaza_paths_y_texto_antes_de_av_open(invalid_stream):
    av_module = _av([])
    with pytest.raises(TypeError, match="stream binario leased"):
        decode_audio(invalid_stream, sample_rate=16000, av_module=av_module)
    assert av_module.opened == []
