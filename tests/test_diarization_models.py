"""Tests for diarization with pyannote.

The integration test is skipped unless HF_TOKEN is available (and the gated models have
been accepted): it verifies that the real path (lazily importing pyannote, loading audio
into memory, running the pipeline, and unpacking the 4.x output) does not crash and
returns the expected types. The configuration test always runs with a double in
sys.modules.
"""
import importlib.util
import os
import subprocess
import sys
import types

import pytest


# The guard requires BOTH things. With only HF_TOKEN, the test started in an environment
# without the [diarize] extra and died while importing pyannote: a failure unrelated to
# the code. `pyannote` is looked up before `pyannote.audio` because find_spec on a dotted
# name imports the parent and RAISES when it is absent: with HF_TOKEN set and only the
# [nemotron] extra, the whole file failed to collect.
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN")
    or importlib.util.find_spec("pyannote") is None
    or importlib.util.find_spec("pyannote.audio") is None,
    reason="requires HF_TOKEN + accepted gated models + the [diarize] extra",
)
def test_diarize_returns_turns_and_embeddings(tmp_path):
    from speechtotext.speakers.diarization import diarize

    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-ar", "16000", "-ac", "1", str(wav),
        ],
        check=True,
        capture_output=True,
    )
    from speechtotext.speakers.diarization import read_wav

    samples, rate = read_wav(wav)
    turns, embeddings = diarize(samples, rate)
    assert isinstance(turns, list)
    assert isinstance(embeddings, dict)


def test_get_pipeline_reduces_the_batch_sizes(monkeypatch):
    """The batch controls pyannote's peak RAM, and the checkpoint specifies 32 in its
    config.yaml. If someone deletes both assignments, the peak quietly doubles: no
    output test fails, but it consumes another 1.2 GB. This test is the guardrail."""
    from speechtotext.speakers import diarization

    created = []

    class _PipelineDouble:
        @staticmethod
        def from_pretrained(name, token=None):
            # The double lacks the attributes: if _get_pipeline does not assign them, the
            # assertion below raises AttributeError instead of passing by accident.
            obj = types.SimpleNamespace()
            created.append((name, token))
            return obj

    fake = types.ModuleType("pyannote.audio")
    fake.Pipeline = _PipelineDouble
    # Importing pyannote.audio is lazy (inside _get_pipeline), so seeding sys.modules
    # before the call avoids loading the real torch/pyannote.
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake)
    # The singleton is global: monkeypatch restores it and does not contaminate other tests.
    monkeypatch.setattr(diarization, "_PIPELINE", None)

    pipe = diarization._get_pipeline()

    assert created == [(diarization.EMBEDDING_MODEL, os.environ.get("HF_TOKEN"))]
    assert pipe.embedding_batch_size == diarization._BATCH == 8
    assert pipe.segmentation_batch_size == diarization._BATCH == 8
    assert diarization._get_pipeline() is pipe  # It remains a singleton.
    assert len(created) == 1


def test_read_wav_averages_channels_and_normalizes(tmp_path):
    import wave

    import numpy as np

    from speechtotext.speakers.diarization import read_wav

    left = np.full(100, 16384, dtype="<i2")
    right = np.full(100, -16384, dtype="<i2")
    stereo = np.column_stack([left, right]).reshape(-1)
    wav = tmp_path / "st.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(stereo.tobytes())
    samples, rate = read_wav(wav)
    assert rate == 8000 and samples.dtype == np.float32 and samples.shape == (100,)
    assert float(np.abs(samples).max()) == 0.0  # The average of +0.5 and -0.5.


def test_waveform_for_pyannote_is_a_1xN_tensor():
    torch = pytest.importorskip("torch")
    import numpy as np

    from speechtotext.speakers.diarization import _waveform

    wf = _waveform(np.zeros(160, dtype=np.float32), 16000)
    assert wf["sample_rate"] == 16000
    assert isinstance(wf["waveform"], torch.Tensor) and tuple(wf["waveform"].shape) == (1, 160)


def test_waveform_for_pyannote_is_writable(monkeypatch):
    """The decoded samples reach pyannote read-only, and `torch.from_numpy`
    warns about a read-only buffer on every diarized run, into the user's
    terminal. The stand-in torch turns that warning into a failure, so this
    runs where torch is not installed."""
    import sys
    import types

    import numpy as np

    from speechtotext.speakers.diarization import _waveform

    class Tensor:
        def __init__(self, arr):
            self.arr = arr

        def unsqueeze(self, dim):
            return self

    def from_numpy(arr):
        assert arr.flags.writeable, "torch.from_numpy was handed a read-only array"
        return Tensor(arr)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=from_numpy))

    samples = np.zeros(160, dtype=np.float32)
    samples.flags.writeable = False
    wf = _waveform(samples, 16000)
    assert wf["sample_rate"] == 16000
    assert np.array_equal(wf["waveform"].arr, samples)
