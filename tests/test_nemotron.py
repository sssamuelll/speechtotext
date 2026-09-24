"""speakers.nemotron: the second diarizer (NVIDIA Nemotron-3-Diarization, through transformers).

The model itself is never loaded here. What runs everywhere is the part we own: probabilities
to turns, the dependency check that has to fail before an hour of transcription and not after,
and the guards that come before the model. The piecewise features are checked against the real
feature extractor, so that test only runs where the [nemotron] extra is installed.
"""
import importlib.metadata
import importlib.util

import numpy as np
import pytest

from speechtotext.speakers import nemotron


# --- probabilities -> turns -------------------------------------------------------------

PROBS = np.array(
    [
        # spk0 spk1 spk2
        [0.9, 0.1, 0.0],  # 0.00 s
        [0.9, 0.1, 0.0],  # 0.01 s
        [0.9, 0.6, 0.0],  # 0.02 s  both speak
        [0.2, 0.7, 0.0],  # 0.03 s
        [0.2, 0.5, 0.0],  # 0.04 s  exactly the threshold: not speech
        [0.8, 0.2, 0.0],  # 0.05 s  the last frame closes the turn at the end
    ],
    dtype=np.float32,
)


def test_turns_from_probs_is_one_turn_per_run_above_the_threshold():
    assert nemotron.turns_from_probs(PROBS) == [
        (0.0, 0.03, "speaker_0"),
        (0.02, 0.04, "speaker_1"),
        (0.05, 0.06, "speaker_0"),
    ]


def test_turns_from_probs_honors_the_threshold():
    assert nemotron.turns_from_probs(PROBS, threshold=0.65) == [
        (0.0, 0.03, "speaker_0"),
        (0.03, 0.04, "speaker_1"),
        (0.05, 0.06, "speaker_0"),
    ]


def test_turns_from_probs_with_no_speech_is_empty():
    assert nemotron.turns_from_probs(np.zeros((50, 8), dtype=np.float32)) == []


# --- what is missing, said before the transcription starts --------------------------------

def _environment(monkeypatch, *, transformers="5.18.0.dev0", modules=("torch", "librosa")):
    def version(name):
        if name != "transformers":
            raise AssertionError(f"only transformers is version-checked, not {name}")
        if transformers is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return transformers

    monkeypatch.setattr(nemotron, "_version", version)
    monkeypatch.setattr(nemotron, "_find_spec", lambda name: object() if name in modules else None)


def test_nothing_is_missing_with_transformers_518(monkeypatch):
    _environment(monkeypatch, transformers="5.18.0.dev0")
    assert nemotron.missing() is None


def test_a_later_major_is_enough(monkeypatch):
    _environment(monkeypatch, transformers="6.0.0")
    assert nemotron.missing() is None


@pytest.mark.parametrize("installed", ["5.17.0", "5.9.0", None])
def test_an_older_or_absent_transformers_names_the_pinned_git_install(monkeypatch, installed):
    # "5.9.0" is the case a string comparison gets wrong: "5.9" > "5.18" as text.
    _environment(monkeypatch, transformers=installed)
    message = nemotron.missing()
    assert message is not None
    assert f"git+https://github.com/huggingface/transformers@{nemotron.TRANSFORMERS_COMMIT}" in message


def test_an_empty_environment_is_told_both_steps_in_order(monkeypatch):
    # `.[nemotron]` cannot resolve until transformers 5.18 is installed from git: telling a
    # bare environment to install the extra first hands it a command that fails.
    _environment(monkeypatch, transformers=None, modules=())
    message = nemotron.missing()
    git = message.index(f"git+https://github.com/huggingface/transformers@{nemotron.TRANSFORMERS_COMMIT}")
    assert git < message.index('".[nemotron]"')


@pytest.mark.parametrize("absent", ["torch", "librosa"])
def test_a_missing_module_points_at_the_extra(monkeypatch, absent):
    _environment(monkeypatch, modules=tuple(m for m in ("torch", "librosa") if m != absent))
    message = nemotron.missing()
    assert message is not None and absent in message and "[nemotron]" in message


# --- loading: pinned, once, and nothing on stdout -----------------------------------------

def test_load_pins_the_measured_revision_once_and_keeps_stdout_clean(monkeypatch, capsys):
    """transformers main prints a docstring check on stdout when AutoProcessor is imported.
    stdout belongs to the caller (the MCP server speaks its protocol there), so anything
    printed while loading goes to stderr. And the checkpoint is the measured one, not
    whatever the hub serves today."""
    import sys
    import types

    calls = []

    class _Pretrained:
        @classmethod
        def from_pretrained(cls, model_id, revision=None):
            print("transformers talking on stdout")
            calls.append((cls.__name__, model_id, revision))
            return cls()

        def eval(self):   # torch's Module.eval(): inference mode. Not Python's eval().
            self.evaluated = True
            return self

    class AutoProcessor(_Pretrained):
        pass

    class AutoModelForAudioFrameClassification(_Pretrained):
        pass

    hf_logging = types.ModuleType("transformers.utils.logging")
    hf_logging.is_progress_bar_enabled = lambda: True
    hf_logging.disable_progress_bar = hf_logging.enable_progress_bar = lambda: None
    utils = types.ModuleType("transformers.utils")
    utils.logging = hf_logging
    transformers = types.ModuleType("transformers")
    transformers.utils = utils
    transformers.AutoProcessor = AutoProcessor
    transformers.AutoModelForAudioFrameClassification = AutoModelForAudioFrameClassification
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", hf_logging)
    monkeypatch.setattr(nemotron, "_MODEL", None)

    processor, model = nemotron._load()
    assert nemotron._load() == (processor, model)          # once per process
    assert calls == [
        ("AutoProcessor", nemotron.MODEL_ID, nemotron.MODEL_REVISION),
        ("AutoModelForAudioFrameClassification", nemotron.MODEL_ID, nemotron.MODEL_REVISION),
    ]
    assert model.evaluated is True
    printed = capsys.readouterr()
    assert "talking" not in printed.out and "talking" in printed.err


# --- guards that come before the model ----------------------------------------------------

def _no_model(monkeypatch):
    def load():
        raise AssertionError("the model must not load for this input")

    monkeypatch.setattr(nemotron, "_load", load)


def test_diarize_rejects_a_rate_the_model_was_not_trained_on(monkeypatch):
    _no_model(monkeypatch)
    with pytest.raises(ValueError, match="16000"):
        nemotron.diarize(np.zeros(8000, dtype=np.float32), 8000)


def test_audio_shorter_than_one_model_frame_has_no_turns(monkeypatch):
    # One encoder frame is 8 mel frames of 10 ms: 80 ms, 1280 samples.
    _no_model(monkeypatch)
    assert nemotron.diarize(np.zeros(1279, dtype=np.float32), 16000) == []


# --- piecewise features: the same numbers as one pass over the whole file -----------------

def _extractor_or_skip():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("librosa") is None:
        pytest.skip("needs the [nemotron] extra")
    try:
        from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
            NemotronAsrStreamingFeatureExtractor,
        )
    except ImportError:
        pytest.skip("needs transformers >= 5.18")
    # The checkpoint's processor_config.json, written out: no network here.
    return NemotronAsrStreamingFeatureExtractor(
        feature_size=128, hop_length=160, n_fft=512, preemphasis=0.97,
        sampling_rate=16000, win_length=400,
    )


@pytest.mark.parametrize("extra_samples", [0, 77])
def test_piecewise_features_match_a_single_pass(extra_samples):
    """The whole-file pass peaked at 4.0 GB for 64 minutes (measured 2026-09-24); the pieces
    exist to cap that. They are only allowed if they change nothing the model sees."""
    fe = _extractor_or_skip()
    import torch

    rng = np.random.default_rng(3)
    n = 20 * 16000 + extra_samples                   # not always a whole number of hops
    t = np.arange(n) / 16000
    x = (0.1 * rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    whole = fe(x, sampling_rate=16000, return_tensors="pt")
    features, mask = nemotron._features(fe, x, piece_frames=250)   # 2.5 s pieces: 8 of them

    assert features.shape == whole["input_features"].shape
    assert torch.equal(mask, whole["attention_mask"])
    assert torch.allclose(features, whole["input_features"], atol=1e-5)
