"""Batch diarization with NVIDIA Nemotron-3-Diarization (lazy: torch, transformers, librosa).

An end-to-end Sortformer: one pass over the recording gives every 10 ms frame a probability
per speaker channel, and the channels are numbered by first arrival. Measured on 2026-09-24
against pyannote community-1 on the same CPU: 64 minutes of a two-person call in 55 s, model
load included, instead of 25 min, with the same share of misattributed words when neither is
told the speaker count
(1.55% against 1.75%). It gives no voice embeddings, so it cannot put names on speakers, and it
takes no speaker count: the model decides, up to eight.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import re

import numpy as np

MODEL_ID = "nvidia/Nemotron-3-Diarization"
# The checkpoint that was measured. A new revision is a new measurement, not a silent upgrade.
MODEL_REVISION = "a435e9867d79e789e90053f9b6d6834053af564a"
# transformers ships the model from 5.18. Until that release, main at this commit is the one
# the measurement ran on.
TRANSFORMERS_COMMIT = "f324707307757d9c0b8dac1c4462eceff911fa2f"
_MIN_TRANSFORMERS = (5, 18)

SAMPLE_RATE = 16000
# The processor's own threshold. Measured flat between 0.3 and 0.7 on the call above.
THRESHOLD = 0.5
_FRAME_S = 0.01           # one probability per 10 ms mel frame
_ENCODER_SAMPLES = 1280   # one encoder frame is 8 mel frames: shorter audio gives the model nothing
# ponytail: 60 s of features at a time. One pass over 64 minutes peaked at 4.0 GB inside the
# STFT (measured), for features that weigh 188 MB. The size only bounds that transient.
_PIECE_FRAMES = 6000

# Seams for the dependency check: tests swap them, nothing else should.
_version = importlib.metadata.version
_find_spec = importlib.util.find_spec

_MODEL = None


def missing() -> str | None:
    """What is missing to run this diarizer, with the command that fixes it; None if nothing.

    Cheap on purpose (it imports nothing heavy): transcribe() asks before an hour of ASR,
    not after it. transformers is checked first because `.[nemotron]` cannot resolve without
    it until 5.18 is on PyPI: sending a bare environment to the extra first fails."""
    try:
        installed = _version("transformers")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    # Compare numbers, not text: as strings, "5.9" sorts after "5.18".
    match = re.match(r"(\d+)\.(\d+)", installed or "")
    if match is None or (int(match.group(1)), int(match.group(2))) < _MIN_TRANSFORMERS:
        found = f"transformers {installed} is installed" if installed else "transformers is not installed"
        return (
            f"{found}; Nemotron-3-Diarization needs 5.18 or later, from git until it is released: "
            f'pip install "transformers @ git+https://github.com/huggingface/transformers@{TRANSFORMERS_COMMIT}", '
            'then pip install -e ".[nemotron]"'
        )
    for module in ("torch", "librosa"):
        if _find_spec(module) is None:
            return f'{module} is not installed: pip install -e ".[nemotron]"'
    return None


def turns_from_probs(probs, threshold: float = THRESHOLD) -> list[tuple[float, float, str]]:
    """Frame probabilities (frames x channels) -> (start, end, speaker_id) turns: one per run
    of frames strictly above the threshold, sorted by start. Overlapping speech gives
    overlapping turns. The same rule as the processor's `extract_speaker_dict`."""
    turns = []
    for channel in range(probs.shape[1]):
        active = (probs[:, channel] > threshold).astype(np.int8)
        edges = np.diff(np.concatenate([[0], active, [0]]))
        turns.extend(
            (round(float(a) * _FRAME_S, 2), round(float(b) * _FRAME_S, 2), f"speaker_{channel}")
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        )
    turns.sort(key=lambda t: (t[0], t[2]))
    return turns


def _features(fe, samples, piece_frames: int = _PIECE_FRAMES):
    """The extractor's output for the whole recording, computed in pieces:
    `(input_features [1, N + 1, mels], attention_mask [1, N + 1])`, where N = len // hop are
    the valid frames and the centered pass's trailing frame is masked and zeroed.

    Every piece carries a margin of real audio on both sides wide enough for half a window,
    so each frame that is kept was computed from the same samples a single pass would use,
    or from the same zero padding that pass adds at the two ends of the file."""
    import torch

    hop = fe.hop_length
    margin = -(-(fe.n_fft // 2) // hop)          # ceil: half a window, in frames
    n = len(samples) // hop
    kept, mask_dtype = [], torch.bool
    for f0 in range(0, n, piece_frames):
        f1 = min(f0 + piece_frames, n)
        a = max(0, (f0 - margin) * hop)
        b = min(len(samples), (f1 + margin) * hop)
        out = fe(samples[a:b], sampling_rate=fe.sampling_rate, return_tensors="pt")
        first = f0 - a // hop
        kept.append(out["input_features"][0, first:first + (f1 - f0)])
        mask_dtype = out["attention_mask"].dtype
    features = torch.cat(kept + [torch.zeros(1, fe.feature_size)])[None]
    mask = torch.zeros(1, n + 1, dtype=mask_dtype)
    mask[0, :n] = 1
    return features, mask


def _load():
    """(processor, model), once per process, pinned to the measured revision.

    Anything transformers prints while loading goes to stderr: its main branch prints a
    docstring check on stdout when AutoProcessor is imported, and stdout belongs to the caller
    (the MCP server speaks its protocol there). The weight-loading bar is off for the same
    reason the core never prints."""
    global _MODEL
    if _MODEL is None:
        import contextlib
        import sys

        from transformers.utils import logging as hf_logging

        bar = hf_logging.is_progress_bar_enabled()
        hf_logging.disable_progress_bar()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from transformers import AutoModelForAudioFrameClassification, AutoProcessor

                processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
                model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        finally:
            if bar:
                hf_logging.enable_progress_bar()
        _MODEL = (processor, model.eval())
    return _MODEL


def diarize(samples, sample_rate: int) -> list[tuple[float, float, str]]:
    """Diarize float32 mono samples at 16 kHz. Returns turns, `(start, end, speaker_id)`, with
    `speaker_0`, `speaker_1`, ... numbered in order of first arrival. One offline pass over the
    whole recording (the model chunks it internally and keeps a speaker cache across chunks)."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Nemotron-3-Diarization takes {SAMPLE_RATE} Hz audio, got {sample_rate}")
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(samples) < _ENCODER_SAMPLES:
        return []

    import torch

    processor, model = _load()
    features, mask = _features(processor.feature_extractor, samples)
    with torch.inference_mode():
        logits = model(input_features=features, attention_mask=mask).logits
    probs = logits[0, : int(mask.sum())].sigmoid().float().numpy()
    return turns_from_probs(probs)
