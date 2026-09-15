"""Transcription segment with an optional speaker."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    # Extent of the segment emitted by ASR, before diarization recompresses
    # the span to its words (C-13). At the end and with a default: all existing
    # constructions are positional and none break. None = "I am the original span."
    src_dur: float | None = None
    # Native signals from faster-whisper (Phase 2, G5): whisper.cpp does not emit them -> None
    # end to end. At the end, following the same pattern as src_dur.
    no_speech: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


def _finite(value: float | None) -> float | None:
    """NaN/inf come in as None. A non-finite value is not a degraded measurement, it is
    the absence of a measurement: letting it through writes NaN/Infinity tokens to JSON
    (json.dumps accepts them but RFC 8259 does not, so jq and JSON.parse blow up) and also
    disables is_suspect exactly where it is needed most, because `NaN > 0.6` is False. The
    other path through the repo already rejects it in asr/types.py::_validate_native_signals;
    here it is omitted, which is G5's answer and does not force anyone to handle an exception
    halfway through the pipeline."""
    return value if value is not None and math.isfinite(value) else None


def native_signals(seg) -> tuple[float | None, float | None, float | None]:
    """Read (no_speech, avg_logprob, compression_ratio) from a Segment, in its two
    dialects: raw faster-whisper (no_speech_prob) and ours (no_speech).
    Absent from both -> None (G5: a signal the engine does not emit is omitted, not filled in)."""
    no_speech = getattr(seg, "no_speech_prob", None)
    if no_speech is None:
        no_speech = getattr(seg, "no_speech", None)
    return (_finite(no_speech), _finite(getattr(seg, "avg_logprob", None)),
            _finite(getattr(seg, "compression_ratio", None)))
