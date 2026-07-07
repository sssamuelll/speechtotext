"""Diarización batch: asignación por solape (pura) + pyannote (perezoso)."""
from __future__ import annotations

from speechtotext.core.segments import LabeledSegment


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_segments(segments, turns: list[tuple[float, float, str]]) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for s in segments:
        best: str | None = None
        best_ov = 0.0
        for t0, t1, spk in turns:
            ov = _overlap(s.start, s.end, t0, t1)
            if ov > best_ov:
                best, best_ov = spk, ov
        out.append(LabeledSegment(start=s.start, end=s.end, text=s.text, speaker=best))
    return out


def humanize_speaker(speaker_id: str) -> str:
    try:
        n = int(speaker_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return speaker_id
    return f"Hablante {n + 1}"


def apply_names(
    labeled: list[LabeledSegment], name_map: dict[str, str]
) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for s in labeled:
        if s.speaker is None:
            spk: str | None = None
        else:
            spk = name_map.get(s.speaker) or humanize_speaker(s.speaker)
        out.append(LabeledSegment(s.start, s.end, s.text, spk))
    return out
