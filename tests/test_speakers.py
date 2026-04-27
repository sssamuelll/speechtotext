"""Tests para assign_speakers: asignación por solapamiento temporal."""

from __future__ import annotations

from main import AnnotatedSegment, assign_speakers


def test_assign_speakers_single_turn_covering_all() -> None:
    segments = [
        AnnotatedSegment(0.0, 1.0, "a"),
        AnnotatedSegment(1.0, 2.0, "b"),
    ]
    turns = [(0.0, 5.0, "S1")]
    out = assign_speakers(segments, turns)
    assert all(seg.speaker == "S1" for seg in out)


def test_assign_speakers_picks_dominant_overlap() -> None:
    segments = [AnnotatedSegment(0.0, 10.0, "mixto")]
    turns = [
        (0.0, 3.0, "S1"),  # 3s de solape
        (3.0, 10.0, "S2"),  # 7s de solape -> gana
    ]
    out = assign_speakers(segments, turns)
    assert out[0].speaker == "S2"


def test_assign_speakers_no_overlap_keeps_none() -> None:
    segments = [AnnotatedSegment(10.0, 12.0, "huérfano")]
    turns = [(0.0, 5.0, "S1")]
    out = assign_speakers(segments, turns)
    assert out[0].speaker is None


def test_assign_speakers_empty_turns() -> None:
    segments = [AnnotatedSegment(0.0, 1.0, "x")]
    assert assign_speakers(segments, []) == segments


def test_assign_speakers_preserves_text_and_times() -> None:
    seg = AnnotatedSegment(2.0, 4.0, "respeta")
    out = assign_speakers([seg], [(0.0, 10.0, "S")])
    assert out[0].start == 2.0
    assert out[0].end == 4.0
    assert out[0].text == "respeta"
    assert out[0].speaker == "S"
