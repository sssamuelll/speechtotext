from types import SimpleNamespace

from speechtotext.speakers.diarization import (
    assign_segments,
    humanize_speaker,
    apply_names,
)


def _seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def test_assign_segment_max_overlap_wins():
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")]
    segs = [_seg(0.8, 2.5, "a caballo")]  # 0.2 con 00, 1.5 con 01 -> gana 01
    out = assign_segments(segs, turns)
    assert out[0].speaker == "SPEAKER_01"
    assert out[0].text == "a caballo"


def test_assign_segment_no_overlap_is_none():
    turns = [(0.0, 1.0, "SPEAKER_00")]
    out = assign_segments([_seg(5.0, 6.0, "solo")], turns)
    assert out[0].speaker is None


def test_humanize_speaker():
    assert humanize_speaker("SPEAKER_00") == "Hablante 1"
    assert humanize_speaker("SPEAKER_01") == "Hablante 2"
    assert humanize_speaker("raro") == "raro"


def test_apply_names_maps_and_humanizes():
    from speechtotext.core.segments import LabeledSegment
    labeled = [
        LabeledSegment(0, 1, "hola", "SPEAKER_00"),
        LabeledSegment(1, 2, "chao", "SPEAKER_01"),
        LabeledSegment(2, 3, "...", None),
    ]
    out = apply_names(labeled, {"SPEAKER_00": "Samuel"})
    assert [s.speaker for s in out] == ["Samuel", "Hablante 2", None]
