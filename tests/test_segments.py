from speechtotext.core.segments import LabeledSegment


def test_labeled_segment_defaults_speaker_none():
    seg = LabeledSegment(start=0.0, end=1.5, text="hola")
    assert seg.speaker is None
    assert (seg.start, seg.end, seg.text) == (0.0, 1.5, "hola")


def test_labeled_segment_with_speaker():
    seg = LabeledSegment(0.0, 1.0, "hola", "Samuel")
    assert seg.speaker == "Samuel"
