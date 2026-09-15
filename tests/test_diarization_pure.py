from types import SimpleNamespace

from speechtotext.speakers.diarization import (
    assign_segments,
    humanize_speaker,
    apply_names,
)


def _seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def _word(start, end, word):
    return SimpleNamespace(start=start, end=end, word=word)


def _seg_words(start, end, text, words):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def test_assign_segment_max_overlap_wins():
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")]
    segs = [_seg(0.8, 2.5, "a caballo")]  # 0.2 with 00, 1.5 with 01 -> 01 wins
    out = assign_segments(segs, turns)
    assert out[0].speaker == "SPEAKER_01"
    assert out[0].text == "a caballo"


def test_assign_segment_no_overlap_is_none():
    turns = [(0.0, 1.0, "SPEAKER_00")]
    out = assign_segments([_seg(5.0, 6.0, "solo")], turns)
    assert out[0].speaker is None


def test_the_majority_speaker_wins_even_if_pyannote_fragments_it():
    # pyannote splits one speaker into several turns. A segment spanning [S0][S1][S0]
    # must go to the speaker with the MOST TOTAL overlap (S0=4s), not the largest turn (S1=3s).
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01"), (5.0, 7.0, "SPEAKER_00")]
    out = assign_segments([_seg(0.0, 7.0, "todo el tramo")], turns)
    assert out[0].speaker == "SPEAKER_00"


def test_a_segment_that_crosses_a_boundary_is_split_by_speaker():
    # The bug: a single Whisper segment spans the speaker change. "al profesor" is said by
    # SPEAKER_00 and "Dave Bennett" by SPEAKER_01. With words, it must be SPLIT rather than
    # labeling the entire segment with one speaker (dragging the tail into the next speaker).
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")]
    words = [
        _word(4.0, 4.4, " al"),
        _word(4.4, 5.0, " profesor"),
        _word(5.0, 5.5, " Dave"),
        _word(5.5, 6.0, " Bennett"),
    ]
    out = assign_segments([_seg_words(4.0, 6.0, " al profesor Dave Bennett", words)], turns)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text.strip() == "al profesor"
    assert out[1].text.strip() == "Dave Bennett"
    # The split follows the words, not the entire segment.
    assert (out[0].start, out[0].end) == (4.0, 5.0)
    assert (out[1].start, out[1].end) == (5.0, 6.0)


def test_a_single_speaker_segment_is_not_fragmented():
    turns = [(0.0, 10.0, "SPEAKER_00")]
    words = [_word(1.0, 1.5, " hola"), _word(1.5, 2.0, " mundo")]
    out = assign_segments([_seg_words(1.0, 2.0, " hola mundo", words)], turns)
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"
    assert out[0].text.strip() == "hola mundo"


def test_a_word_in_a_gap_between_turns_inherits_the_neighboring_speaker():
    # Regression: pyannote does not cover the entire timeline. A word that falls in the gap
    # between two turns by the SAME speaker must NOT become None mid-sentence (Speaker ?).
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_00")]  # gap 2.0-2.5
    words = [_word(1.0, 1.5, " hola"), _word(2.1, 2.4, " mundo"), _word(2.6, 3.0, " cruel")]
    out = assign_segments([_seg_words(1.0, 3.0, " hola mundo cruel", words)], turns)
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"
    assert out[0].text.strip() == "hola mundo cruel"


def test_a_zero_duration_word_does_not_fragment_the_segment():
    # A zero-duration word (start==end) -> zero overlap -> None; it must inherit, not fragment.
    turns = [(0.0, 5.0, "SPEAKER_00")]
    words = [_word(1.0, 1.5, " a"), _word(1.5, 1.5, " b"), _word(1.5, 2.0, " c")]
    out = assign_segments([_seg_words(1.0, 2.0, " a b c", words)], turns)
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"


def test_a_gap_in_an_actual_transition_is_assigned_to_the_previous_speaker():
    # In an actual transition, the word in the gap inherits the previous speaker; the split occurs.
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.5, 10.0, "SPEAKER_01")]  # gap 5.0-5.5
    words = [_word(4.0, 4.9, " cierro"), _word(5.1, 5.4, " y"), _word(5.6, 6.2, " abro")]
    out = assign_segments([_seg_words(4.0, 6.2, " cierro y abro", words)], turns)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text.strip() == "cierro y"
    assert out[1].text.strip() == "abro"


# --- 5.2.3 · src_dur: the ASR segment duration survives recompression ------------------


def test_assign_segments_fills_src_dur_on_the_word_route():
    # The plan's canonical case: a 30 s ASR segment with a single 1 s word.
    # The run is recompressed to the word, but src_dur preserves the parent's 30 s.
    turns = [(0.0, 30.0, "SPEAKER_00")]
    out = assign_segments([_seg_words(0.0, 30.0, " Gracias.", [_word(0.4, 1.4, " Gracias.")])], turns)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0.4, 1.4)  # span recompressed to the word
    assert out[0].src_dur == 30.0


def test_assign_segments_gives_every_run_from_a_segment_the_same_src_dur():
    # The N runs from one segment inherit the parent's src_dur: they come from the same
    # decoding window.
    turns = [(0.0, 15.0, "SPEAKER_00"), (15.0, 30.0, "SPEAKER_01")]
    words = [_word(1.0, 2.0, " hola"), _word(16.0, 17.0, " chao")]
    out = assign_segments([_seg_words(0.0, 30.0, " hola chao", words)], turns)
    assert len(out) == 2
    assert [s.src_dur for s in out] == [30.0, 30.0]


def test_assign_segments_fills_src_dur_on_the_coarse_route():
    # Without words (whispercpp): one speaker per segment, and src_dur is the entire span.
    out = assign_segments([_seg(0.0, 30.0, "Gracias.")], [(0.0, 30.0, "SPEAKER_00")])
    assert out[0].src_dur == 30.0


def test_apply_names_propagates_src_dur():
    from speechtotext.core.segments import LabeledSegment
    labeled = [LabeledSegment(0.4, 1.4, " Gracias.", "SPEAKER_00", src_dur=30.0)]
    out = apply_names(labeled, {"SPEAKER_00": "Alice"})
    assert out[0].speaker == "Alice"
    assert out[0].src_dur == 30.0


def test_humanize_speaker():
    assert humanize_speaker("SPEAKER_00") == "Speaker 1"
    assert humanize_speaker("SPEAKER_01") == "Speaker 2"
    assert humanize_speaker("raro") == "raro"


def test_apply_names_maps_and_humanizes():
    from speechtotext.core.segments import LabeledSegment
    labeled = [
        LabeledSegment(0, 1, "hola", "SPEAKER_00"),
        LabeledSegment(1, 2, "chao", "SPEAKER_01"),
        LabeledSegment(2, 3, "...", None),
    ]
    out = apply_names(labeled, {"SPEAKER_00": "Alice"})
    assert [s.speaker for s in out] == ["Alice", "Speaker 2", None]
