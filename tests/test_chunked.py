from types import SimpleNamespace

from speechtotext.core.chunked import (
    CHUNK_THRESHOLD,
    TimedSegment,
    TimedWord,
    parse_silences,
    pick_cuts,
    should_chunk,
    shift_segments,
)


def _w(start, end, word):
    return SimpleNamespace(start=start, end=end, word=word)


def _s(start, end, text, words=None):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def test_shift_offsets_the_segment_and_its_words():
    segs = [_s(0.0, 2.0, " hola", [_w(0.0, 1.0, " hola")])]
    out = shift_segments(segs, 600.0)
    assert isinstance(out[0], TimedSegment)
    assert (out[0].start, out[0].end) == (600.0, 602.0)
    assert out[0].text == " hola"
    assert isinstance(out[0].words[0], TimedWord)
    assert (out[0].words[0].start, out[0].words[0].end) == (600.0, 601.0)
    assert out[0].words[0].word == " hola"


def test_shift_without_words_leaves_words_as_none():
    out = shift_segments([_s(1.0, 2.0, "x", None)], 10.0)
    assert out[0].words is None
    assert (out[0].start, out[0].end) == (11.0, 12.0)


def test_parse_silences_extracts_pairs():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 12.5\n"
        "[silencedetect @ 0x1] silence_end: 13.2 | silence_duration: 0.7\n"
        "[silencedetect @ 0x1] silence_start: 601.0\n"
        "[silencedetect @ 0x1] silence_end: 602.4 | silence_duration: 1.4\n"
    )
    assert parse_silences(stderr) == [(12.5, 13.2), (601.0, 602.4)]


def test_parse_silences_discards_a_start_without_an_end():
    stderr = "silence_start: 5.0\nsilence_end: 6.0\nsilence_start: 900.0\n"
    assert parse_silences(stderr) == [(5.0, 6.0)]


def test_pick_cuts_cuts_at_a_nearby_silence():
    # Boundary at 600; silence at 601.0-602.4 (midpoint 601.7) is within ±60 -> cut at 601.7.
    chunks = pick_cuts([(601.0, 602.4)], duration=1200.0, target_len=600.0, search=60.0)
    assert chunks == [(0.0, 601.7), (601.7, 1200.0)]


def test_pick_cuts_uses_a_fixed_cut_when_there_is_no_nearby_silence():
    # Silence far from the boundary at 600 -> fixed cut at 600.
    chunks = pick_cuts([(100.0, 101.0)], duration=1200.0, target_len=600.0, search=60.0)
    assert chunks == [(0.0, 600.0), (600.0, 1200.0)]


def test_pick_cuts_returns_one_chunk_for_short_audio():
    assert pick_cuts([], duration=300.0, target_len=600.0) == [(0.0, 300.0)]


def test_pick_cuts_covers_the_entire_duration_contiguously():
    chunks = pick_cuts([], duration=1500.0, target_len=600.0)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 1500.0
    for a, b in zip(chunks, chunks[1:]):
        assert a[1] == b[0]  # contiguous, with no gaps or overlaps


from pathlib import Path

import speechtotext.core.chunked as chunked


def test_plan_chunks_uses_silencedetect(monkeypatch):
    fake = SimpleNamespace(stderr=b"silence_start: 601.0\nsilence_end: 602.4\n", returncode=0)
    monkeypatch.setattr(chunked.subprocess, "run", lambda *a, **k: fake)
    chunks = chunked.plan_chunks(Path("x.mp3"), duration=1200.0, target_len=600.0)
    assert chunks == [(0.0, 601.7), (601.7, 1200.0)]


def test_plan_chunks_falls_back_to_fixed_cuts_if_ffmpeg_fails(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no ffmpeg")
    monkeypatch.setattr(chunked.subprocess, "run", boom)
    chunks = chunked.plan_chunks(Path("x.mp3"), duration=1200.0, target_len=600.0)
    assert chunks == [(0.0, 600.0), (600.0, 1200.0)]  # fixed cuts


from speechtotext.core.chunked import seg_from_dict, seg_to_dict


def test_segment_roundtrip_with_words():
    seg = TimedSegment(600.0, 602.0, " hola", [TimedWord(600.0, 601.0, " hola")])
    back = seg_from_dict(seg_to_dict(seg))
    assert back == seg


def test_segment_roundtrip_without_words():
    seg = TimedSegment(1.0, 2.0, "x", None)
    assert seg_from_dict(seg_to_dict(seg)) == seg


def test_should_chunk_automatically_uses_the_threshold():
    assert should_chunk(CHUNK_THRESHOLD + 1, None) is True
    assert should_chunk(CHUNK_THRESHOLD - 1, None) is False


def test_should_chunk_obeys_the_explicit_flag():
    assert should_chunk(10.0, True) is True       # force on for short audio
    assert should_chunk(99999.0, False) is False  # force off for long audio


def test_clip_to_end_discards_padding_and_clips_protruding_segments():
    segs = [
        TimedSegment(601.0, 602.0, " hola"),
        # Protrudes by 0.4 s: clip it, along with the word that crosses the boundary.
        TimedSegment(1190.0, 1200.4, " cierra la frase",
                     [TimedWord(1190.0, 1195.0, " cierra"), TimedWord(1195.0, 1200.4, " la frase")]),
        # 0.1 s of audio and 29.9 s of padding: hallucination, discard it.
        TimedSegment(1199.9, 1229.9, " Gracias por ver el video.",
                     [TimedWord(1199.9, 1229.9, " Gracias")]),
    ]
    out = chunked.clip_to_end(segs, 1200.0)
    assert [(s.start, s.end, s.text) for s in out] == [
        (601.0, 602.0, " hola"), (1190.0, 1200.0, " cierra la frase"),
    ]
    assert [(w.start, w.end) for w in out[1].words] == [(1190.0, 1195.0), (1195.0, 1200.0)]


def test_clip_to_end_leaves_words_as_none_when_all_words_are_in_the_padding():
    seg = TimedSegment(1195.0, 1201.0, " x", [TimedWord(1200.5, 1201.0, " x")])
    out = chunked.clip_to_end([seg], 1200.0)
    assert out[0].end == 1200.0 and out[0].words is None


def test_chunk_path_is_deterministic_and_sensitive_to_identity_and_range(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    a = chunked.chunk_path("id-1", 0.0, 600.0)
    assert a == chunked.chunk_path("id-1", 0.0, 600.0)
    assert a.parent == tmp_path / "chunks" and a.suffix == ".json"
    assert a != chunked.chunk_path("id-2", 0.0, 600.0)
    assert a != chunked.chunk_path("id-1", 600.0, 1200.0)
