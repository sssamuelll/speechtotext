from types import SimpleNamespace

from speechtotext.core.chunked import TimedSegment, TimedWord, parse_silences, shift_segments


def _w(start, end, word):
    return SimpleNamespace(start=start, end=end, word=word)


def _s(start, end, text, words=None):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def test_shift_desplaza_segmento_y_palabras():
    segs = [_s(0.0, 2.0, " hola", [_w(0.0, 1.0, " hola")])]
    out = shift_segments(segs, 600.0)
    assert isinstance(out[0], TimedSegment)
    assert (out[0].start, out[0].end) == (600.0, 602.0)
    assert out[0].text == " hola"
    assert isinstance(out[0].words[0], TimedWord)
    assert (out[0].words[0].start, out[0].words[0].end) == (600.0, 601.0)
    assert out[0].words[0].word == " hola"


def test_shift_sin_palabras_deja_words_none():
    out = shift_segments([_s(1.0, 2.0, "x", None)], 10.0)
    assert out[0].words is None
    assert (out[0].start, out[0].end) == (11.0, 12.0)


def test_parse_silences_extrae_pares():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 12.5\n"
        "[silencedetect @ 0x1] silence_end: 13.2 | silence_duration: 0.7\n"
        "[silencedetect @ 0x1] silence_start: 601.0\n"
        "[silencedetect @ 0x1] silence_end: 602.4 | silence_duration: 1.4\n"
    )
    assert parse_silences(stderr) == [(12.5, 13.2), (601.0, 602.4)]


def test_parse_silences_descarta_start_sin_end():
    stderr = "silence_start: 5.0\nsilence_end: 6.0\nsilence_start: 900.0\n"
    assert parse_silences(stderr) == [(5.0, 6.0)]
