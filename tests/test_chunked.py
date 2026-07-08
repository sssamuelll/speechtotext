from types import SimpleNamespace

from speechtotext.core.chunked import TimedSegment, TimedWord, shift_segments


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
