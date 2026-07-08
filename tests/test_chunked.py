from types import SimpleNamespace

from speechtotext.core.chunked import TimedSegment, TimedWord, parse_silences, pick_cuts, shift_segments


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


def test_pick_cuts_corta_en_silencio_cercano():
    # frontera en 600; hay silencio en 601.0-602.4 (mid 601.7) dentro de ±60 -> corta en 601.7
    chunks = pick_cuts([(601.0, 602.4)], duration=1200.0, target_len=600.0, search=60.0)
    assert chunks == [(0.0, 601.7), (601.7, 1200.0)]


def test_pick_cuts_fijo_si_no_hay_silencio_cerca():
    # silencio lejos de la frontera 600 -> corte fijo en 600
    chunks = pick_cuts([(100.0, 101.0)], duration=1200.0, target_len=600.0, search=60.0)
    assert chunks == [(0.0, 600.0), (600.0, 1200.0)]


def test_pick_cuts_audio_corto_un_solo_trozo():
    assert pick_cuts([], duration=300.0, target_len=600.0) == [(0.0, 300.0)]


def test_pick_cuts_cubre_toda_la_duracion_contiguo():
    chunks = pick_cuts([], duration=1500.0, target_len=600.0)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 1500.0
    for a, b in zip(chunks, chunks[1:]):
        assert a[1] == b[0]  # contiguo, sin huecos ni solapes


from pathlib import Path

import speechtotext.core.chunked as chunked


def test_plan_chunks_usa_silencedetect(monkeypatch):
    fake = SimpleNamespace(stderr=b"silence_start: 601.0\nsilence_end: 602.4\n", returncode=0)
    monkeypatch.setattr(chunked.subprocess, "run", lambda *a, **k: fake)
    chunks = chunked.plan_chunks(Path("x.mp3"), duration=1200.0, target_len=600.0)
    assert chunks == [(0.0, 601.7), (601.7, 1200.0)]


def test_plan_chunks_fallback_si_ffmpeg_falla(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no ffmpeg")
    monkeypatch.setattr(chunked.subprocess, "run", boom)
    chunks = chunked.plan_chunks(Path("x.mp3"), duration=1200.0, target_len=600.0)
    assert chunks == [(0.0, 600.0), (600.0, 1200.0)]  # cortes fijos


def _opts(**over):
    base = dict(language="es", beam_size=5, vad_filter=True, hotwords=None, word_timestamps=False)
    base.update(over)
    return base


def test_chunk_path_determinista(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTEXT_UNUSED", "x")
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    p1 = chunked.chunk_path(audio, _opts(), "large-v3", 0.0, 600.0)
    p2 = chunked.chunk_path(audio, _opts(), "large-v3", 0.0, 600.0)
    assert p1 == p2
    assert p1.parent == tmp_path / "chunks"


def test_chunk_path_cambia_con_cada_parametro(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    base = chunked.chunk_path(audio, _opts(), "large-v3", 0.0, 600.0)
    assert chunked.chunk_path(audio, _opts(), "medium", 0.0, 600.0) != base
    assert chunked.chunk_path(audio, _opts(hotwords="Boconó"), "large-v3", 0.0, 600.0) != base
    assert chunked.chunk_path(audio, _opts(word_timestamps=True), "large-v3", 0.0, 600.0) != base
    assert chunked.chunk_path(audio, _opts(), "large-v3", 600.0, 1200.0) != base
