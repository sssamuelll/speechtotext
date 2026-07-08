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


from speechtotext.core.chunked import seg_from_dict, seg_to_dict


def test_seg_roundtrip_con_palabras():
    seg = TimedSegment(600.0, 602.0, " hola", [TimedWord(600.0, 601.0, " hola")])
    back = seg_from_dict(seg_to_dict(seg))
    assert back == seg


def test_seg_roundtrip_sin_palabras():
    seg = TimedSegment(1.0, 2.0, "x", None)
    assert seg_from_dict(seg_to_dict(seg)) == seg


import json


def test_transcribe_chunk_usa_cache_si_existe(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    # sembrar checkpoint con un segmento global ya offseteado
    p = chunked.chunk_path(audio, _opts(), "large-v3", 600.0, 1200.0)
    p.write_text(json.dumps({"segments": [{"start": 601.0, "end": 602.0, "text": " cache"}]}))

    def boom(*a, **k):
        raise AssertionError("no debió transcribir ni llamar ffmpeg")
    monkeypatch.setattr(chunked.subprocess, "run", boom)
    model = SimpleNamespace(transcribe=boom)

    segs, cached = chunked.transcribe_chunk(audio, 600.0, 1200.0, _opts(), model, "large-v3")
    assert cached is True
    assert segs[0].text == " cache" and segs[0].start == 601.0


def test_transcribe_chunk_transcribe_y_guarda(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    monkeypatch.setattr(chunked.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr=b""))
    # modelo devuelve segmentos LOCALES (relativos al trozo)
    local = [SimpleNamespace(start=1.0, end=2.0, text=" hola", words=None)]
    model = SimpleNamespace(transcribe=lambda wav, **k: (iter(local), SimpleNamespace()))

    segs, cached = chunked.transcribe_chunk(audio, 600.0, 1200.0, _opts(), model, "large-v3")
    assert cached is False
    assert segs[0].start == 601.0 and segs[0].end == 602.0  # +600 offset
    # persistió el checkpoint con timestamps globales
    p = chunked.chunk_path(audio, _opts(), "large-v3", 600.0, 1200.0)
    assert p.exists()
    assert json.loads(p.read_text())["segments"][0]["start"] == 601.0


def test_run_chunked_reensambla_en_orden_y_arma_info(monkeypatch):
    monkeypatch.setattr(chunked, "WhisperModel", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(chunked, "probe_duration", lambda audio: 1200.0)
    monkeypatch.setattr(chunked, "plan_chunks", lambda audio, dur, **k: [(0.0, 600.0), (600.0, 1200.0)])

    def fake_chunk(audio, start, end, opts, model, model_name):
        return [TimedSegment(start + 1.0, start + 2.0, f" t{int(start)}")], False
    monkeypatch.setattr(chunked, "transcribe_chunk", fake_chunk)

    lines = []
    segs, info = chunked.run_chunked(
        Path("x.mp3"), _opts(), jobs=2, model_name="large-v3",
        device="cpu", compute_type="int8", log=lines.append,
    )
    assert [s.text for s in segs] == [" t0", " t600"]  # orden de trozo
    assert (segs[0].start, segs[1].start) == (1.0, 601.0)  # timestamps globales
    assert info.duration == 1200.0
    assert len(lines) == 2  # una línea por trozo


def test_run_chunked_marca_cache(monkeypatch):
    monkeypatch.setattr(chunked, "WhisperModel", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(chunked, "probe_duration", lambda audio: 600.0)
    monkeypatch.setattr(chunked, "plan_chunks", lambda audio, dur, **k: [(0.0, 600.0)])
    monkeypatch.setattr(chunked, "transcribe_chunk",
                        lambda *a, **k: ([TimedSegment(1.0, 2.0, " x")], True))
    lines = []
    chunked.run_chunked(Path("x.mp3"), _opts(), 1, "m", "cpu", "int8", log=lines.append)
    assert "cache" in lines[0]


def test_probe_duration_desde_pyav(monkeypatch):
    container = SimpleNamespace(duration=1245_000000)  # microsegundos

    class FakeAv:
        @staticmethod
        def open(path):
            return container
    monkeypatch.setattr(chunked, "av", FakeAv)
    assert chunked.probe_duration(Path("x.mp3")) == 1245.0


def test_should_chunk_auto_por_umbral():
    assert should_chunk(CHUNK_THRESHOLD + 1, None) is True
    assert should_chunk(CHUNK_THRESHOLD - 1, None) is False


def test_should_chunk_flag_explicito_manda():
    assert should_chunk(10.0, True) is True       # forzar en audio corto
    assert should_chunk(99999.0, False) is False  # forzar off en audio largo
