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
    segs = [_seg(0.8, 2.5, "a caballo")]  # 0.2 con 00, 1.5 con 01 -> gana 01
    out = assign_segments(segs, turns)
    assert out[0].speaker == "SPEAKER_01"
    assert out[0].text == "a caballo"


def test_assign_segment_no_overlap_is_none():
    turns = [(0.0, 1.0, "SPEAKER_00")]
    out = assign_segments([_seg(5.0, 6.0, "solo")], turns)
    assert out[0].speaker is None


def test_hablante_mayoritario_gana_aunque_pyannote_lo_fragmente():
    # pyannote parte a un mismo hablante en varios turnos. Un segmento que abarca [S0][S1][S0]
    # debe ir al hablante con MÁS solape TOTAL (S0=4s), no al turno individual más grande (S1=3s).
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01"), (5.0, 7.0, "SPEAKER_00")]
    out = assign_segments([_seg(0.0, 7.0, "todo el tramo")], turns)
    assert out[0].speaker == "SPEAKER_00"


def test_segmento_que_cruza_frontera_se_parte_por_hablante():
    # El bug: un solo segmento Whisper abarca el cambio de hablante. "al profesor" lo dice
    # SPEAKER_00 y "Simón Ballesteros" SPEAKER_01. Con palabras debe PARTIRSE, no etiquetar
    # todo el segmento con un solo hablante (la cola arrastrada al siguiente).
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")]
    words = [
        _word(4.0, 4.4, " al"),
        _word(4.4, 5.0, " profesor"),
        _word(5.0, 5.5, " Simón"),
        _word(5.5, 6.0, " Ballesteros"),
    ]
    out = assign_segments([_seg_words(4.0, 6.0, " al profesor Simón Ballesteros", words)], turns)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text.strip() == "al profesor"
    assert out[1].text.strip() == "Simón Ballesteros"
    # el corte sigue a las palabras, no al segmento entero
    assert (out[0].start, out[0].end) == (4.0, 5.0)
    assert (out[1].start, out[1].end) == (5.0, 6.0)


def test_segmento_de_un_solo_hablante_no_se_fragmenta():
    turns = [(0.0, 10.0, "SPEAKER_00")]
    words = [_word(1.0, 1.5, " hola"), _word(1.5, 2.0, " mundo")]
    out = assign_segments([_seg_words(1.0, 2.0, " hola mundo", words)], turns)
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"
    assert out[0].text.strip() == "hola mundo"


def test_palabra_en_hueco_entre_turnos_hereda_al_vecino():
    # Regresión: pyannote no cubre toda la línea de tiempo. Una palabra que cae en el hueco
    # entre dos turnos del MISMO hablante NO debe salir como None mid-frase (Hablante ?).
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_00")]  # hueco 2.0-2.5
    words = [_word(1.0, 1.5, " hola"), _word(2.1, 2.4, " mundo"), _word(2.6, 3.0, " cruel")]
    out = assign_segments([_seg_words(1.0, 3.0, " hola mundo cruel", words)], turns)
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"
    assert out[0].text.strip() == "hola mundo cruel"


def test_palabra_de_duracion_cero_no_fragmenta():
    # Palabra de duración 0 (start==end) -> solape 0 -> None; debe heredar, no fragmentar.
    turns = [(0.0, 5.0, "SPEAKER_00")]
    words = [_word(1.0, 1.5, " a"), _word(1.5, 1.5, " b"), _word(1.5, 2.0, " c")]
    out = assign_segments([_seg_words(1.0, 2.0, " a b c", words)], turns)
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"


def test_hueco_en_transicion_real_se_asigna_al_previo():
    # En una transición real, la palabra del hueco hereda del hablante previo; el split ocurre.
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.5, 10.0, "SPEAKER_01")]  # hueco 5.0-5.5
    words = [_word(4.0, 4.9, " cierro"), _word(5.1, 5.4, " y"), _word(5.6, 6.2, " abro")]
    out = assign_segments([_seg_words(4.0, 6.2, " cierro y abro", words)], turns)
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0].text.strip() == "cierro y"
    assert out[1].text.strip() == "abro"


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
