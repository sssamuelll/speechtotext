import json
from types import SimpleNamespace

from speechtotext.core.segments import LabeledSegment
from speechtotext.core.formats import (
    is_suspect,
    write_txt,
    write_srt,
    write_vtt,
    write_json,
)


# Evidencia real del brief: "Gracias." son 8 caracteres en 30 s = 0.27 char/s, muy por debajo
# de los 12-15 char/s del español hablado y de los 3-5 del susurro esparcido.
ALUCINADO = LabeledSegment(0, 30, "Gracias.")
REAL = LabeledSegment(30, 33, "Bueno, entonces quedamos así.")


def test_txt_groups_consecutive_speaker(tmp_path):
    segs = [
        LabeledSegment(0, 1, "hola", "Samuel"),
        LabeledSegment(1, 2, "qué tal", "Samuel"),
        LabeledSegment(2, 3, "bien", "Ale"),
    ]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == "Samuel: hola qué tal\nAle: bien\n"


def test_txt_without_speaker_unchanged(tmp_path):
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(1, 2, "chao")]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == "hola\nchao\n"


def test_srt_prefixes_speaker(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Samuel")]
    p = tmp_path / "o.srt"
    write_srt(segs, p)
    assert "Samuel: hola" in p.read_text(encoding="utf-8")


def test_json_has_speaker_and_speakers(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Ale")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["speakers"] == ["Ale"]
    assert data["segments"][0]["speaker"] == "Ale"


def test_json_without_speaker_omits_fields(tmp_path):
    segs = [LabeledSegment(0, 1, "hola")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "speakers" not in data
    assert "speaker" not in data["segments"][0]


def test_vtt_prefixes_speaker(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Samuel")]
    p = tmp_path / "o.vtt"
    write_vtt(segs, p)
    assert "Samuel: hola" in p.read_text(encoding="utf-8")


# --- 1.2: segundos con voz y huecos ---


def test_json_speech_s_y_gaps(tmp_path):
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(40, 41, "chao")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=60.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["speech_s"] == 2.0
    assert data["gaps"] == [[1, 40], [41, 60]]
    # "duration" sigue siendo la del archivo, no la de la voz
    assert data["duration"] == 60.0


def test_json_gaps_ignora_huecos_cortos(tmp_path):
    # 2 s de silencio entre segmentos: respiración, no pérdida de audio
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(3, 4, "chao")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=4.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    assert json.loads(p.read_text(encoding="utf-8"))["gaps"] == []


# --- 1.5: marca [?] ---


def test_is_suspect_por_densidad():
    assert is_suspect(ALUCINADO)
    assert not is_suspect(REAL)


def test_is_suspect_no_speech_apaga_la_densidad():
    # el campo no existe hasta la Fase 2: la rama queda inerte hoy y se enciende sola
    largo_pero_hablado = LabeledSegment(0, 30, "x" * 100)
    assert not is_suspect(largo_pero_hablado)
    largo_pero_hablado.no_speech = 0.9
    assert is_suspect(largo_pero_hablado)


def test_txt_marca_sospechoso_sin_diarizacion(tmp_path):
    p = tmp_path / "o.txt"
    write_txt([ALUCINADO, REAL], p)
    assert p.read_text(encoding="utf-8") == (
        "[?] Gracias.\nBueno, entonces quedamos así.\n"
    )


def test_txt_marca_sospechoso_con_diarizacion(tmp_path):
    segs = [
        LabeledSegment(0, 30, "Gracias.", "Samuel"),
        LabeledSegment(30, 33, "Bueno, entonces quedamos así.", "Ale"),
    ]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == (
        "Samuel: [?] Gracias.\nAle: Bueno, entonces quedamos así.\n"
    )


def test_srt_marca_sospechoso(tmp_path):
    p = tmp_path / "o.srt"
    write_srt([ALUCINADO, REAL], p)
    txt = p.read_text(encoding="utf-8")
    assert "[?] Gracias." in txt
    assert "[?] Bueno" not in txt


def test_vtt_marca_sospechoso(tmp_path):
    p = tmp_path / "o.vtt"
    write_vtt([ALUCINADO, REAL], p)
    txt = p.read_text(encoding="utf-8")
    assert "[?] Gracias." in txt
    assert "[?] Bueno" not in txt


def test_json_marca_sospechoso_y_deja_el_texto_limpio(tmp_path):
    info = SimpleNamespace(language="es", language_probability=1.0, duration=33.0)
    p = tmp_path / "o.json"
    write_json([ALUCINADO, REAL], info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["segments"][0]["suspect"] is True
    assert data["segments"][0]["text"] == "Gracias."
    assert "suspect" not in data["segments"][1]


# --- 1.6: idioma medido vs forzado ---


def test_json_omite_language_probability_cuando_es_none(tmp_path):
    segs = [LabeledSegment(0, 1, "hola")]
    info = SimpleNamespace(language="es", language_probability=None, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "language_probability" not in data
    assert data["language"] == "es"


def test_json_conserva_language_probability_forzada(tmp_path):
    segs = [LabeledSegment(0, 1, "hola")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    assert json.loads(p.read_text(encoding="utf-8"))["language_probability"] == 1.0
