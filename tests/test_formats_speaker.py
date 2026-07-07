import json
from types import SimpleNamespace

from speechtotext.core.segments import LabeledSegment
from speechtotext.core.formats import write_txt, write_srt, write_vtt, write_json


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
