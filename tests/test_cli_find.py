import json

from typer.testing import CliRunner

from speechtotext.cli.app import app
from speechtotext.core.finder import index_path

runner = CliRunner()


def _seed(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "programa.wav"
    audio.write_bytes(b"x" * 100)
    seed = {"segments": [
        {"start": 10.0, "end": 12.0, "text": "hablamos de vulnerabilidad sismica hoy"},
        {"start": 12.0, "end": 14.0, "text": "mas sismica todavia aqui"},
        {"start": 600.0, "end": 601.0, "text": "otra cosa distinta"},
    ]}
    index_path(audio, "tiny").write_text(json.dumps(seed), encoding="utf-8")
    return audio


def test_find_locate_prints_region(tmp_path, monkeypatch):
    audio = _seed(tmp_path, monkeypatch)
    result = runner.invoke(app, ["find", str(audio), "sismica"])
    assert result.exit_code == 0
    assert "00:10" in result.stdout
    assert "regiones" in result.stdout.lower() or "región" in result.stdout.lower()


def test_find_no_match(tmp_path, monkeypatch):
    audio = _seed(tmp_path, monkeypatch)
    result = runner.invoke(app, ["find", str(audio), "baloncesto"])
    assert result.exit_code == 0
    assert "No se encontró" in result.stdout
