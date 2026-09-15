import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    assert "regions" in result.stdout.lower() or "region" in result.stdout.lower()


def test_find_no_match(tmp_path, monkeypatch):
    audio = _seed(tmp_path, monkeypatch)
    result = runner.invoke(app, ["find", str(audio), "baloncesto"])
    assert result.exit_code == 0
    assert "No match for" in result.stdout


def test_find_extract_clips_and_transcribes_with_the_new_defaults(tmp_path, monkeypatch):
    from speechtotext.cli import app as app_mod

    audio = _seed(tmp_path, monkeypatch)
    runs = []

    def fake_run(cmd, check=True, capture_output=True):
        Path(cmd[-1]).write_bytes(b"RIFF")   # ffmpeg "clipped"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(app_mod, "transcribe_file", lambda *a, **k: runs.append((a, k)))
    result = runner.invoke(app, ["find", str(audio), "sismica", "--extract"])
    assert result.exit_code == 0, result.stdout
    (args, kw), = runs
    clip, base_dir, language, model, formats, device, compute_type, vad, beam = args[:9]
    assert clip.name.startswith("programa_") and clip.suffix == ".wav" and base_dir == tmp_path
    assert (language, model, formats) == ("auto", "large-v3", "txt,srt")
    assert (device, compute_type, vad, beam) == ("auto", "auto", False, 5)
    assert kw == {"hotwords": None}
