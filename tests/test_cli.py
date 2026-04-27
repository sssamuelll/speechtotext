"""Smoke test del CLI completo con faster_whisper stubeado en conftest."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from main import app


def _fake_audio(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return path


def test_cli_produces_all_formats(tmp_path: Path) -> None:
    audio = _fake_audio(tmp_path)
    result = CliRunner().invoke(
        app, [str(audio), "--formats", "txt,srt,vtt,json", "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    for ext in ("txt", "srt", "vtt", "json"):
        assert (tmp_path / f"sample.{ext}").exists()


def test_cli_json_payload_shape(tmp_path: Path) -> None:
    audio = _fake_audio(tmp_path)
    result = CliRunner().invoke(app, [str(audio), "--formats", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert data["language"] == "es"
    assert len(data["segments"]) == 3
    assert data["segments"][0]["speaker"] is None  # sin --diarize


def test_cli_rejects_invalid_format(tmp_path: Path) -> None:
    audio = _fake_audio(tmp_path)
    result = CliRunner().invoke(app, [str(audio), "--formats", "doc"])
    assert result.exit_code != 0


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--diarize" in result.output
    assert "--language" in result.output
