"""Tests para helpers puros: timestamps, parsing y escritores TXT/SRT/VTT/JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from main import (
    AnnotatedSegment,
    format_timestamp,
    parse_formats,
    resolve_output_base,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)


class _FakeInfo:
    language = "es"
    language_probability = 0.97
    duration = 12.34


@pytest.fixture
def segments() -> list[AnnotatedSegment]:
    return [
        AnnotatedSegment(0.0, 3.5, "Hola mundo."),
        AnnotatedSegment(3.5, 7.2, "¿Cómo estás?"),
        AnnotatedSegment(7.2, 12.3, "Probando 1 2 3."),
    ]


@pytest.fixture
def segments_with_speakers() -> list[AnnotatedSegment]:
    return [
        AnnotatedSegment(0.0, 3.5, "Hola mundo.", speaker="SPEAKER_00"),
        AnnotatedSegment(3.5, 7.2, "¿Cómo estás?", speaker="SPEAKER_01"),
    ]


# --- format_timestamp -------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "srt", "expected"),
    [
        (0.0, True, "00:00:00,000"),
        (0.0, False, "00:00:00.000"),
        (3.5, True, "00:00:03,500"),
        (3725.123, True, "01:02:05,123"),
        (3725.123, False, "01:02:05.123"),
        (-1.0, True, "00:00:00,000"),  # negativos saturan a cero
    ],
)
def test_format_timestamp(seconds: float, srt: bool, expected: str) -> None:
    assert format_timestamp(seconds, srt=srt) == expected


# --- parse_formats ----------------------------------------------------------


def test_parse_formats_valid() -> None:
    assert parse_formats("txt,srt,json") == {"txt", "srt", "json"}
    assert parse_formats(" TXT , Srt ") == {"txt", "srt"}


def test_parse_formats_invalid_raises() -> None:
    with pytest.raises(typer.BadParameter):
        parse_formats("txt,docx")


# --- resolve_output_base ----------------------------------------------------


def test_resolve_output_base_default(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    assert resolve_output_base(audio, None) == tmp_path / "sample"


def test_resolve_output_base_to_dir(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    out_dir = tmp_path / "outs"
    out_dir.mkdir()
    assert resolve_output_base(audio, out_dir) == out_dir / "sample"


def test_resolve_output_base_explicit_path(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    target = tmp_path / "subdir" / "result.txt"
    base = resolve_output_base(audio, target)
    assert base == tmp_path / "subdir" / "result"
    assert (tmp_path / "subdir").is_dir()


# --- writers ----------------------------------------------------------------


def test_write_txt_no_speakers(tmp_path: Path, segments: list[AnnotatedSegment]) -> None:
    path = tmp_path / "out.txt"
    write_txt(segments, path)
    assert path.read_text(encoding="utf-8") == "Hola mundo.\n¿Cómo estás?\nProbando 1 2 3.\n"


def test_write_txt_with_speakers(
    tmp_path: Path, segments_with_speakers: list[AnnotatedSegment]
) -> None:
    path = tmp_path / "out.txt"
    write_txt(segments_with_speakers, path)
    assert path.read_text(encoding="utf-8") == (
        "[SPEAKER_00] Hola mundo.\n[SPEAKER_01] ¿Cómo estás?\n"
    )


def test_write_srt(tmp_path: Path, segments: list[AnnotatedSegment]) -> None:
    path = tmp_path / "out.srt"
    write_srt(segments, path)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("1\n00:00:00,000 --> 00:00:03,500\nHola mundo.\n\n")
    assert "2\n00:00:03,500 --> 00:00:07,200" in content
    assert "3\n00:00:07,200 --> 00:00:12,300" in content


def test_write_vtt_header(tmp_path: Path, segments: list[AnnotatedSegment]) -> None:
    path = tmp_path / "out.vtt"
    write_vtt(segments, path)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:03.500" in content


def test_write_json(tmp_path: Path, segments_with_speakers: list[AnnotatedSegment]) -> None:
    path = tmp_path / "out.json"
    write_json(segments_with_speakers, _FakeInfo(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["language"] == "es"
    assert payload["duration"] == 12.34
    assert len(payload["segments"]) == 2
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["segments"][1]["text"] == "¿Cómo estás?"
