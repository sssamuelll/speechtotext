"""Serializadores de transcripción a txt/srt/vtt/json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

VALID_FORMATS: frozenset[str] = frozenset({"txt", "srt", "vtt", "json"})


def format_timestamp(seconds: float, *, srt: bool) -> str:
    millis = max(0, round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1_000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def parse_formats(formats: str) -> set[str]:
    requested = {f.strip().lower() for f in formats.split(",") if f.strip()}
    invalid = requested - VALID_FORMATS
    if invalid:
        raise ValueError(
            f"Formatos no soportados: {sorted(invalid)}. Usa: {sorted(VALID_FORMATS)}."
        )
    return requested


def write_txt(segments: Iterable, path: Path) -> None:
    path.write_text("\n".join(s.text.strip() for s in segments) + "\n", encoding="utf-8")


def write_srt(segments: Iterable, path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{format_timestamp(seg.start, srt=True)} --> {format_timestamp(seg.end, srt=True)}"
        )
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: Iterable, path: Path) -> None:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{format_timestamp(seg.start, srt=False)} --> {format_timestamp(seg.end, srt=False)}"
        )
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(segments: Iterable, info, path: Path) -> None:
    seg_list = list(segments)
    payload = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 2),
        "segments": [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
            }
            for i, s in enumerate(seg_list)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
