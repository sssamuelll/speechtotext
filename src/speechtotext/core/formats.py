"""Serializadores de transcripción a txt/srt/vtt/json."""
from __future__ import annotations

import json
from pathlib import Path

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


def _speaker(seg):
    return getattr(seg, "speaker", None)


def write_txt(segments, path: Path) -> None:
    segs = list(segments)
    if any(_speaker(s) for s in segs):
        lines: list[str] = []
        cur: str | None = None
        buf: list[str] = []
        for s in segs:
            spk = _speaker(s) or "Hablante ?"
            if spk != cur:
                if buf:
                    lines.append(f"{cur}: {' '.join(buf)}")
                cur, buf = spk, [s.text.strip()]
            else:
                buf.append(s.text.strip())
        if buf:
            lines.append(f"{cur}: {' '.join(buf)}")
    else:
        lines = [s.text.strip() for s in segs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments, path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{format_timestamp(seg.start, srt=True)} --> {format_timestamp(seg.end, srt=True)}"
        )
        spk = _speaker(seg)
        text = seg.text.strip()
        lines.append(f"{spk}: {text}" if spk else text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments, path: Path) -> None:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{format_timestamp(seg.start, srt=False)} --> {format_timestamp(seg.end, srt=False)}"
        )
        spk = _speaker(seg)
        text = seg.text.strip()
        lines.append(f"{spk}: {text}" if spk else text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(segments, info, path: Path) -> None:
    seg_list = list(segments)
    speakers = sorted({_speaker(s) for s in seg_list} - {None})
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
                **({"speaker": _speaker(s)} if speakers else {}),
            }
            for i, s in enumerate(seg_list)
        ],
    }
    if speakers:
        payload["speakers"] = speakers
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
