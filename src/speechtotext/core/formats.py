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


def is_suspect(seg) -> bool:
    # ponytail: heurística sin calibrar, techo conocido; sube a calibrador si algún día hay corpus
    dur = seg.end - seg.start
    ns = getattr(seg, "no_speech", None)
    if ns is not None and ns > 0.6:          # se enciende sola cuando llegue la Fase 2
        return True
    return dur >= 10.0 and len(seg.text.strip()) / dur < 1.0


def _marked(seg) -> str:
    """Texto del segmento con la marca de sospecha. Marcar, no borrar: un falso positivo
    cuesta un '[?]' de más; un falso negativo cuesta leer una invención como si fuera tu
    conversación. El campo `text` del JSON queda limpio a propósito (allí va un booleano)."""
    text = seg.text.strip()
    return f"[?] {text}" if is_suspect(seg) else text


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
                cur, buf = spk, [_marked(s)]
            else:
                buf.append(_marked(s))
        if buf:
            lines.append(f"{cur}: {' '.join(buf)}")
    else:
        lines = [_marked(s) for s in segs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments, path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{format_timestamp(seg.start, srt=True)} --> {format_timestamp(seg.end, srt=True)}"
        )
        spk = _speaker(seg)
        text = _marked(seg)
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
        text = _marked(seg)
        lines.append(f"{spk}: {text}" if spk else text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _gaps(seg_list, duration: float) -> list[list[float]]:
    """Complemento de los segmentos sobre [0, duration]: el audio que no produjo texto.
    Los segmentos de Whisper no se solapan dentro de una pasada y los trozos son contiguos
    por construcción, así que basta un barrido con cursor."""
    # ponytail: umbral arbitrario, súbelo si el JSON se llena de huecos de respiración
    min_gap = 5.0
    out: list[list[float]] = []
    cursor = 0.0
    for s in seg_list:
        if s.start - cursor >= min_gap:
            out.append([round(cursor, 2), round(s.start, 2)])
        cursor = max(cursor, s.end)
    if duration - cursor >= min_gap:
        out.append([round(cursor, 2), round(duration, 2)])
    return out


def write_json(segments, info, path: Path) -> None:
    seg_list = list(segments)
    speakers = sorted({_speaker(s) for s in seg_list} - {None})
    prob = info.language_probability
    payload = {
        "language": info.language,
        # prob es None cuando el idioma se detectó de verdad en ruta troceada y la medición se
        # perdió; omitir la clave dice "no lo sé" sin fabricar un número.
        **({"language_probability": round(prob, 4)} if prob is not None else {}),
        # "duration" sigue siendo la del archivo; "speech_s" es lo que produjo texto.
        "duration": round(info.duration, 2),
        "speech_s": round(sum(s.end - s.start for s in seg_list), 2),
        "gaps": _gaps(seg_list, info.duration),
        "segments": [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                **({"speaker": _speaker(s)} if speakers else {}),
                **({"suspect": True} if is_suspect(s) else {}),
            }
            for i, s in enumerate(seg_list)
        ],
    }
    if speakers:
        payload["speakers"] = speakers
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
