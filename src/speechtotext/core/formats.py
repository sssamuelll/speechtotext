"""Transcription serializers for txt/srt/vtt/json."""
from __future__ import annotations

import json
from pathlib import Path

from speechtotext.core.segments import native_signals

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
            f"Unsupported formats: {sorted(invalid)}. Use: {sorted(VALID_FORMATS)}."
        )
    return requested


def _speaker(seg):
    return getattr(seg, "speaker", None)


def is_suspect(seg) -> bool:
    # ponytail: uncalibrated heuristic, known ceiling; promote to a calibrator if a corpus ever exists
    # src_dur is the extent of the segment emitted by ASR, before diarization
    # recompresses the span to its words (C-13): without it the 10 s gate stops firing
    # under --diarize. None (non-diarized path) -> its own span, as always.
    dur = getattr(seg, "src_dur", None) or (seg.end - seg.start)
    ns = getattr(seg, "no_speech", None)
    if ns is not None and ns > 0.6:          # turns on by itself when Phase 2 arrives
        return True
    return dur >= 10.0 and len(seg.text.strip()) / dur < 1.0


def _signal_keys(seg) -> dict:
    """Native signals that are present, ready for the payload. Absent = omitted key,
    following the same pattern as speaker and language_probability: silence says "I don't know"
    without fabrication."""
    names = ("no_speech", "avg_logprob", "compression_ratio")
    values = native_signals(seg)
    return {n: round(v, 4) for n, v in zip(names, values) if v is not None}


def _marked(seg) -> str:
    """Segment text with the suspect mark. Mark, do not delete: a false positive costs an
    extra '[?]'; a false negative costs reading an invention as if it were your conversation.
    The JSON `text` field is deliberately left clean (the boolean goes there)."""
    text = seg.text.strip()
    return f"[?] {text}" if is_suspect(seg) else text


def write_txt(segments, path: Path) -> None:
    segs = list(segments)
    if any(_speaker(s) for s in segs):
        lines: list[str] = []
        cur: str | None = None
        buf: list[str] = []
        for s in segs:
            spk = _speaker(s) or "Speaker ?"
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


def find_gaps(seg_list, duration: float) -> list[list[float]]:
    """Complement of the segments over [0, duration]: audio that produced no text.
    Whisper segments do not overlap within a pass and the chunks are contiguous by
    construction, so a cursor sweep is enough."""
    # ponytail: arbitrary threshold, raise it if the JSON fills with breathing gaps
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


def write_json(segments, info, path: Path, *, engine_info=None, speech_s=None, gaps=None) -> None:
    seg_list = list(segments)
    speakers = sorted({_speaker(s) for s in seg_list} - {None})
    prob = info.language_probability
    # speech_s/gaps arrive calculated by the CLI (once, over the segments emitted by ASR,
    # before diarization: C-13). None = calculate them here as always, following the same
    # conditional pattern as engine_info.
    if speech_s is None:
        speech_s = round(sum(s.end - s.start for s in seg_list), 2)
    if gaps is None:
        gaps = find_gaps(seg_list, info.duration)
    payload = {
        "language": info.language,
        # prob is None when the language was actually detected on the chunked code path and the
        # measurement was lost; omitting the key says "I don't know" without fabricating a number.
        **({"language_probability": round(prob, 4)} if prob is not None else {}),
        # "duration" remains the file's duration; "speech_s" is what produced text.
        "duration": round(info.duration, 2),
        "speech_s": speech_s,
        "gaps": gaps,
        "segments": [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                **({"speaker": _speaker(s)} if speakers else {}),
                # Rounded like start/end/language_probability: float32 values from the engine
                # arrive as -0.30000001192092896 and that precision noise is not data.
                **_signal_keys(s),
                **({"suspect": True} if is_suspect(s) else {}),
            }
            for i, s in enumerate(seg_list)
        ],
    }
    if speakers:
        payload["speakers"] = speakers
    # The CLI builds the dict ({name, version, model, quant, device, selection} and, under
    # --diarize, "diarization"); here it is only emitted as-is. None = absent key, following the
    # same conditional pattern as language_probability/speaker: omission says "not applicable"
    # without inventing.
    if engine_info is not None:
        payload["engine"] = engine_info
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
