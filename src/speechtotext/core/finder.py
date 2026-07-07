"""Buscador de segmento: índice de transcripción tiny + búsqueda por regiones."""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Region:
    start: float
    end: float
    hits: int
    matches: int
    snippet: str


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _terms(query: str) -> list[str]:
    return [normalize(t) for t in query.split() if t.strip()]


def _snip(text: str, width: int = 80) -> str:
    t = " ".join(text.split())
    return t if len(t) <= width else t[:width].rstrip() + "…"


def cluster_regions(hits: list[tuple[float, float, str, int]], gap: float) -> list[Region]:
    """Fusiona hits (start, end, text, match_count) cercanos (< gap) en una región."""
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h[0])
    regions: list[Region] = []
    start, end, snippet, n, m = ordered[0][0], ordered[0][1], ordered[0][2], 1, ordered[0][3]
    for h_start, h_end, _text, mc in ordered[1:]:
        if h_start - end <= gap:
            end = max(end, h_end)
            n += 1
            m += mc
        else:
            regions.append(Region(start, end, n, m, _snip(snippet)))
            start, end, snippet, n, m = h_start, h_end, _text, 1, mc
    regions.append(Region(start, end, n, m, _snip(snippet)))
    return regions


def search(segments: list[dict], query: str, gap: float = 60.0, top: int = 5) -> list[Region]:
    """Regiones (top-N por densidad) donde aparece la consulta."""
    terms = _terms(query)
    if not terms:
        return []
    hits: list[tuple[float, float, str, int]] = []
    for seg in segments:
        norm = normalize(seg["text"])
        mc = sum(1 for t in terms if t in norm)
        if mc:
            hits.append((seg["start"], seg["end"], seg["text"], mc))
    regions = cluster_regions(hits, gap)
    regions.sort(key=lambda r: (r.hits, r.matches), reverse=True)
    return regions[:top]


def clip_window(start: float, end: float, context: float) -> tuple[float, float]:
    """(inicio, duración) para recortar con `context` segundos de margen a cada lado."""
    begin = max(0.0, start - context)
    duration = (end - start) + 2 * context
    return begin, duration


def _home() -> Path:
    env = os.environ.get("SPEECHTOTEXT_HOME")
    return Path(env) if env else Path.home() / ".speechtotext"


def index_path(audio: Path, scan_model: str) -> Path:
    st = audio.stat()
    key = f"{audio.resolve()}|{st.st_size}|{int(st.st_mtime)}|{scan_model}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    d = _home() / "index"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"


def build_index(audio: Path, scan_model: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(scan_model, device="cpu", compute_type="int8")
    segments_iter, _info = model.transcribe(str(audio), vad_filter=True)
    return [
        {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
        for s in segments_iter
    ]


def load_or_build_index(
    audio: Path, scan_model: str, rebuild: bool = False
) -> tuple[list[dict], bool]:
    path = index_path(audio, scan_model)
    if not rebuild and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["segments"], True
        except (json.JSONDecodeError, KeyError):
            pass  # caché corrupta → reconstruir
    segments = build_index(audio, scan_model)
    st = audio.stat()
    path.write_text(
        json.dumps(
            {
                "audio": str(audio.resolve()),
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "scan_model": scan_model,
                "segments": segments,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return segments, False
