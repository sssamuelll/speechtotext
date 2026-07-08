# Transcripción por trozos (resumible + paralela) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocear el pase de transcripción de audio largo en segmentos conscientes del silencio, con checkpoint por trozo (resume tras cortes) y paralelismo por pool de workers, componiendo con la diarización existente.

**Architecture:** Nuevo módulo `core/chunked.py` con lógica pura (representación de segmentos con offset, elección de cortes, clave de checkpoint, serialización) más `transcribe_chunk` (checkpoint/resume) y `run_chunked` (pool + reensamblaje). `transcribe_file` se ramifica por duración/flag y sustituye solo el bloque `segments = list(segments_iter)`. Todo lo demás (diarización, `normalize_hours`, writers) queda intacto.

**Tech Stack:** Python 3.14, faster-whisper (`WhisperModel`, `num_workers`), PyAV (duración), binario `ffmpeg` (silencedetect + recorte), `concurrent.futures.ThreadPoolExecutor`, pytest.

## Global Constraints

- Plataforma Windows; shell de tests: `.venv/Scripts/python.exe -m pytest`.
- **Ningún test invoca modelos reales ni ffmpeg real.** Se mockean `subprocess.run`, `WhisperModel`, `av.open`, y las funciones de trozo. Datos sintéticos con `SimpleNamespace`.
- Reusar el patrón de `finder`: `_home()` (`finder.py:77`), clave de contenido tipo `index_path` (`finder.py:82-88`), carga-o-computa con fallback si el JSON está corrupto (`finder.py:102-127`).
- `_transcribe_opts` ya fija `condition_on_previous_text=False` (`cli/app.py`); las ventanas son independientes, por eso trocear no degrada.
- Los `Segment` de faster-whisper son **inmutables**: no reasignar sus timestamps; construir `TimedSegment`/`TimedWord` nuevos con el offset aplicado a segmento **y palabras**.
- Timestamps guardados en checkpoint y reensamblados son **globales** (offset ya aplicado).
- Umbral auto-troceo `CHUNK_THRESHOLD = 1200.0` s (20 min); `target_len` por defecto `600.0` s; `search` por defecto `60.0` s. `--chunk-len` NO se expone en v1 (constante interna).
- Progreso: `run_chunked` emite **una línea por trozo terminado** vía un callback `log` (sin spinner de rich en la ruta troceada — es más legible en paralelo y no queda mudo off-TTY).

---

## File Structure

- **Create:** `src/speechtotext/core/chunked.py` — todo lo troceado.
- **Create:** `tests/test_chunked.py` — tests de lógica pura + `transcribe_chunk`/`run_chunked` con mocks.
- **Modify:** `src/speechtotext/cli/app.py` — `probe_duration`, `_should_chunk`, flags `--chunk/--no-chunk` y `--jobs`, y la rama en `transcribe_file`.

---

### Task 1: Representación de segmentos con offset

**Files:**
- Create: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Produces: `TimedWord(start: float, end: float, word: str)`; `TimedSegment(start: float, end: float, text: str, words: list[TimedWord] | None = None)`; `shift_segments(segments, offset: float) -> list[TimedSegment]` (acepta cualquier objeto con `.start/.end/.text` y `.words` opcional cuyos ítems tengan `.start/.end/.word`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py
from types import SimpleNamespace

from speechtotext.core.chunked import TimedSegment, TimedWord, shift_segments


def _w(start, end, word):
    return SimpleNamespace(start=start, end=end, word=word)


def _s(start, end, text, words=None):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def test_shift_desplaza_segmento_y_palabras():
    segs = [_s(0.0, 2.0, " hola", [_w(0.0, 1.0, " hola")])]
    out = shift_segments(segs, 600.0)
    assert isinstance(out[0], TimedSegment)
    assert (out[0].start, out[0].end) == (600.0, 602.0)
    assert out[0].text == " hola"
    assert isinstance(out[0].words[0], TimedWord)
    assert (out[0].words[0].start, out[0].words[0].end) == (600.0, 601.0)
    assert out[0].words[0].word == " hola"


def test_shift_sin_palabras_deja_words_none():
    out = shift_segments([_s(1.0, 2.0, "x", None)], 10.0)
    assert out[0].words is None
    assert (out[0].start, out[0].end) == (11.0, 12.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'speechtotext.core.chunked'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py
"""Transcripción por trozos: durabilidad (checkpoint/resume) + paralelismo."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimedWord:
    start: float
    end: float
    word: str


@dataclass
class TimedSegment:
    start: float
    end: float
    text: str
    words: list[TimedWord] | None = None


def shift_segments(segments, offset: float) -> list[TimedSegment]:
    """Copia segmentos aplicando `offset` a start/end del segmento y de cada palabra.
    Los Segment de faster-whisper son inmutables; devolvemos TimedSegment nuevos."""
    out: list[TimedSegment] = []
    for s in segments:
        words = getattr(s, "words", None)
        tw = (
            [TimedWord(w.start + offset, w.end + offset, w.word) for w in words]
            if words
            else None
        )
        out.append(TimedSegment(s.start + offset, s.end + offset, s.text, tw))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): TimedSegment/TimedWord + shift_segments (offset)"
```

---

### Task 2: Parseo de silencedetect

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Produces: `parse_silences(stderr: str) -> list[tuple[float, float]]` — extrae pares `(silence_start, silence_end)` de la salida de `ffmpeg silencedetect`. Silencios sin cierre (final de archivo) se descartan.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
from speechtotext.core.chunked import parse_silences


def test_parse_silences_extrae_pares():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 12.5\n"
        "[silencedetect @ 0x1] silence_end: 13.2 | silence_duration: 0.7\n"
        "[silencedetect @ 0x1] silence_start: 601.0\n"
        "[silencedetect @ 0x1] silence_end: 602.4 | silence_duration: 1.4\n"
    )
    assert parse_silences(stderr) == [(12.5, 13.2), (601.0, 602.4)]


def test_parse_silences_descarta_start_sin_end():
    stderr = "silence_start: 5.0\nsilence_end: 6.0\nsilence_start: 900.0\n"
    assert parse_silences(stderr) == [(5.0, 6.0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_parse_silences_extrae_pares -q`
Expected: FAIL — `ImportError: cannot import name 'parse_silences'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir; import re arriba)
import re

_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)")


def parse_silences(stderr: str) -> list[tuple[float, float]]:
    starts = [float(m.group(1)) for m in _SIL_START.finditer(stderr)]
    ends = [float(m.group(1)) for m in _SIL_END.finditer(stderr)]
    return list(zip(starts, ends))  # zip corta el start final sin end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): parse_silences (salida de ffmpeg silencedetect)"
```

---

### Task 3: Elección de cortes conscientes del silencio

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Consumes: `parse_silences` (Task 2, indirecto).
- Produces: `pick_cuts(silences: list[tuple[float, float]], duration: float, target_len: float = 600.0, search: float = 60.0) -> list[tuple[float, float]]` — rangos `(start, end)` contiguos que cubren `[0, duration]`. Corta en el punto medio del silencio más cercano a cada frontera `cut_previo + target_len`, si hay uno dentro de `±search`; si no, corte fijo en la frontera. `duration <= target_len` → `[(0.0, duration)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
from speechtotext.core.chunked import pick_cuts


def test_pick_cuts_corta_en_silencio_cercano():
    # frontera en 600; hay silencio en 601.0-602.4 (mid 601.7) dentro de ±60 -> corta en 601.7
    chunks = pick_cuts([(601.0, 602.4)], duration=1200.0, target_len=600.0, search=60.0)
    assert chunks == [(0.0, 601.7), (601.7, 1200.0)]


def test_pick_cuts_fijo_si_no_hay_silencio_cerca():
    # silencio lejos de la frontera 600 -> corte fijo en 600
    chunks = pick_cuts([(100.0, 101.0)], duration=1200.0, target_len=600.0, search=60.0)
    assert chunks == [(0.0, 600.0), (600.0, 1200.0)]


def test_pick_cuts_audio_corto_un_solo_trozo():
    assert pick_cuts([], duration=300.0, target_len=600.0) == [(0.0, 300.0)]


def test_pick_cuts_cubre_toda_la_duracion_contiguo():
    chunks = pick_cuts([], duration=1500.0, target_len=600.0)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 1500.0
    for a, b in zip(chunks, chunks[1:]):
        assert a[1] == b[0]  # contiguo, sin huecos ni solapes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_pick_cuts_corta_en_silencio_cercano -q`
Expected: FAIL — `ImportError: cannot import name 'pick_cuts'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir)
def pick_cuts(
    silences: list[tuple[float, float]],
    duration: float,
    target_len: float = 600.0,
    search: float = 60.0,
) -> list[tuple[float, float]]:
    mids = [(s + e) / 2 for s, e in silences]
    cuts: list[float] = []
    prev = 0.0
    boundary = target_len
    while boundary < duration - 1.0:
        near = [m for m in mids if abs(m - boundary) <= search and m > prev + 1.0]
        cut = min(near, key=lambda m: abs(m - boundary)) if near else boundary
        cuts.append(cut)
        prev = cut
        boundary = cut + target_len
    bounds = [0.0, *cuts, duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): pick_cuts (cortes conscientes del silencio con fallback fijo)"
```

---

### Task 4: `plan_chunks` (ffmpeg silencedetect + cortes)

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Consumes: `parse_silences`, `pick_cuts`.
- Produces: `plan_chunks(audio: Path, duration: float, target_len: float = 600.0) -> list[tuple[float, float]]` — corre `ffmpeg -i audio -af silencedetect=noise=-30dB:d=0.5 -f null -`, parsea stderr, devuelve `pick_cuts(...)`. Si ffmpeg falla, fallback a cortes fijos (`pick_cuts([], duration, target_len)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
from pathlib import Path

import speechtotext.core.chunked as chunked


def test_plan_chunks_usa_silencedetect(monkeypatch):
    fake = SimpleNamespace(stderr=b"silence_start: 601.0\nsilence_end: 602.4\n", returncode=0)
    monkeypatch.setattr(chunked.subprocess, "run", lambda *a, **k: fake)
    chunks = chunked.plan_chunks(Path("x.mp3"), duration=1200.0, target_len=600.0)
    assert chunks == [(0.0, 601.7), (601.7, 1200.0)]


def test_plan_chunks_fallback_si_ffmpeg_falla(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no ffmpeg")
    monkeypatch.setattr(chunked.subprocess, "run", boom)
    chunks = chunked.plan_chunks(Path("x.mp3"), duration=1200.0, target_len=600.0)
    assert chunks == [(0.0, 600.0), (600.0, 1200.0)]  # cortes fijos
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_plan_chunks_usa_silencedetect -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'plan_chunks'` (o `subprocess`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir; import subprocess arriba)
import subprocess
from pathlib import Path


def plan_chunks(audio: Path, duration: float, target_len: float = 600.0) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio),
        "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
        silences = parse_silences(proc.stderr.decode("utf-8", errors="ignore"))
    except (FileNotFoundError, OSError):
        silences = []  # sin ffmpeg -> cortes fijos
    return pick_cuts(silences, duration, target_len)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): plan_chunks (silencedetect + fallback fijo)"
```

---

### Task 5: Clave de checkpoint por trozo

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Produces: `chunk_path(audio: Path, opts: dict, model: str, start: float, end: float) -> Path` — `~/.speechtotext/chunks/{sha1[:16]}.json`, clave = `{audio.resolve()}|{size}|{mtime}|{model}|{opts["language"]}|{opts["beam_size"]}|{opts["vad_filter"]}|{opts["hotwords"]}|{opts["word_timestamps"]}|{start}|{end}`. Reusa `finder._home`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
def _opts(**over):
    base = dict(language="es", beam_size=5, vad_filter=True, hotwords=None, word_timestamps=False)
    base.update(over)
    return base


def test_chunk_path_determinista(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTEXT_UNUSED", "x")
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    p1 = chunked.chunk_path(audio, _opts(), "large-v3", 0.0, 600.0)
    p2 = chunked.chunk_path(audio, _opts(), "large-v3", 0.0, 600.0)
    assert p1 == p2
    assert p1.parent == tmp_path / "chunks"


def test_chunk_path_cambia_con_cada_parametro(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    base = chunked.chunk_path(audio, _opts(), "large-v3", 0.0, 600.0)
    assert chunked.chunk_path(audio, _opts(), "medium", 0.0, 600.0) != base
    assert chunked.chunk_path(audio, _opts(hotwords="Boconó"), "large-v3", 0.0, 600.0) != base
    assert chunked.chunk_path(audio, _opts(word_timestamps=True), "large-v3", 0.0, 600.0) != base
    assert chunked.chunk_path(audio, _opts(), "large-v3", 600.0, 1200.0) != base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_chunk_path_determinista -q`
Expected: FAIL — `AttributeError: ... 'chunk_path'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir; import hashlib arriba)
import hashlib

from speechtotext.core.finder import _home


def chunk_path(audio: Path, opts: dict, model: str, start: float, end: float) -> Path:
    st = audio.stat()
    key = "|".join(str(x) for x in (
        audio.resolve(), st.st_size, int(st.st_mtime), model,
        opts["language"], opts["beam_size"], opts["vad_filter"],
        opts["hotwords"], opts["word_timestamps"], start, end,
    ))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    d = _home() / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): chunk_path (clave de contenido por trozo, patrón finder)"
```

---

### Task 6: Serialización de segmentos (round-trip JSON)

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Consumes: `TimedSegment`, `TimedWord` (Task 1).
- Produces: `seg_to_dict(seg: TimedSegment) -> dict`; `seg_from_dict(d: dict) -> TimedSegment`. Round-trip exacto, incluyendo `words=None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
from speechtotext.core.chunked import seg_from_dict, seg_to_dict


def test_seg_roundtrip_con_palabras():
    seg = TimedSegment(600.0, 602.0, " hola", [TimedWord(600.0, 601.0, " hola")])
    back = seg_from_dict(seg_to_dict(seg))
    assert back == seg


def test_seg_roundtrip_sin_palabras():
    seg = TimedSegment(1.0, 2.0, "x", None)
    assert seg_from_dict(seg_to_dict(seg)) == seg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_seg_roundtrip_con_palabras -q`
Expected: FAIL — `ImportError: cannot import name 'seg_to_dict'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir)
def seg_to_dict(seg: TimedSegment) -> dict:
    d = {"start": seg.start, "end": seg.end, "text": seg.text}
    if seg.words is not None:
        d["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in seg.words]
    return d


def seg_from_dict(d: dict) -> TimedSegment:
    words = d.get("words")
    tw = [TimedWord(w["start"], w["end"], w["word"]) for w in words] if words is not None else None
    return TimedSegment(d["start"], d["end"], d["text"], tw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): serialización round-trip de TimedSegment"
```

---

### Task 7: `transcribe_chunk` (checkpoint / resume)

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Consumes: `chunk_path`, `shift_segments`, `seg_to_dict`, `seg_from_dict`.
- Produces: `transcribe_chunk(audio: Path, start: float, end: float, opts: dict, model, model_name: str) -> tuple[list[TimedSegment], bool]` — devuelve `(segmentos_globales, from_cache)`. Si el checkpoint existe y parsea → cargar (sin ffmpeg ni modelo). Si no → recortar wav con ffmpeg, `model.transcribe(wav, **opts)`, `shift_segments(+start)`, guardar JSON, borrar wav.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
import json


def test_transcribe_chunk_usa_cache_si_existe(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    # sembrar checkpoint con un segmento global ya offseteado
    p = chunked.chunk_path(audio, _opts(), "large-v3", 600.0, 1200.0)
    p.write_text(json.dumps({"segments": [{"start": 601.0, "end": 602.0, "text": " cache"}]}))

    def boom(*a, **k):
        raise AssertionError("no debió transcribir ni llamar ffmpeg")
    monkeypatch.setattr(chunked.subprocess, "run", boom)
    model = SimpleNamespace(transcribe=boom)

    segs, cached = chunked.transcribe_chunk(audio, 600.0, 1200.0, _opts(), model, "large-v3")
    assert cached is True
    assert segs[0].text == " cache" and segs[0].start == 601.0


def test_transcribe_chunk_transcribe_y_guarda(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"12345")
    monkeypatch.setattr(chunked.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr=b""))
    # modelo devuelve segmentos LOCALES (relativos al trozo)
    local = [SimpleNamespace(start=1.0, end=2.0, text=" hola", words=None)]
    model = SimpleNamespace(transcribe=lambda wav, **k: (iter(local), SimpleNamespace()))

    segs, cached = chunked.transcribe_chunk(audio, 600.0, 1200.0, _opts(), model, "large-v3")
    assert cached is False
    assert segs[0].start == 601.0 and segs[0].end == 602.0  # +600 offset
    # persistió el checkpoint con timestamps globales
    p = chunked.chunk_path(audio, _opts(), "large-v3", 600.0, 1200.0)
    assert p.exists()
    assert json.loads(p.read_text())["segments"][0]["start"] == 601.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_transcribe_chunk_usa_cache_si_existe -q`
Expected: FAIL — `AttributeError: ... 'transcribe_chunk'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir; import json, tempfile, os arriba)
import json
import os
import tempfile


def transcribe_chunk(audio, start, end, opts, model, model_name):
    path = chunk_path(audio, opts, model_name, start, end)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [seg_from_dict(d) for d in data["segments"]], True
        except (json.JSONDecodeError, KeyError):
            pass  # checkpoint corrupto -> recomputar

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(end - start), "-i", str(audio),
            "-ar", "16000", "-ac", "1", tmp,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        segments_iter, _info = model.transcribe(tmp, **opts)
        segs = shift_segments(list(segments_iter), start)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    path.write_text(
        json.dumps({"segments": [seg_to_dict(s) for s in segs]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return segs, False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): transcribe_chunk con checkpoint/resume por trozo"
```

---

### Task 8: `probe_duration`

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Produces: `probe_duration(audio: Path) -> float` — usa PyAV (`av.open(audio).duration / 1e6`, en microsegundos). Si el contenedor no reporta duración (`None`), fallback a `ffprobe`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
def test_probe_duration_desde_pyav(monkeypatch):
    container = SimpleNamespace(duration=1245_000000)  # microsegundos

    class FakeAv:
        @staticmethod
        def open(path):
            return container
    monkeypatch.setattr(chunked, "av", FakeAv)
    assert chunked.probe_duration(Path("x.mp3")) == 1245.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_probe_duration_desde_pyav -q`
Expected: FAIL — `AttributeError: ... 'probe_duration'` (o `av`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir; import av arriba)
import av


def probe_duration(audio: Path) -> float:
    container = av.open(str(audio))
    if container.duration:
        return container.duration / 1_000_000.0
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(audio)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): probe_duration (PyAV + fallback ffprobe)"
```

---

### Task 9: `run_chunked` (pool + reensamblaje + info + progreso)

**Files:**
- Modify: `src/speechtotext/core/chunked.py`
- Test: `tests/test_chunked.py`

**Interfaces:**
- Consumes: `plan_chunks`, `transcribe_chunk`, `probe_duration`.
- Produces: `run_chunked(audio: Path, opts: dict, jobs: int, model_name: str, device: str, compute_type: str, log=print) -> tuple[list[TimedSegment], SimpleNamespace]`. Construye **un** `WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=max(1, (os.cpu_count() or 1)//jobs), num_workers=jobs)`. `ThreadPoolExecutor(max_workers=jobs)` corre `transcribe_chunk` por trozo. Reensambla en orden de trozo. `info = SimpleNamespace(language, language_probability, duration)`. Emite `[k/n] mm:ss–mm:ss OK (cache|nuevo)` por trozo vía `log`.

**Nota de test:** monkeypatch `chunked.WhisperModel`, `chunked.probe_duration`, `chunked.plan_chunks` y `chunked.transcribe_chunk` para no cargar modelos.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
def test_run_chunked_reensambla_en_orden_y_arma_info(monkeypatch):
    monkeypatch.setattr(chunked, "WhisperModel", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(chunked, "probe_duration", lambda audio: 1200.0)
    monkeypatch.setattr(chunked, "plan_chunks", lambda audio, dur, **k: [(0.0, 600.0), (600.0, 1200.0)])

    def fake_chunk(audio, start, end, opts, model, model_name):
        return [TimedSegment(start + 1.0, start + 2.0, f" t{int(start)}")], False
    monkeypatch.setattr(chunked, "transcribe_chunk", fake_chunk)

    lines = []
    segs, info = chunked.run_chunked(
        Path("x.mp3"), _opts(), jobs=2, model_name="large-v3",
        device="cpu", compute_type="int8", log=lines.append,
    )
    assert [s.text for s in segs] == [" t0", " t600"]  # orden de trozo
    assert (segs[0].start, segs[1].start) == (1.0, 601.0)  # timestamps globales
    assert info.duration == 1200.0
    assert len(lines) == 2  # una línea por trozo


def test_run_chunked_marca_cache(monkeypatch):
    monkeypatch.setattr(chunked, "WhisperModel", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(chunked, "probe_duration", lambda audio: 600.0)
    monkeypatch.setattr(chunked, "plan_chunks", lambda audio, dur, **k: [(0.0, 600.0)])
    monkeypatch.setattr(chunked, "transcribe_chunk",
                        lambda *a, **k: ([TimedSegment(1.0, 2.0, " x")], True))
    lines = []
    chunked.run_chunked(Path("x.mp3"), _opts(), 1, "m", "cpu", "int8", log=lines.append)
    assert "cache" in lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_run_chunked_reensambla_en_orden_y_arma_info -q`
Expected: FAIL — `AttributeError: ... 'run_chunked'` (o `WhisperModel`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/speechtotext/core/chunked.py (añadir; imports arriba)
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

from faster_whisper import WhisperModel


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def run_chunked(audio, opts, jobs, model_name, device, compute_type, log=print):
    duration = probe_duration(audio)
    chunks = plan_chunks(audio, duration)
    jobs = max(1, min(jobs, len(chunks)))
    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        cpu_threads=max(1, (os.cpu_count() or 1) // jobs), num_workers=jobs,
    )
    results: list = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(transcribe_chunk, audio, s, e, opts, model, model_name): i
            for i, (s, e) in enumerate(chunks)
        }
        for done, fut in enumerate(as_completed(futs), start=1):
            i = futs[fut]
            results[i], from_cache = fut.result()
            s, e = chunks[i]
            tag = "cache" if from_cache else "nuevo"
            log(f"[{done}/{len(chunks)}] {_mmss(s)}-{_mmss(e)} OK ({tag})")
    segments = [seg for chunk_segs in results for seg in chunk_segs]
    info = SimpleNamespace(language=opts.get("language") or "es",
                           language_probability=1.0, duration=duration)
    return segments, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/chunked.py tests/test_chunked.py
git commit -m "feat(chunked): run_chunked (pool compartido + reensamblaje + progreso)"
```

---

### Task 10: Decisión de troceo + wiring en el CLI

**Files:**
- Modify: `src/speechtotext/core/chunked.py` (añadir `_should_chunk`, `CHUNK_THRESHOLD`)
- Modify: `src/speechtotext/cli/app.py` (flags + rama en `transcribe_file`)
- Test: `tests/test_chunked.py`

**Interfaces:**
- Consumes: `run_chunked`, `probe_duration`.
- Produces: `should_chunk(duration: float, chunk_flag: bool | None, threshold: float = CHUNK_THRESHOLD) -> bool` — `chunk_flag` explícito manda; si es `None`, auto = `duration > threshold`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunked.py (añadir)
from speechtotext.core.chunked import CHUNK_THRESHOLD, should_chunk


def test_should_chunk_auto_por_umbral():
    assert should_chunk(CHUNK_THRESHOLD + 1, None) is True
    assert should_chunk(CHUNK_THRESHOLD - 1, None) is False


def test_should_chunk_flag_explicito_manda():
    assert should_chunk(10.0, True) is True       # forzar en audio corto
    assert should_chunk(99999.0, False) is False  # forzar off en audio largo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py::test_should_chunk_auto_por_umbral -q`
Expected: FAIL — `ImportError: cannot import name 'should_chunk'`

- [ ] **Step 3: Write minimal implementation (chunked.py)**

```python
# src/speechtotext/core/chunked.py (añadir cerca de los otros)
CHUNK_THRESHOLD = 1200.0  # s (20 min): por encima, auto-trocea


def should_chunk(duration: float, chunk_flag: bool | None, threshold: float = CHUNK_THRESHOLD) -> bool:
    if chunk_flag is not None:
        return chunk_flag
    return duration > threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chunked.py -q`
Expected: PASS

- [ ] **Step 5: Wire into `transcribe_file` (cli/app.py)**

Añade el parámetro `chunk: bool | None = None` y `jobs: int = 4` a la firma de `transcribe_file` (tras `hotwords`). Sustituye el bloque de transcripción (el `with Progress(...)` que envuelve `whisper.transcribe(...)` hasta `segments = list(segments_iter)`) por la rama:

```python
    from speechtotext.core.chunked import run_chunked, should_chunk, probe_duration

    opts = _transcribe_opts(lang, beam_size, vad, hotwords, word_timestamps=diarize)
    if should_chunk(probe_duration(audio), chunk):
        console.print(f"[bold]Troceado[/bold] (jobs={jobs}) · {model}")
        segments, info = run_chunked(
            audio, opts, jobs, model_name=model, device=device,
            compute_type=compute_type, log=lambda m: console.print(f"  {m}", markup=False),
        )
    else:
        whisper = WhisperModel(model, device=device, compute_type=compute_type)
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(), console=console,
        ) as progress:
            task = progress.add_task(f"Transcribiendo {audio.name}", total=None)
            segments_iter, info = whisper.transcribe(str(audio), **opts)
            segments = list(segments_iter)
            progress.update(task, completed=1)
```

Nota: la construcción de `whisper` y `_transcribe_opts` se mueve dentro de la rama `else` (ya no antes). Verifica que `console.print("Idioma detectado...")` siga funcionando: `info` en ambas ramas tiene `.language`, `.language_probability`, `.duration`.

- [ ] **Step 6: Add CLI flags to `transcribe` command (cli/app.py)**

En la firma del comando `transcribe`, tras `hotwords_file`, añade:

```python
    chunk: Optional[bool] = typer.Option(
        None, "--chunk/--no-chunk",
        help="Trocear el audio para checkpoint/resume + paralelismo. Auto si dura > 20 min.",
    ),
    jobs: int = typer.Option(
        4, "--jobs", "-j", help="Trozos en paralelo al trocear (comparten un modelo).",
    ),
```

Y pásalos en la llamada a `transcribe_file(...)`:

```python
    transcribe_file(
        audio, output, language, model, formats, device, compute_type,
        vad, beam_size, diarize, speakers, identify, threshold,
        hotwords=_resolve_hotwords(hotwords, hotwords_file),
        chunk=chunk, jobs=jobs,
    )
```

- [ ] **Step 7: Run full suite + smoke import**

Run:
```bash
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
.venv/Scripts/speechtotext.exe transcribe --help
```
Expected: todos los tests PASS; el `--help` muestra `--chunk/--no-chunk` y `--jobs`.

- [ ] **Step 8: Commit**

```bash
git add src/speechtotext/core/chunked.py src/speechtotext/cli/app.py tests/test_chunked.py
git commit -m "feat(chunked): should_chunk + wiring en transcribe_file (flags --chunk/--jobs)"
```

---

## Verificación de extremo a extremo (tras el plan)

No cubierta por tests unitarios (usa modelos/ffmpeg reales) — la corre Samuel:

```bash
# archivo largo con troceo automático + resume (interrumpible)
.venv/Scripts/speechtotext.exe transcribe "<audio 2h>.mp3" -m large-v3 -f txt,srt,json
# repetir tras Ctrl-C: debe saltar los trozos ya cacheados
```

Criterios: interrumpir/reanudar pierde ≤ 1 trozo; wall-clock baja con `-j`; txt/srt/json con timestamps globales; archivo corto sin cambios.

## Self-review (hecho)

- **Cobertura del spec:** representación+offset (T1), fronteras silencio (T2-T4), checkpoint (T5-T7), duración (T8), pool+reensamblaje+info+progreso (T9), decisión+wiring+diarización-compose (T10, la diarización compone sin tocar `diarization.py` porque los segmentos ya traen timestamps y palabras globales). Todos los criterios de aceptación mapeados.
- **Placeholders:** ninguno; código real en cada paso.
- **Consistencia de tipos:** `TimedSegment`/`TimedWord`, `opts` (dict de `_transcribe_opts`), firmas de `transcribe_chunk`/`run_chunked`/`chunk_path` coinciden entre tareas.
