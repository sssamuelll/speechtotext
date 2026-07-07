# `find` (buscador de segmento) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subcomando `speechtotext find AUDIO "consulta"` que ubica contenido en un audio largo (índice `tiny` cacheado + regiones agrupadas) y, con `--extract`, recorta y transcribe en calidad el tramo elegido.

**Architecture:** Módulo nuevo `core/finder.py` con la lógica pura (normalizar, buscar, agrupar regiones) más la construcción/caché del índice (faster-whisper). El comando `find` vive en `cli/app.py`; el cuerpo de `transcribe` se extrae a `transcribe_file(...)` para que `find --extract` lo reutilice.

**Tech Stack:** Python ≥3.10, faster-whisper (ya, base), typer, rich, ffmpeg (vía subprocess), pytest.

## Global Constraints

- Python ≥ 3.10; strings de cara al usuario en español, sin emoji.
- Sin dependencias nuevas: `finder` usa stdlib + faster-whisper (ya en base).
- Directorio de datos: `${SPEECHTOTEXT_HOME:-~/.speechtotext}`; el índice va en su subcarpeta `index/`.
- `core/finder.py`: las funciones puras (`normalize`, `search`, `cluster_regions`, `clip_window`) NO importan faster-whisper a nivel de módulo (import perezoso dentro de `build_index`).
- Coincidencia: sin acentos, sin mayúsculas, un segmento pega si contiene CUALQUIER término de la consulta.
- Trabajar en la rama `feat/find-buscador`. Venv Windows en `.\.venv\`; pytest con `.\.venv\Scripts\python.exe -m pytest`. Si UnicodeEncodeError, `PYTHONUTF8=1`.

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `src/speechtotext/core/finder.py` | CREAR · índice (build/load/cache) + normalize + search + cluster_regions + clip_window |
| `src/speechtotext/cli/app.py` | MODIFICAR · refactor `transcribe`→`transcribe_file`; comando `find` |
| `tests/test_finder.py` | CREAR · tests puros (normalize, search, cluster_regions, clip_window) |
| `tests/test_finder_index.py` | CREAR · tests de índice/caché (load desde caché sembrada; build gated) |
| `tests/test_cli_find.py` | CREAR · test del comando `find` en modo ubicar (caché sembrada) |
| `README.md` | MODIFICAR · sección de `find` |

---

## Task 1: `finder.py` — lógica pura (normalize, regiones, clip_window)

**Files:**
- Create: `src/speechtotext/core/finder.py`
- Create: `tests/test_finder.py`

**Interfaces:**
- Produces:
  - `@dataclass Region(start: float, end: float, hits: int, matches: int, snippet: str)`
  - `normalize(text: str) -> str`
  - `cluster_regions(hits: list[tuple[float, float, str, int]], gap: float) -> list[Region]`
  - `search(segments: list[dict], query: str, gap: float = 60.0, top: int = 5) -> list[Region]`
  - `clip_window(start: float, end: float, context: float) -> tuple[float, float]` (devuelve `(inicio, duración)`)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_finder.py`:
```python
from speechtotext.core.finder import (
    normalize, search, cluster_regions, clip_window, Region,
)


def _seg(s, e, t):
    return {"start": s, "end": e, "text": t}


def test_normalize_strips_accents_and_case():
    assert normalize("Sísmica ÑOÑO") == "sismica nono"


def test_search_groups_contiguous_into_one_region():
    segs = [
        _seg(0, 1, "hola"),
        _seg(1, 2, "la vulnerabilidad sismica"),
        _seg(2, 3, "sismica otra vez"),
        _seg(500, 501, "nada"),
    ]
    regs = search(segs, "vulnerabilidad sismica", gap=60, top=5)
    assert len(regs) == 1
    assert regs[0].start == 1 and regs[0].end == 3
    assert regs[0].hits == 2


def test_search_splits_on_large_gap():
    segs = [_seg(0, 1, "sismica"), _seg(200, 201, "sismica")]  # hueco 199 > 60
    regs = search(segs, "sismica", gap=60, top=5)
    assert len(regs) == 2


def test_search_ranks_denser_region_first():
    segs = [
        _seg(0, 1, "sismica"),
        _seg(500, 501, "sismica"), _seg(501, 502, "sismica"), _seg(502, 503, "sismica"),
    ]
    regs = search(segs, "sismica", gap=60, top=5)
    assert regs[0].start == 500  # la región densa va primero


def test_search_accent_and_case_insensitive():
    regs = search([_seg(0, 1, "la SÍSMICA de hoy")], "sismica", gap=60, top=5)
    assert len(regs) == 1


def test_search_no_match_is_empty():
    assert search([_seg(0, 1, "hola")], "inexistente", gap=60, top=5) == []


def test_clip_window_clamps_and_pads():
    assert clip_window(100.0, 160.0, 10.0) == (90.0, 80.0)
    assert clip_window(5.0, 15.0, 10.0) == (0.0, 30.0)  # no baja de 0
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_finder.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar la parte pura de `finder.py`**

```python
"""Buscador de segmento: índice de transcripción tiny + búsqueda por regiones."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


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
```

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_finder.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/finder.py tests/test_finder.py
git commit -m "feat(finder): lógica pura de búsqueda por regiones + clip_window"
```

---

## Task 2: `finder.py` — índice y caché

**Files:**
- Modify: `src/speechtotext/core/finder.py`
- Create: `tests/test_finder_index.py`

**Interfaces:**
- Produces:
  - `index_path(audio: Path, scan_model: str) -> Path`
  - `build_index(audio: Path, scan_model: str) -> list[dict]` (usa faster-whisper; import perezoso)
  - `load_or_build_index(audio: Path, scan_model: str, rebuild: bool = False) -> tuple[list[dict], bool]`
    (el bool es `True` si se usó caché)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_finder_index.py`:
```python
import json

from speechtotext.core.finder import index_path, load_or_build_index


def test_index_path_deterministic_and_model_sensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    p1 = index_path(audio, "tiny")
    p2 = index_path(audio, "tiny")
    p3 = index_path(audio, "base")
    assert p1 == p2
    assert p1 != p3


def test_load_uses_seeded_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    seed = {"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}
    index_path(audio, "tiny").write_text(json.dumps(seed), encoding="utf-8")

    segments, cached = load_or_build_index(audio, "tiny", rebuild=False)
    assert cached is True
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hola"}]
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_finder_index.py -v`
Expected: FAIL — funciones inexistentes.

- [ ] **Step 3: Añadir a `finder.py` (imports arriba + funciones al final)**

Añade estos imports al inicio del archivo (junto a los existentes):
```python
import hashlib
import json
import os
from pathlib import Path
```

Añade estas funciones al final del archivo:
```python
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

    from speechtotext.core.audio import transcode_to_wav

    wav = transcode_to_wav(audio.read_bytes())
    try:
        model = WhisperModel(scan_model, device="cpu", compute_type="int8")
        segments_iter, _info = model.transcribe(str(wav), vad_filter=True)
        return [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            for s in segments_iter
        ]
    finally:
        wav.unlink(missing_ok=True)


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
```

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_finder_index.py tests/test_finder.py -v`
Expected: PASS (los 2 nuevos + los 7 de Task 1).

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/finder.py tests/test_finder_index.py
git commit -m "feat(finder): índice tiny cacheado por archivo (build/load)"
```

---

## Task 3: Refactor `transcribe` → `transcribe_file`

**Files:**
- Modify: `src/speechtotext/cli/app.py`

**Interfaces:**
- Produces: `transcribe_file(audio: Path, output: Optional[Path], language: str, model: str, formats: str, device: str, compute_type: str, vad: bool, beam_size: int, diarize: bool, speakers: Optional[int], identify: bool, threshold: float) -> None`

**Objetivo:** extraer el CUERPO actual de la función `transcribe` a una función libre `transcribe_file(...)`, sin cambiar comportamiento. El comando `transcribe` conserva su firma (con todas las opciones typer) y solo delega.

- [ ] **Step 1: Leer el archivo y mover el cuerpo**

Lee `src/speechtotext/cli/app.py`. La función `transcribe` tiene una firma con opciones typer y un cuerpo que hace: `parse_formats`, `_resolve_output_base`, resolución de `lang`/`compute_type`, print del modelo, `WhisperModel` + `Progress` + `segments`, print del idioma, `if diarize: segments = _run_diarization(...)`, y el loop de `writers`.

Crea una función libre `transcribe_file` con la firma de arriba que contenga EXACTAMENTE ese cuerpo actual (los nombres de parámetro coinciden: `audio, output, language, model, formats, device, compute_type, vad, beam_size, diarize, speakers, identify, threshold`). NO reescribas la lógica de memoria: mueve el cuerpo tal cual.

Ubica `transcribe_file` justo DESPUÉS de `_resolve_output_base` y ANTES de la función `transcribe`.

- [ ] **Step 2: Hacer que `transcribe` delegue**

Reemplaza el cuerpo de la función-comando `transcribe` (todo lo que hay tras el docstring) por una sola llamada:
```python
    transcribe_file(
        audio, output, language, model, formats, device, compute_type,
        vad, beam_size, diarize, speakers, identify, threshold,
    )
```
Deja la firma del comando `transcribe` (con sus `typer.Option`/`typer.Argument`) y su docstring intactos.

- [ ] **Step 3: Verificar que nada se rompió**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS (toda la suite existente, sin regresiones).

Run: `.\.venv\Scripts\python.exe -c "from speechtotext.cli.app import transcribe_file; print('ok')"`
Expected: imprime `ok`.

Run: `.\.venv\Scripts\speechtotext.exe transcribe --help`
Expected: el comando `transcribe` muestra sus opciones como antes (incluyendo `--diarize`).

- [ ] **Step 4: Commit**

```bash
git add src/speechtotext/cli/app.py
git commit -m "refactor(cli): extraer transcribe_file() reusable del comando transcribe"
```

---

## Task 4: Comando `find` — modo ubicar

**Files:**
- Modify: `src/speechtotext/cli/app.py`
- Create: `tests/test_cli_find.py`

**Interfaces:**
- Consumes: `finder.load_or_build_index`, `finder.search` (Tasks 1-2).
- Produces: comando `find` (modo ubicar; `--extract` se implementa en Task 5 pero el parámetro se declara aquí y por ahora solo avisa "no implementado" si se usa... NO: en esta task `--extract` se declara y se deja el hook; ver Step 3).

- [ ] **Step 1: Escribir el test que falla**

`tests/test_cli_find.py`:
```python
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
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli_find.py -v`
Expected: FAIL — comando `find` no existe.

- [ ] **Step 3: Añadir el helper `_fmt` y el comando `find` a `app.py`**

Añade el helper (cerca de `_resolve_output_base`):
```python
def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
```

Añade el comando (después de `transcribe`, antes de `enroll`):
```python
@app.command()
def find(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio o vídeo a buscar."),
    query: str = typer.Argument(..., help="Palabras a buscar."),
    extract: bool = typer.Option(False, "--extract", "-e", help="Recortar + transcribir la región."),
    region: int = typer.Option(1, "--region", help="Qué región extraer (1 = la más densa)."),
    model: str = typer.Option("small", "--model", "-m", help="Modelo para la transcripción en calidad."),
    scan_model: str = typer.Option("tiny", "--scan-model", help="Modelo del índice."),
    language: str = typer.Option("es", "--language", "-l", help="Idioma de la transcripción del tramo."),
    formats: str = typer.Option("txt,srt", "--formats", "-f", help="Formatos de salida del tramo."),
    diarize: bool = typer.Option(False, "--diarize", "-D", help="Diarizar el tramo extraído."),
    speakers: Optional[int] = typer.Option(None, "--speakers", help="Nº de hablantes (pista)."),
    identify: bool = typer.Option(True, "--identify/--no-identify", help="Nombrar voces registradas."),
    threshold: float = typer.Option(0.5, "--threshold", help="Umbral de coincidencia de voz."),
    context: float = typer.Option(10.0, "--context", help="Segundos de margen al recortar."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Carpeta de salida del tramo."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Forzar reconstrucción del índice."),
    top: int = typer.Option(5, "--top", help="Cuántas regiones listar."),
) -> None:
    """Busca contenido en un audio largo; con --extract recorta y transcribe el tramo."""
    from speechtotext.core import finder

    segments, cached = finder.load_or_build_index(audio, scan_model, rebuild)
    console.print(f"Índice: {'caché' if cached else 'construido'} ({scan_model}, {len(segments)} segmentos)")

    regions = finder.search(segments, query, top=top)
    if not regions:
        console.print(f'No se encontró "{query}" en el audio.')
        raise typer.Exit(0)

    console.print(f"{len(regions)} regiones para \"{query}\":")
    for i, r in enumerate(regions, start=1):
        console.print(f"  {i}.  {_fmt(r.start)} – {_fmt(r.end)}  ({r.hits})  \"{r.snippet}\"")

    if extract:
        _extract_region(
            audio, regions, region, output, language, model, formats,
            diarize, speakers, identify, threshold, context,
        )
```

En esta task, `_extract_region` aún no existe; para que el módulo importe, añade un stub temporal JUSTO antes del comando `find`:
```python
def _extract_region(audio, regions, region, output, language, model, formats,
                    diarize, speakers, identify, threshold, context):
    raise NotImplementedError  # se implementa en Task 5
```
(Los tests de esta task no usan `--extract`, así que el stub no se ejecuta.)

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli_find.py tests/ -q`
Expected: PASS (los 2 nuevos + toda la suite).

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/cli/app.py tests/test_cli_find.py
git commit -m "feat(cli): comando find (modo ubicar) con regiones rankeadas"
```

---

## Task 5: `find --extract` — recorte + transcripción del tramo

**Files:**
- Modify: `src/speechtotext/cli/app.py`

**Interfaces:**
- Consumes: `finder.clip_window` (Task 1), `transcribe_file` (Task 3), `core.audio.FfmpegMissingError`.

- [ ] **Step 1: Añadir el helper `_fmt_file` y reemplazar el stub `_extract_region`**

Añade el helper (cerca de `_fmt`):
```python
def _fmt_file(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
```

Reemplaza el stub `_extract_region` por:
```python
def _extract_region(audio, regions, region, output, language, model, formats,
                    diarize, speakers, identify, threshold, context):
    import subprocess

    from speechtotext.core.finder import clip_window

    if region < 1 or region > len(regions):
        console.print(f"[red]Región {region} fuera de rango (hay {len(regions)}).[/red]")
        raise typer.Exit(1)

    r = regions[region - 1]
    begin, duration = clip_window(r.start, r.end, context)
    base_dir = output if output is not None else audio.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{audio.stem}_{_fmt_file(r.start)}-{_fmt_file(r.end)}"
    clip = base_dir / f"{stem}.wav"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(begin), "-t", str(duration), "-i", str(audio),
        "-ar", "16000", "-ac", "1", str(clip),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        console.print(
            "[red]ffmpeg no está en el PATH.[/red] "
            "Instálalo: [cyan]winget install Gyan.FFmpeg[/cyan]"
        )
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]No se pudo recortar el audio:[/red] {e.stderr.decode(errors='ignore')[:200]}")
        raise typer.Exit(1)

    console.print(f"  [green]Recorte[/green] {clip} ({_fmt(r.start)}–{_fmt(r.end)})")
    transcribe_file(
        clip, base_dir, language, model, formats,
        "cpu", "auto", True, 5, diarize, speakers, identify, threshold,
    )
```

Nota: se pasa `base_dir` como `output` a `transcribe_file`; como el clip se llama `<stem>_<tramo>.wav`, el transcript sale como `<stem>_<tramo>.txt/.srt` junto al clip.

- [ ] **Step 2: Verificar que la suite sigue verde**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS (sin regresiones; los tests de `find` no ejercen `--extract`, que se valida a mano abajo).

- [ ] **Step 3: Smoke manual de `--extract`**

Con un audio de voz de ~1-2 min (`muestra.wav`) que contenga una palabra buscable:
Run: `.\.venv\Scripts\speechtotext.exe find muestra.wav "<palabra>" --extract -m tiny`
Expected: imprime la región, crea `muestra_<tramo>.wav` y `muestra_<tramo>.txt` junto al audio, con el texto del tramo.

- [ ] **Step 4: Commit**

```bash
git add src/speechtotext/cli/app.py
git commit -m "feat(cli): find --extract recorta y transcribe la región elegida"
```

---

## Task 6: README + verificación end-to-end

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Documentar `find` en el README**

Añade una sección "Buscar un segmento" tras la de diarización:
- Explica el flujo: escaneo `tiny` cacheado + regiones; `--extract` para recortar+transcribir.
- Ejemplos:
  ```bash
  speechtotext find programa.mp3 "vulnerabilidad sísmica"
  speechtotext find programa.mp3 "vulnerabilidad sísmica" --extract --region 1
  speechtotext find programa.mp3 "entrevista" -e -D --speakers 4
  ```
- Nota: el primer `find` sobre un archivo construye el índice (lento una vez); luego es instantáneo. Índice en `~/.speechtotext/index/`, `--rebuild` lo fuerza.

- [ ] **Step 2: Suite completa**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS (todo verde; el build de índice con whisper queda como smoke manual).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: sección de find (buscador de segmento) en el README"
```

---

## Self-Review

**Cobertura del spec:**
- §3 comando/opciones → Tasks 4, 5 (todas las opciones declaradas). ✓
- §4 índice + caché → Task 2. ✓
- §5 coincidencia/regiones → Task 1 (normalize/search/cluster_regions). ✓
- §6 flujo --extract → Task 5 (clip_window + ffmpeg + transcribe_file). ✓
- §7 arquitectura (finder.py + refactor transcribe_file) → Tasks 1-2 (finder), Task 3 (refactor). ✓
- §8 errores (sin match, región fuera de rango, ffmpeg, caché corrupta) → Tasks 2, 4, 5. ✓
- §9 tests (normalize, search/cluster, índice, cli) → Tasks 1, 2, 4. ✓

**Placeholders:** el único stub es el `_extract_region` temporal de Task 4, explícitamente reemplazado en Task 5 Step 1 (patrón intencional para que el módulo importe entre tasks).

**Consistencia de tipos:** `Region(start,end,hits,matches,snippet)` igual en Tasks 1,4. `search(segments, query, gap, top)` y `load_or_build_index(audio, scan_model, rebuild) -> (segments, bool)` consistentes Tasks 1-2 ↔ 4. `clip_window(start,end,context) -> (begin,duration)` Task 1 ↔ 5. `transcribe_file(audio, output, language, model, formats, device, compute_type, vad, beam_size, diarize, speakers, identify, threshold)` idéntica Task 3 (definición) ↔ Task 5 (uso).
