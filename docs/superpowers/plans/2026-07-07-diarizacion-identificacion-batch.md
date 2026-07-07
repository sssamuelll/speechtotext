# Diarización + identificación de hablantes (batch) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que `speechtotext` marque quién dijo qué en una grabación de conversación (diarización offline con pyannote) y ponga nombre a las voces registradas (enrollment + identificación), reusando la transcripción faster-whisper actual.

**Architecture:** La transcripción existente queda intacta. Se añade el paquete `speakers/` (embeddings + registro de voces + identificación + diarización batch) y un tipo `LabeledSegment`. El CLI pasa a subcomandos. Las funciones puras (asignación por solape, coseno, registro) no importan torch/pyannote a nivel de módulo → son unit-testeables en el venv base; los modelos pesados se importan de forma perezosa dentro de las funciones que los usan.

**Tech Stack:** Python ≥3.10, faster-whisper (ya), pyannote.audio ≥3.1 + torch ≥2.0 (extra opcional `[diarize]`), numpy, typer, rich, pytest.

## Global Constraints

- Python ≥ 3.10. Verbatim del spec.
- **Offline:** sin subir audio a terceros. Todo modelo corre local.
- El `pip install -e .` base **sigue liviano**: pyannote/torch van SOLO en el extra `[diarize]`.
- Deps del extra, verbatim: `pyannote.audio>=3.1`, `numpy>=1.24`, `torch>=2.0`. `numpy>=1.24` pasa además a deps base (lo usan `identify`/`registry`).
- Umbral de identificación por defecto: **0.5** (coseno).
- Directorio de datos: `${SPEECHTOTEXT_HOME:-~/.speechtotext}`.
- Strings de cara al usuario en **español**, sin emoji (estilo del repo).
- **Regla de imports perezosos:** `identify.py`, `registry.py`, `core/segments.py`, `core/formats.py` y las funciones puras de `diarization.py` NO deben importar `torch`/`pyannote` a nivel de módulo. Los imports pesados van dentro de las funciones que los necesitan.
- Modelos gated de pyannote requieren `HF_TOKEN` y aceptar términos una vez.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `src/speechtotext/core/segments.py` | CREAR · dataclass `LabeledSegment`. |
| `src/speechtotext/core/formats.py` | MODIFICAR · writers renderizan hablante. |
| `src/speechtotext/speakers/__init__.py` | CREAR · vacío. |
| `src/speechtotext/speakers/identify.py` | CREAR · coseno + `assign_names` (numpy puro). |
| `src/speechtotext/speakers/registry.py` | CREAR · enrollment store (numpy + fs). |
| `src/speechtotext/speakers/diarization.py` | CREAR · `assign_segments`/`apply_names` (puras) + `diarize`/`cluster_embeddings` (pyannote perezoso). |
| `src/speechtotext/speakers/embedding.py` | CREAR · carga modelo + `embed` (pyannote perezoso). |
| `src/speechtotext/cli/app.py` | MODIFICAR · subcomandos `transcribe`/`enroll`/`voices`/`forget` + wiring. |
| `pyproject.toml` | MODIFICAR · numpy en base, extra `[diarize]` y `[dev]`. |
| `README.md` | MODIFICAR · instalación, subcomandos, ejemplos. |
| `tests/` | CREAR · unit tests de las partes puras. |

---

## Task 1: Scaffolding de tests + `LabeledSegment`

**Files:**
- Create: `src/speechtotext/core/segments.py`
- Create: `tests/test_segments.py`
- Modify: `pyproject.toml` (extra `[dev]` con pytest; numpy en base)

**Interfaces:**
- Produces: `LabeledSegment(start: float, end: float, text: str, speaker: str | None = None)`

- [ ] **Step 1: Añadir numpy a base y extras a `pyproject.toml`**

En `[project].dependencies` añadir `"numpy>=1.24"`. Tras `dependencies`, añadir:

```toml
[project.optional-dependencies]
diarize = [
    "pyannote.audio>=3.1",
    "numpy>=1.24",
    "torch>=2.0",
]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Instalar dev en el venv**

Run: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
Expected: instala pytest sin errores.

- [ ] **Step 3: Escribir el test que falla**

`tests/test_segments.py`:
```python
from speechtotext.core.segments import LabeledSegment


def test_labeled_segment_defaults_speaker_none():
    seg = LabeledSegment(start=0.0, end=1.5, text="hola")
    assert seg.speaker is None
    assert (seg.start, seg.end, seg.text) == (0.0, 1.5, "hola")


def test_labeled_segment_with_speaker():
    seg = LabeledSegment(0.0, 1.0, "hola", "Samuel")
    assert seg.speaker == "Samuel"
```

- [ ] **Step 4: Correr el test y verlo fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: speechtotext.core.segments`.

- [ ] **Step 5: Implementar `segments.py`**

```python
"""Segmento de transcripción con hablante opcional."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
```

- [ ] **Step 6: Correr el test y verlo pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_segments.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/speechtotext/core/segments.py tests/test_segments.py
git commit -m "feat(speakers): LabeledSegment + scaffolding de tests"
```

---

## Task 2: Writers renderizan hablante

**Files:**
- Modify: `src/speechtotext/core/formats.py`
- Create: `tests/test_formats_speaker.py`

**Interfaces:**
- Consumes: `LabeledSegment` (Task 1).
- Produces: `write_txt(segments, path)`, `write_srt(segments, path)`, `write_vtt(segments, path)`, `write_json(segments, info, path)` — ahora leen `getattr(seg, "speaker", None)`.

- [ ] **Step 1: Escribir tests que fallan**

`tests/test_formats_speaker.py`:
```python
import json
from types import SimpleNamespace

from speechtotext.core.segments import LabeledSegment
from speechtotext.core.formats import write_txt, write_srt, write_json


def test_txt_groups_consecutive_speaker(tmp_path):
    segs = [
        LabeledSegment(0, 1, "hola", "Samuel"),
        LabeledSegment(1, 2, "qué tal", "Samuel"),
        LabeledSegment(2, 3, "bien", "Ale"),
    ]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == "Samuel: hola qué tal\nAle: bien\n"


def test_txt_without_speaker_unchanged(tmp_path):
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(1, 2, "chao")]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == "hola\nchao\n"


def test_srt_prefixes_speaker(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Samuel")]
    p = tmp_path / "o.srt"
    write_srt(segs, p)
    assert "Samuel: hola" in p.read_text(encoding="utf-8")


def test_json_has_speaker_and_speakers(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Ale")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["speakers"] == ["Ale"]
    assert data["segments"][0]["speaker"] == "Ale"
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_formats_speaker.py -v`
Expected: FAIL (los writers aún no renderizan hablante).

- [ ] **Step 3: Modificar `formats.py`**

Reemplazar `write_txt`, `write_srt`, `write_vtt`, `write_json` por:

```python
def _speaker(seg):
    return getattr(seg, "speaker", None)


def write_txt(segments, path: Path) -> None:
    segs = list(segments)
    if any(_speaker(s) for s in segs):
        lines: list[str] = []
        cur: str | None = object()  # type: ignore[assignment]
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
        "speakers": speakers,
        "segments": [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                "speaker": _speaker(s),
            }
            for i, s in enumerate(seg_list)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Correr y ver pasar (incluye los tests viejos)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/core/formats.py tests/test_formats_speaker.py
git commit -m "feat(formats): renderizar hablante en txt/srt/vtt/json"
```

---

## Task 3: Identificación — coseno + `assign_names`

**Files:**
- Create: `src/speechtotext/speakers/__init__.py` (vacío)
- Create: `src/speechtotext/speakers/identify.py`
- Create: `tests/test_identify.py`

**Interfaces:**
- Produces:
  - `cosine(a: np.ndarray, b: np.ndarray) -> float`
  - `assign_names(clusters: dict[str, np.ndarray], enrolled: dict[str, np.ndarray], threshold: float) -> dict[str, str]` — devuelve solo los `speaker_id` que matchearon (los no matcheados quedan ausentes). Cada nombre se asigna a lo sumo a un `speaker_id` (el de mayor score).

- [ ] **Step 1: Escribir tests que fallan**

`tests/test_identify.py`:
```python
import numpy as np

from speechtotext.speakers.identify import cosine, assign_names


def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == 1.0


def test_assign_names_matches_closest_above_threshold():
    enrolled = {"Samuel": np.array([1.0, 0.0]), "Ale": np.array([0.0, 1.0])}
    clusters = {"SPEAKER_00": np.array([0.9, 0.1]), "SPEAKER_01": np.array([0.1, 0.9])}
    got = assign_names(clusters, enrolled, threshold=0.5)
    assert got == {"SPEAKER_00": "Samuel", "SPEAKER_01": "Ale"}


def test_assign_names_below_threshold_unmatched():
    enrolled = {"Samuel": np.array([1.0, 0.0])}
    clusters = {"SPEAKER_00": np.array([0.0, 1.0])}  # ortogonal → coseno 0
    assert assign_names(clusters, enrolled, threshold=0.5) == {}


def test_assign_names_no_collision_same_name():
    # dos clusters parecidos a Samuel: solo el mejor se lleva el nombre
    enrolled = {"Samuel": np.array([1.0, 0.0])}
    clusters = {"SPEAKER_00": np.array([1.0, 0.0]), "SPEAKER_01": np.array([0.8, 0.2])}
    got = assign_names(clusters, enrolled, threshold=0.5)
    assert got == {"SPEAKER_00": "Samuel"}
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_identify.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar `identify.py`**

```python
"""Identificación de hablantes: comparar embeddings contra voces registradas."""
from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def assign_names(
    clusters: dict[str, np.ndarray],
    enrolled: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, str]:
    """Asigna cada speaker_id anónimo a un nombre registrado (greedy por score)."""
    if not clusters or not enrolled:
        return {}
    candidates = [
        (cosine(vec, ref), sid, name)
        for sid, vec in clusters.items()
        for name, ref in enrolled.items()
    ]
    candidates.sort(reverse=True)  # mayor score primero
    result: dict[str, str] = {}
    used_names: set[str] = set()
    for score, sid, name in candidates:
        if score < threshold:
            break
        if sid in result or name in used_names:
            continue
        result[sid] = name
        used_names.add(name)
    return result
```

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_identify.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/speakers/__init__.py src/speechtotext/speakers/identify.py tests/test_identify.py
git commit -m "feat(speakers): identificación por coseno + assign_names"
```

---

## Task 4: Registro de voces (enrollment store)

**Files:**
- Create: `src/speechtotext/speakers/registry.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `home() -> Path` (respeta `SPEECHTOTEXT_HOME`)
  - `enroll(name: str, embedding: np.ndarray, *, seconds: float, model: str) -> None`
  - `list_voices() -> list[dict]` (claves: `name`, `seconds`, `model`, `enrolled_at`)
  - `get_embeddings() -> dict[str, np.ndarray]`
  - `remove(name: str) -> bool`

- [ ] **Step 1: Escribir tests que fallan**

`tests/test_registry.py`:
```python
import numpy as np

from speechtotext.speakers import registry


def test_enroll_list_get_remove_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    registry.enroll("Samuel", vec, seconds=12.0, model="pyannote/embedding")

    voices = registry.list_voices()
    assert [v["name"] for v in voices] == ["Samuel"]
    assert voices[0]["seconds"] == 12.0

    got = registry.get_embeddings()
    assert np.allclose(got["Samuel"], vec)

    assert registry.remove("Samuel") is True
    assert registry.list_voices() == []
    assert registry.remove("Samuel") is False


def test_home_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert registry.home() == tmp_path
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_registry.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar `registry.py`**

```python
"""Registro de voces para identificación: guarda un embedding por persona."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def home() -> Path:
    env = os.environ.get("SPEECHTOTEXT_HOME")
    return Path(env) if env else Path.home() / ".speechtotext"


def _voices_dir() -> Path:
    d = home() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return _voices_dir() / "manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_manifest(m: dict) -> None:
    _manifest_path().write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def enroll(name: str, embedding: np.ndarray, *, seconds: float, model: str) -> None:
    fname = f"{_slug(name)}.npy"
    np.save(_voices_dir() / fname, np.asarray(embedding, dtype=np.float32))
    m = _load_manifest()
    m[name] = {
        "file": fname,
        "seconds": round(float(seconds), 1),
        "model": model,
        "enrolled_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_manifest(m)


def list_voices() -> list[dict]:
    return [{"name": k, **v} for k, v in sorted(_load_manifest().items())]


def get_embeddings() -> dict[str, np.ndarray]:
    d = _voices_dir()
    return {name: np.load(d / meta["file"]) for name, meta in _load_manifest().items()}


def remove(name: str) -> bool:
    m = _load_manifest()
    if name not in m:
        return False
    (_voices_dir() / m[name]["file"]).unlink(missing_ok=True)
    del m[name]
    _save_manifest(m)
    return True
```

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_registry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/speakers/registry.py tests/test_registry.py
git commit -m "feat(speakers): registro de voces (enroll/list/get/remove)"
```

---

## Task 5: Asignación por solape + aplicar nombres (partes puras)

**Files:**
- Create: `src/speechtotext/speakers/diarization.py` (solo las funciones puras en esta task)
- Create: `tests/test_diarization_pure.py`

**Interfaces:**
- Consumes: `LabeledSegment` (Task 1).
- Produces:
  - `assign_segments(segments, turns: list[tuple[float, float, str]]) -> list[LabeledSegment]` — cada segmento se lleva el hablante con más solape; sin solape → `speaker=None`.
  - `humanize_speaker(speaker_id: str) -> str` — `"SPEAKER_00"` → `"Hablante 1"`.
  - `apply_names(labeled: list[LabeledSegment], name_map: dict[str, str]) -> list[LabeledSegment]` — reemplaza `speaker` por su nombre; si no está en `name_map`, lo humaniza; `None` se mantiene `None`.

- [ ] **Step 1: Escribir tests que fallan**

`tests/test_diarization_pure.py`:
```python
from types import SimpleNamespace

from speechtotext.speakers.diarization import (
    assign_segments,
    humanize_speaker,
    apply_names,
)


def _seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def test_assign_segment_max_overlap_wins():
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")]
    segs = [_seg(0.8, 2.5, "a caballo")]  # 0.2 con 00, 1.5 con 01 → gana 01
    out = assign_segments(segs, turns)
    assert out[0].speaker == "SPEAKER_01"
    assert out[0].text == "a caballo"


def test_assign_segment_no_overlap_is_none():
    turns = [(0.0, 1.0, "SPEAKER_00")]
    out = assign_segments([_seg(5.0, 6.0, "solo")], turns)
    assert out[0].speaker is None


def test_humanize_speaker():
    assert humanize_speaker("SPEAKER_00") == "Hablante 1"
    assert humanize_speaker("SPEAKER_01") == "Hablante 2"
    assert humanize_speaker("raro") == "raro"


def test_apply_names_maps_and_humanizes():
    from speechtotext.core.segments import LabeledSegment
    labeled = [
        LabeledSegment(0, 1, "hola", "SPEAKER_00"),
        LabeledSegment(1, 2, "chao", "SPEAKER_01"),
        LabeledSegment(2, 3, "...", None),
    ]
    out = apply_names(labeled, {"SPEAKER_00": "Samuel"})
    assert [s.speaker for s in out] == ["Samuel", "Hablante 2", None]
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_diarization_pure.py -v`
Expected: FAIL — módulo inexistente.

- [ ] **Step 3: Implementar las funciones puras en `diarization.py`**

```python
"""Diarización batch: asignación por solape (pura) + pyannote (perezoso)."""
from __future__ import annotations

from speechtotext.core.segments import LabeledSegment


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_segments(segments, turns: list[tuple[float, float, str]]) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for s in segments:
        best: str | None = None
        best_ov = 0.0
        for t0, t1, spk in turns:
            ov = _overlap(s.start, s.end, t0, t1)
            if ov > best_ov:
                best, best_ov = spk, ov
        out.append(LabeledSegment(start=s.start, end=s.end, text=s.text, speaker=best))
    return out


def humanize_speaker(speaker_id: str) -> str:
    try:
        n = int(speaker_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return speaker_id
    return f"Hablante {n + 1}"


def apply_names(
    labeled: list[LabeledSegment], name_map: dict[str, str]
) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for s in labeled:
        if s.speaker is None:
            spk: str | None = None
        else:
            spk = name_map.get(s.speaker) or humanize_speaker(s.speaker)
        out.append(LabeledSegment(s.start, s.end, s.text, spk))
    return out
```

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_diarization_pure.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speechtotext/speakers/diarization.py tests/test_diarization_pure.py
git commit -m "feat(speakers): asignación por solape + apply_names (puras)"
```

---

## Task 6: Modelos pyannote — `embedding.py` + `diarize`/`cluster_embeddings`

**Files:**
- Create: `src/speechtotext/speakers/embedding.py`
- Modify: `src/speechtotext/speakers/diarization.py` (añadir `diarize` y `cluster_embeddings`)
- Create: `tests/test_models_smoke.py` (integración, se salta sin `HF_TOKEN`)
- Create: `scripts/smoke_diarize.py` (verificación manual)

**Interfaces:**
- Produces:
  - `load_embedding_model()` → modelo de embeddings (objeto pyannote).
  - `embed(wav_path, model, region: tuple[float, float] | None = None) -> np.ndarray`
  - `diarize(wav_path, num_speakers: int | None = None) -> list[tuple[float, float, str]]`
  - `cluster_embeddings(wav_path, turns, model) -> dict[str, np.ndarray]`

**Nota de implementación:** la API exacta de pyannote (`Pipeline.from_pretrained`, `Inference`, `use_auth_token`) debe **verificarse contra la versión instalada** al implementar — pyannote cambia firmas entre menores. Los tests unitarios NO cubren esto (requiere modelos gated); la verificación es el smoke test manual del Step 5.

- [ ] **Step 1: Instalar el extra diarize**

Run: `.\.venv\Scripts\python.exe -m pip install -e ".[diarize]"`
Expected: instala pyannote.audio + torch (pesado; puede tardar).

- [ ] **Step 2: Implementar `embedding.py` (imports perezosos)**

```python
"""Embeddings de voz vía pyannote. Imports perezosos (torch/pyannote pesan)."""
from __future__ import annotations

import os

import numpy as np

_EMBEDDING_MODEL = "pyannote/embedding"


def load_embedding_model():
    from pyannote.audio import Model

    return Model.from_pretrained(_EMBEDDING_MODEL, use_auth_token=os.environ.get("HF_TOKEN"))


def embed(wav_path, model, region: tuple[float, float] | None = None) -> np.ndarray:
    from pyannote.audio import Inference
    from pyannote.core import Segment

    inference = Inference(model, window="whole")
    if region is not None:
        vec = inference.crop(str(wav_path), Segment(region[0], region[1]))
    else:
        vec = inference(str(wav_path))
    return np.asarray(vec, dtype=np.float32).reshape(-1)
```

- [ ] **Step 3: Añadir `diarize` y `cluster_embeddings` a `diarization.py`**

Añadir al final del archivo (imports perezosos dentro de las funciones):

```python
_DIARIZATION_PIPELINE = "pyannote/speaker-diarization-3.1"


def diarize(wav_path, num_speakers: int | None = None) -> list[tuple[float, float, str]]:
    import os

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        _DIARIZATION_PIPELINE, use_auth_token=os.environ.get("HF_TOKEN")
    )
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipeline(str(wav_path), **kwargs)
    return [
        (turn.start, turn.end, label)
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def cluster_embeddings(wav_path, turns, model) -> dict:
    from collections import defaultdict

    import numpy as np

    from speechtotext.speakers.embedding import embed

    by_speaker: dict[str, list] = defaultdict(list)
    for t0, t1, spk in turns:
        if t1 - t0 >= 0.5:  # ignora turnos muy cortos (embedding poco fiable)
            by_speaker[spk].append(embed(wav_path, model, region=(t0, t1)))
    return {spk: np.mean(vecs, axis=0) for spk, vecs in by_speaker.items() if vecs}
```

- [ ] **Step 4: Test de integración que se salta sin token**

`tests/test_models_smoke.py`:
```python
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"), reason="requiere HF_TOKEN y modelos gated"
)


def test_diarize_returns_turns(tmp_path):
    # Genera 2s de tono con ffmpeg y verifica que diarize no explota.
    import subprocess

    from speechtotext.speakers.diarization import diarize

    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=2",
         "-ar", "16000", "-ac", "1", str(wav)],
        check=True, capture_output=True,
    )
    turns = diarize(str(wav))
    assert isinstance(turns, list)
```

- [ ] **Step 5: Script de smoke manual**

`scripts/smoke_diarize.py`:
```python
"""Smoke manual: diariza un audio de dos voces. Uso: python scripts/smoke_diarize.py audio.wav"""
import sys

from speechtotext.speakers.diarization import diarize

turns = diarize(sys.argv[1])
for t0, t1, spk in turns:
    print(f"{t0:6.2f} - {t1:6.2f}  {spk}")
print(f"\n{len({s for _, _, s in turns})} hablantes detectados, {len(turns)} turnos")
```

Verificación manual (requiere `HF_TOKEN` y términos aceptados en las URLs que pyannote imprima):
Run: `.\.venv\Scripts\python.exe scripts/smoke_diarize.py <audio_de_dos_voces>.wav`
Expected: imprime turnos con 2 etiquetas SPEAKER distintas.

- [ ] **Step 6: Correr la suite (integración se salta sin token)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: unit PASS; `test_models_smoke` SKIPPED si no hay `HF_TOKEN`.

- [ ] **Step 7: Commit**

```bash
git add src/speechtotext/speakers/embedding.py src/speechtotext/speakers/diarization.py tests/test_models_smoke.py scripts/smoke_diarize.py
git commit -m "feat(speakers): diarización pyannote + embeddings (perezosos)"
```

---

## Task 7: CLI a subcomandos + wiring de diarización

**Files:**
- Modify: `src/speechtotext/cli/app.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: todo lo anterior + `faster_whisper.WhisperModel`.
- Produces: comandos `transcribe`, `enroll`, `voices`, `forget`.

- [ ] **Step 1: Escribir tests de CLI (con CliRunner y modelos mockeados)**

`tests/test_cli.py`:
```python
import numpy as np
from typer.testing import CliRunner

from speechtotext.cli.app import app
from speechtotext.speakers import registry

runner = CliRunner()


def test_voices_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["voices"])
    assert result.exit_code == 0
    assert "Sin voces" in result.stdout


def test_forget_missing_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["forget", "Nadie"])
    assert result.exit_code == 1


def test_voices_lists_enrolled(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model="m")
    result = runner.invoke(app, ["voices"])
    assert "Samuel" in result.stdout
```

- [ ] **Step 2: Correr y ver fallar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: FAIL (comandos aún no existen).

- [ ] **Step 3: Reescribir `app.py` a subcomandos**

Mantener el bloque UTF-8 y el `transcribe` existente como comando nombrado; añadir los nuevos. Estructura (el cuerpo de `transcribe` conserva su lógica actual de faster-whisper y añade el bloque de diarización tras obtener `segments`):

```python
# ... (cabecera de módulo, bloque UTF-8, imports existentes) ...
import typer
from rich.console import Console
from rich.table import Table

from speechtotext.core.formats import parse_formats, write_json, write_srt, write_txt, write_vtt

app = typer.Typer(add_completion=False, help="Transcripción y diarización offline con Whisper.")
console = Console()


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    # ... (todas las opciones actuales: output, language, model, formats, device, ...) ...
    diarize: bool = typer.Option(False, "--diarize", "-D", help="Marcar quién habla."),
    speakers: Optional[int] = typer.Option(None, "--speakers", help="Nº de hablantes (pista)."),
    identify: bool = typer.Option(True, "--identify/--no-identify", help="Poner nombre a voces registradas."),
    threshold: float = typer.Option(0.5, "--threshold", help="Umbral de match de voz (coseno)."),
) -> None:
    """Transcribe localmente; con --diarize marca los hablantes."""
    # ... lógica existente hasta obtener `segments` (list) e `info` ...

    if diarize:
        segments = _run_diarization(audio, segments, speakers, identify, threshold)

    # ... writers existentes (ya renderizan speaker si LabeledSegment lo trae) ...


def _run_diarization(audio, segments, speakers, identify, threshold):
    try:
        from speechtotext.speakers import diarization, registry
        from speechtotext.speakers.identify import assign_names
    except ImportError:
        console.print("[red]Falta el extra de diarización:[/red] pip install -e \".[diarize]\"")
        raise typer.Exit(1)

    try:
        turns = diarization.diarize(str(audio), num_speakers=speakers)
    except Exception as e:  # auth/model gated
        console.print(f"[red]Diarización falló:[/red] {e}")
        console.print("Acepta los términos del modelo en huggingface.co y exporta HF_TOKEN.")
        raise typer.Exit(1)

    labeled = diarization.assign_segments(segments, turns)
    name_map = {}
    if identify:
        enrolled = registry.get_embeddings()
        if enrolled:
            from speechtotext.speakers.embedding import load_embedding_model
            model = load_embedding_model()
            clusters = diarization.cluster_embeddings(str(audio), turns, model)
            name_map = assign_names(clusters, enrolled, threshold)
    return diarization.apply_names(labeled, name_map)


@app.command()
def enroll(name: str = typer.Argument(...), sample: Path = typer.Argument(..., exists=True)) -> None:
    """Registra la voz de una persona desde una muestra de audio."""
    from speechtotext.core.audio import transcode_to_wav
    from speechtotext.speakers import registry
    from speechtotext.speakers.embedding import embed, load_embedding_model
    import wave, contextlib

    wav = transcode_to_wav(sample.read_bytes())
    try:
        with contextlib.closing(wave.open(str(wav))) as w:
            seconds = w.getnframes() / float(w.getframerate())
        if seconds < 10:
            console.print(f"[yellow]Aviso:[/yellow] muestra corta ({seconds:.0f}s); ≥10s es más fiable.")
        model = load_embedding_model()
        vec = embed(str(wav), model)
    finally:
        wav.unlink(missing_ok=True)
    registry.enroll(name, vec, seconds=seconds, model="pyannote/embedding")
    console.print(f"  [green]OK[/green] voz de {name} registrada.")


@app.command()
def voices() -> None:
    """Lista las voces registradas."""
    from speechtotext.speakers import registry
    vs = registry.list_voices()
    if not vs:
        console.print("Sin voces registradas. Usa: speechtotext enroll <nombre> <muestra.wav>")
        return
    table = Table("Nombre", "Segundos", "Registrada")
    for v in vs:
        table.add_row(v["name"], str(v["seconds"]), v.get("enrolled_at", ""))
    console.print(table)


@app.command()
def forget(name: str = typer.Argument(...)) -> None:
    """Borra una voz registrada."""
    from speechtotext.speakers import registry
    if registry.remove(name):
        console.print(f"  [green]OK[/green] {name} borrada.")
    else:
        console.print(f"[red]No existe una voz llamada {name}.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

**Importante:** copiar el cuerpo real del `transcribe` actual (no reescribirlo de memoria) y solo (a) convertir el iterador `segments` a `list` antes del bloque de diarización, y (b) insertar la llamada a `_run_diarization`.

- [ ] **Step 4: Correr y ver pasar**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Verificar que la transcripción normal sigue viva**

Run: `.\.venv\Scripts\speechtotext.exe transcribe <un_audio>.wav -m tiny`
Expected: transcribe como antes (ahora bajo el subcomando `transcribe`).

- [ ] **Step 6: Commit**

```bash
git add src/speechtotext/cli/app.py tests/test_cli.py
git commit -m "feat(cli): subcomandos transcribe/enroll/voices/forget + wiring de diarización"
```

---

## Task 8: README + verificación end-to-end

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Actualizar `README.md`**

- Instalación: añadir `pip install -e ".[diarize]"` para diarización.
- Cambiar todos los ejemplos `speechtotext audio.wav` → `speechtotext transcribe audio.wav`.
- Nueva sección "Diarización e identificación" con:
  ```bash
  speechtotext enroll "Samuel" muestra_samuel.wav
  speechtotext transcribe conversacion.mp3 --diarize --speakers 2
  speechtotext voices
  ```
- Documentar `HF_TOKEN` + aceptar términos de los modelos pyannote (URLs).
- Nota de límite: etiqueta a nivel de segmento.

- [ ] **Step 2: Smoke end-to-end manual (con HF_TOKEN)**

Con un audio real de dos personas y una voz enrolada:
Run: `.\.venv\Scripts\speechtotext.exe transcribe conversacion.wav --diarize --speakers 2 -f txt`
Expected: el `.txt` muestra turnos prefijados por hablante (nombre si enrolado, si no `Hablante 1/2`).

- [ ] **Step 3: Suite completa**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: unit PASS; integración SKIPPED sin token.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: diarización, identificación y subcomandos en el README"
```

---

## Self-Review

**Cobertura del spec:**
- §3 estructura → Tasks 1,3,4,5,6,7. ✓
- §4 componentes (segments/embedding/registry/identify/diarization) → Tasks 1,6,4,3,5,6. ✓
- §5 flujo de datos → Task 7 (`_run_diarization`). ✓
- §6 CLI subcomandos + flags → Task 7. ✓
- §7 enrollment (enroll/voices/forget, aviso <10s, SPEECHTOTEXT_HOME) → Tasks 4,7. ✓
- §8 salida txt/srt/vtt/json → Task 2. ✓
- §9 errores (extra faltante, token, muestra corta, sin voces, N≠2) → Tasks 6,7. ✓
- §10 deps `[diarize]` + numpy base → Task 1. ✓
- §11 testing (assign_segments, identify, registry, writers) → Tasks 2,3,4,5. ✓
- §12 límites → documentados en README (Task 8). ✓

**Placeholders:** los únicos "..." están en Task 7 Step 3, donde se indica explícitamente copiar el cuerpo real del `transcribe` existente (no reescribirlo) — es una instrucción, no un placeholder de lógica nueva.

**Consistencia de tipos:** `LabeledSegment(start,end,text,speaker)` idéntico en Tasks 1,2,5. `assign_names(clusters, enrolled, threshold) -> dict[str,str]` consistente Task 3 ↔ 7. `turns: list[tuple[float,float,str]]` consistente Tasks 5,6,7. `get_embeddings() -> dict[str,np.ndarray]` consistente Tasks 4,7.
