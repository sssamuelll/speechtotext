# Aurelius — plan de implementación del esqueleto andante

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un asistente de voz 100% local con loop completo — wake word → escucha → transcribe → verifica que es Samuel → piensa (LLM local) → habla — con bus de audio fullband 48 kHz/10 ms y costura DSP enchufable, medido con eventos JSONL.

**Architecture:** Un solo proceso Python: callback de captura (sounddevice/WASAPI, 48 kHz) → cola → thread pipeline (re-partición a frames de 10 ms → cadena DSP → decimador con estado a 16 kHz → wake word + VAD) → FSM orquestadora → workers de ASR/gate/LLM → TTS Piper por oración → playback con ring de referencia far-end. Dos subprocesos externos: `llama-server` y `piper`. Spec: `docs/superpowers/specs/2026-07-13-jarvis-voice-core-design.md`.

**Tech Stack:** Python 3.11+, numpy, scipy, sounddevice, pysilero-vad, openwakeword, faster-whisper, sherpa-onnx, requests, typer, pytest. Externos: llama.cpp (`llama-server`), Piper (binario GPL aislado), Qwen2.5-7B-Instruct Q4_K_M.

## Global Constraints

- Repo nuevo: `D:\Desktop\projects\aurelius`, público, licencia **MIT**. Paquete `aurelius`, CLI `aurelius`.
- `requires-python = ">=3.11"` (tomllib en stdlib). Todo corre en **CPU** (la GTX 980 no existe).
- Contrato del bus: **mono float32, 48 000 Hz, frames de 480 muestras (10 ms)**. Vista ML: 16 000 Hz (frames de 160).
- El callback de audio **jamás** bloquea, aloca de más ni ejecuta inferencia: solo copia y `put_nowait`.
- Colas acotadas con contador de drops visible. Threads de inferencia limitados: faster-whisper `cpu_threads=8`, llama-server `-t 9`.
- `llama-server` siempre con `cache_prompt: true` en cada request.
- Piper es GPL-3.0: **subprocess externo por oración, jamás importado ni vendorizado**. Modelos: descargados por script, jamás commiteados.
- pyannote NO se usa en aurelius (queda en el batch de speechtotext). Identidad de voz: sherpa-onnx + `speechtotext.speakers.registry`.
- Eventos JSONL con `time.monotonic()`: 5 fronteras por turno (`speech_end`, `asr_done`, `first_token`, `first_sentence`, `first_audio`) + `xrun`, `drop`, `gate`.
- Commits en español, convención `tipo(módulo): descripción` (como speechtotext). Docs en español, identificadores en inglés.
- Latencia objetivo (§10 del spec): fin de habla → primera sílaba 3.5-5 s p50 con cache caliente.

## Estructura de archivos (se crea en las tareas)

```
D:\Desktop\projects\aurelius\
├── pyproject.toml
├── LICENSE                       MIT
├── README.md                     stub en Task 1; vitrina en Task 18
├── docs\setup-modelos.md         descargas exactas (Task 1)
├── aurelius.toml.example         config (Task 14)
├── src\aurelius\
│   ├── __init__.py
│   ├── events.py                 EventLog JSONL            (Task 2)
│   ├── config.py                 Config TOML               (Task 14)
│   ├── audio\
│   │   ├── __init__.py
│   │   ├── frames.py             Reframer                  (Task 3)
│   │   ├── decimate.py           Decimator48to16           (Task 5)
│   │   ├── farend.py             FarEndRing                (Task 6)
│   │   ├── capture.py            Capture                   (Task 8)
│   │   └── playback.py           Player                    (Task 8)
│   ├── dsp\
│   │   ├── __init__.py
│   │   ├── chain.py              Stage (Protocol) + Chain  (Task 4)
│   │   └── stages.py             Passthrough, HighpassBiquad (Task 4)
│   ├── ears\
│   │   ├── __init__.py
│   │   ├── vad.py                UtteranceSegmenter + make_silero_vad (Task 7)
│   │   ├── wake.py               WakeDetector              (Task 9)
│   │   ├── asr.py                Transcriber               (Task 10)
│   │   └── gate.py               VoiceGate                 (Task 11)
│   ├── brain\
│   │   ├── __init__.py
│   │   └── llm.py                Brain                     (Task 12)
│   ├── voice\
│   │   ├── __init__.py
│   │   ├── split.py              SentenceSplitter          (Task 15)
│   │   └── tts.py                PiperTTS                  (Task 16)
│   ├── fsm.py                    FSM pura                  (Task 13)
│   └── cli.py                    typer: devices/listen/enroll/run/bench (Tasks 8, 14, 17, 18)
└── tests\
    ├── test_events.py, test_frames.py, test_chain.py, test_decimate.py,
    ├── test_farend.py, test_vad.py, test_gate.py, test_brain.py,
    ├── test_fsm.py, test_split.py, test_tts.py
    └── fixtures\                 wavs opcionales grabados a mano
```

---

# FASE S1 — el camino del audio

### Task 1: Bootstrap del repo aurelius

**Files:**
- Create: `D:\Desktop\projects\aurelius\pyproject.toml`
- Create: `D:\Desktop\projects\aurelius\LICENSE` (MIT, año 2026, Samuel Ballesteros)
- Create: `D:\Desktop\projects\aurelius\README.md`
- Create: `D:\Desktop\projects\aurelius\.gitignore`
- Create: `D:\Desktop\projects\aurelius\docs\setup-modelos.md`
- Create: `D:\Desktop\projects\aurelius\src\aurelius\__init__.py` (y los `__init__.py` de `audio/`, `dsp/`, `ears/`, `brain/`, `voice/`, todos vacíos)
- Create: `D:\Desktop\projects\aurelius\tests\test_smoke.py`

**Interfaces:**
- Produces: paquete `aurelius` instalable editable; entry point CLI `aurelius` (apunta a `aurelius.cli:app`, que existe desde Task 8 — el entry point se declara ya pero no se usa hasta entonces).

- [ ] **Step 1: Crear repo y venv**

```powershell
mkdir D:\Desktop\projects\aurelius; cd D:\Desktop\projects\aurelius
git init -b main
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

- [ ] **Step 2: Escribir pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "aurelius"
version = "0.1.0"
description = "Asistente de voz 100% local: wake word, identidad de voz, DSP en vivo, LLM y TTS en tu máquina."
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "sounddevice>=0.4.6",
    "pysilero-vad>=2.0",
    "openwakeword>=0.6",
    "faster-whisper>=1.0",
    "sherpa-onnx>=1.10",
    "requests>=2.31",
    "typer>=0.9",
    "speechtotext @ git+https://github.com/sssamuelll/speechtotext",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
aurelius = "aurelius.cli:app"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Escribir .gitignore y README stub**

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
models/
*.onnx
*.gguf
eventos*.jsonl
```

`README.md`:

```markdown
# aurelius

Asistente de voz 100% local. Tu audio nunca sale de tu máquina.

En construcción — esqueleto andante. Spec y plan en el repo speechtotext
(docs/superpowers/). Setup de modelos: docs/setup-modelos.md.
```

- [ ] **Step 4: Escribir docs\setup-modelos.md**

```markdown
# Setup de modelos y binarios (una sola vez)

Todo va en `models/` (ignorado por git). Verifica cada URL contra la página
de releases si falla — las versiones rotan.

## llama.cpp (cerebro)
1. Baja el zip `llama-*-bin-win-cpu-x64.zip` más reciente de
   https://github.com/ggml-org/llama.cpp/releases y descomprime en `models\llama\`.
2. GGUF oficial (≈4.7 GB):
   `pip install huggingface-hub` y luego
   `hf download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models`
3. Arranque (así lo espera `aurelius run`):
   `models\llama\llama-server.exe -m models\qwen2.5-7b-instruct-q4_k_m.gguf -t 9 -c 4096 --port 8080`

## Piper (voz — binario GPL, externo)
1. Baja `piper_windows_amd64.zip` de https://github.com/rhasspy/piper/releases
   (release 2023.11.14-2) y descomprime en `models\piper\`.
2. Voz es_MX (verifica licencia en su MODEL_CARD antes de citarla en el README):
   descarga `es_MX-claude-high.onnx` y `es_MX-claude-high.onnx.json` desde
   https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_MX/claude/high
   a `models\voces\`.

## openWakeWord (wake word)
`python -c "import openwakeword.utils; openwakeword.utils.download_models()"`
(baja los modelos base + hey_jarvis a la caché del paquete).

## sherpa-onnx (identidad de voz)
Baja `wespeaker_en_voxceleb_resnet34_LM.onnx` del release
https://github.com/k2-fsa/sherpa-onnx/releases (tag `speaker-recongition-models`,
el typo es del tag real) a `models\`.

## faster-whisper y Silero VAD
Se descargan solos en el primer uso (caché de HF / paquete pysilero-vad).
```

- [ ] **Step 5: Crear los `__init__.py` y el smoke test**

`src\aurelius\__init__.py`: `__version__ = "0.1.0"` — los demás `__init__.py` vacíos.

`tests\test_smoke.py`:

```python
def test_importa():
    import aurelius
    assert aurelius.__version__ == "0.1.0"
```

- [ ] **Step 6: Instalar y verificar**

Run: `pip install -e ".[dev]"` y luego `pytest -q`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```powershell
git add -A; git commit -m "feat: bootstrap del paquete aurelius (pyproject, setup de modelos, smoke test)"
```

---

### Task 2: EventLog — instrumentación JSONL

**Files:**
- Create: `src\aurelius\events.py`
- Test: `tests\test_events.py`

**Interfaces:**
- Produces: `EventLog(path).emit(kind: str, **fields) -> None`, `EventLog.close()`. Cada línea: JSON con `ts` (`time.monotonic()`, float) y `kind`, más los fields. Thread-safe (lock).

- [ ] **Step 1: Test que falla**

```python
import json
from aurelius.events import EventLog

def test_emite_jsonl(tmp_path):
    p = tmp_path / "ev.jsonl"
    log = EventLog(p)
    log.emit("speech_end", turn=1)
    log.emit("xrun")
    log.close()
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [l["kind"] for l in lines] == ["speech_end", "xrun"]
    assert lines[0]["turn"] == 1
    assert isinstance(lines[0]["ts"], float)
    assert lines[1]["ts"] >= lines[0]["ts"]
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_events.py -q` → `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Implementación mínima**

```python
"""Eventos JSONL: la instrumentación es ciudadana de primera clase (spec §9)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: str | Path):
        self._f = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields) -> None:
        line = json.dumps({"ts": time.monotonic(), "kind": kind, **fields}, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.close()
```

- [ ] **Step 4: Verificar que pasa** — Run: `pytest tests/test_events.py -q` → `1 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(events): EventLog JSONL thread-safe con reloj monotónico"`

---

### Task 3: Reframer — de bloques del device a frames de 10 ms

**Files:**
- Create: `src\aurelius\audio\frames.py`
- Test: `tests\test_frames.py`

**Interfaces:**
- Produces: `Reframer(frame_size: int).push(block: np.ndarray) -> list[np.ndarray]` — devuelve todos los frames completos de `frame_size` muestras float32; el sobrante queda buffereado para el próximo push. Sin pérdida ni duplicación.

- [ ] **Step 1: Test que falla**

```python
import numpy as np
from aurelius.audio.frames import Reframer

def test_reparticion_sin_perdida():
    r = Reframer(480)
    src = np.arange(480 * 3 + 100, dtype=np.float32)  # 3 frames + resto
    out = []
    for start in range(0, len(src), 333):              # bloques que no dividen exacto
        out.extend(r.push(src[start:start + 333]))
    got = np.concatenate(out)
    assert all(len(f) == 480 for f in out)
    assert np.array_equal(got, src[: len(got)])
    assert len(got) == 480 * 3                          # el resto sigue buffereado

def test_bloque_gigante():
    r = Reframer(480)
    assert len(r.push(np.zeros(480 * 5, dtype=np.float32))) == 5
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_frames.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Re-partición: bloques de tamaño arbitrario del device → frames exactos del bus."""
from __future__ import annotations

import numpy as np


class Reframer:
    def __init__(self, frame_size: int):
        self.frame_size = frame_size
        self._rest = np.empty(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._rest, np.asarray(block, dtype=np.float32).reshape(-1)])
        n = len(buf) // self.frame_size
        self._rest = buf[n * self.frame_size:]
        return [buf[i * self.frame_size:(i + 1) * self.frame_size] for i in range(n)]
```

- [ ] **Step 4: Verificar que pasa** — Run: `pytest tests/test_frames.py -q` → `2 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(audio): Reframer de bloques del device a frames de 10 ms"`

---

### Task 4: Cadena DSP — Stage, Chain, Passthrough y HighpassBiquad

**Files:**
- Create: `src\aurelius\dsp\chain.py`
- Create: `src\aurelius\dsp\stages.py`
- Test: `tests\test_chain.py`

**Interfaces:**
- Produces (contrato público de la fase 2, spec §6):
  - `Stage` (Protocol): `process(frame: np.ndarray) -> np.ndarray` (float32, misma longitud) y `reset() -> None`.
  - `Chain(stages: list[Stage])`: `process(frame) -> np.ndarray`, `reset()`, atributo `last_ms: float` (duración del último process, para el presupuesto de 10 ms).
  - `Passthrough()`, `HighpassBiquad(cutoff_hz: float = 80.0, rate: int = 48000)`.

- [ ] **Step 1: Tests que fallan**

```python
import numpy as np
from scipy.signal import butter, sosfilt
from aurelius.dsp.chain import Chain
from aurelius.dsp.stages import HighpassBiquad, Passthrough

def test_passthrough_identidad():
    f = np.random.default_rng(0).standard_normal(480).astype(np.float32)
    out = Chain([Passthrough()]).process(f)
    assert np.array_equal(out, f)

def test_highpass_estado_sobrevive_entre_frames():
    """Filtrar frame a frame == filtrar todo de una vez. Si el estado se pierde, difiere."""
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(480 * 50).astype(np.float32)
    hp = HighpassBiquad(80.0, 48000)
    por_frames = np.concatenate([hp.process(sig[i:i + 480]) for i in range(0, len(sig), 480)])
    sos = butter(2, 80.0, "highpass", fs=48000, output="sos")
    entero = sosfilt(sos, sig)
    assert np.allclose(por_frames, entero, atol=1e-5)

def test_chain_mide_tiempo():
    c = Chain([Passthrough()])
    c.process(np.zeros(480, dtype=np.float32))
    assert c.last_ms >= 0.0
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_chain.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

`src\aurelius\dsp\chain.py`:

```python
"""La costura DSP (spec §6): frames float32 de 10 ms, etapas enchufables con estado."""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class Stage(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...


class Chain:
    def __init__(self, stages: list[Stage]):
        self.stages = stages
        self.last_ms: float = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        for s in self.stages:
            frame = s.process(frame)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return frame

    def reset(self) -> None:
        for s in self.stages:
            s.reset()
```

`src\aurelius\dsp\stages.py`:

```python
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class Passthrough:
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def reset(self) -> None:
        pass


class HighpassBiquad:
    """Highpass de 2º orden. Prueba viviente de que el estado entre frames funciona."""

    def __init__(self, cutoff_hz: float = 80.0, rate: int = 48000):
        self._sos = butter(2, cutoff_hz, "highpass", fs=rate, output="sos")
        self.reset()

    def process(self, frame: np.ndarray) -> np.ndarray:
        out, self._zi = sosfilt(self._sos, frame, zi=self._zi)
        return out.astype(np.float32)

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos) * 0.0
```

- [ ] **Step 4: Verificar que pasan** — Run: `pytest tests/test_chain.py -q` → `3 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(dsp): contrato Stage + Chain con timer y highpass con estado persistente"`

---

### Task 5: Decimator48to16 — la vista ML, con estado

**Files:**
- Create: `src\aurelius\audio\decimate.py`
- Test: `tests\test_decimate.py`

**Interfaces:**
- Produces: `Decimator48to16()`: `process(frame480: np.ndarray) -> np.ndarray` (160 muestras float32 a 16 kHz), `reset()`. Anti-alias lowpass (butter 8º orden, corte 7 200 Hz) con `zi` persistente + tomar 1 de cada 3. **Nunca** `resample_poly` por frame (stateless: corrompe las fronteras — hallazgo de la auditoría).

- [ ] **Step 1: Test que falla**

```python
import numpy as np
from scipy.signal import butter, sosfilt
from aurelius.audio.decimate import Decimator48to16

def test_frame_a_frame_igual_que_offline():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(480 * 100).astype(np.float32)
    d = Decimator48to16()
    por_frames = np.concatenate([d.process(sig[i:i + 480]) for i in range(0, len(sig), 480)])
    sos = butter(8, 7200.0, "lowpass", fs=48000, output="sos")
    offline = sosfilt(sos, sig)[::3]
    assert len(por_frames) == 160 * 100
    assert np.allclose(por_frames, offline, atol=1e-4)
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_decimate.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Decimador 48→16 kHz CON estado: la vista de los órganos ML (spec §4)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class Decimator48to16:
    def __init__(self):
        self._sos = butter(8, 7200.0, "lowpass", fs=48000, output="sos")
        self.reset()

    def process(self, frame480: np.ndarray) -> np.ndarray:
        out, self._zi = sosfilt(self._sos, frame480, zi=self._zi)
        return out[::3].astype(np.float32)

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos) * 0.0
```

- [ ] **Step 4: Verificar que pasa** — Run: `pytest tests/test_decimate.py -q` → `1 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(audio): decimador 48→16 con estado persistente entre frames"`

---

### Task 6: FarEndRing — la semilla del AEC

**Files:**
- Create: `src\aurelius\audio\farend.py`
- Test: `tests\test_farend.py`

**Interfaces:**
- Produces: `FarEndRing(seconds: float = 2.0, rate: int = 48000)`: `write(frame: np.ndarray) -> None` (lo llama el callback de playback), `snapshot(n: int) -> np.ndarray` (últimas n muestras, en orden). Lock interno (escrituras de ~10 ms, contención despreciable). La fase AEC (F2) le añadirá timestamps de device — hoy no.

- [ ] **Step 1: Test que falla**

```python
import numpy as np
from aurelius.audio.farend import FarEndRing

def test_snapshot_ultimas_muestras_con_wraparound():
    r = FarEndRing(seconds=0.01, rate=48000)  # capacidad: 480 muestras
    r.write(np.arange(300, dtype=np.float32))
    r.write(np.arange(300, 600, dtype=np.float32))  # fuerza wraparound
    got = r.snapshot(480)
    assert np.array_equal(got, np.arange(120, 600, dtype=np.float32))

def test_snapshot_mayor_que_lo_escrito_rellena_con_ceros():
    r = FarEndRing(seconds=0.01)
    r.write(np.ones(100, dtype=np.float32))
    got = r.snapshot(480)
    assert len(got) == 480 and got[-100:].sum() == 100 and got[:380].sum() == 0
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_farend.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Ring de referencia far-end: lo que sonó por los parlantes (para el AEC de F2)."""
from __future__ import annotations

import threading

import numpy as np


class FarEndRing:
    def __init__(self, seconds: float = 2.0, rate: int = 48000):
        self._buf = np.zeros(int(seconds * rate), dtype=np.float32)
        self._pos = 0
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        with self._lock:
            n = len(self._buf)
            idx = (self._pos + np.arange(len(frame))) % n
            self._buf[idx] = frame
            self._pos = (self._pos + len(frame)) % n

    def snapshot(self, n: int) -> np.ndarray:
        with self._lock:
            idx = (self._pos - n + np.arange(n)) % len(self._buf)
            return self._buf[idx].copy()
```

- [ ] **Step 4: Verificar que pasa** — Run: `pytest tests/test_farend.py -q` → `2 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(audio): FarEndRing de referencia para el AEC de fase 2"`

---

### Task 7: UtteranceSegmenter — VAD y cierre de utterance

**Files:**
- Create: `src\aurelius\ears\vad.py`
- Test: `tests\test_vad.py`

**Interfaces:**
- Consumes: frames de 160 muestras (vista 16 kHz del Decimator).
- Produces:
  - `Utterance` (dataclass): `audio: np.ndarray` (float32 16 kHz), `started_at: float`, `ended_at: float`.
  - `UtteranceSegmenter(vad_prob, *, threshold=0.5, silence_ms=400, max_s=30.0, pre_roll_ms=200)`: `push(frame160: np.ndarray, now: float) -> Utterance | None`, `reset()`, propiedad `speaking: bool`. `vad_prob: Callable[[np.ndarray], float]` recibe chunks de **512 muestras** (requisito de Silero) — inyectable para testear sin modelo.
  - `make_silero_vad() -> Callable[[np.ndarray], float]` — el vad_prob real (pysilero-vad, sin torch).

- [ ] **Step 1: Tests que fallan (lógica pura, VAD falso)**

```python
import numpy as np
from aurelius.ears.vad import UtteranceSegmenter

def _seg(probs):
    """vad_prob falso: devuelve probs en orden, luego 0."""
    it = iter(probs)
    return UtteranceSegmenter(lambda _c: next(it, 0.0), threshold=0.5,
                              silence_ms=64, pre_roll_ms=32)  # 64ms = 2 chunks de silencio

def test_emite_utterance_tras_silencio():
    # 4 chunks de voz (prob .9) + 3 de silencio (prob .1) → 1 utterance
    seg = _seg([0.9] * 4 + [0.1] * 3)
    got = None
    for i in range(7):
        for _ in range(512 // 160 + 1):  # frames de 160 hasta completar cada chunk
            u = seg.push(np.full(160, 0.5, dtype=np.float32), now=float(i))
            got = u or got
    assert got is not None
    assert got.ended_at > got.started_at
    assert len(got.audio) >= 4 * 512          # al menos la voz + pre-roll

def test_sin_voz_no_emite():
    seg = _seg([0.1] * 10)
    for i in range(10):
        for _ in range(4):
            assert seg.push(np.zeros(160, dtype=np.float32), now=float(i)) is None
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_vad.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Segmentador de utterances sobre VAD (Silero v5 vía pysilero-vad, chunks de 512 @16k)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

CHUNK = 512          # muestras por chunk de VAD (32 ms @ 16 kHz)
RATE = 16000


@dataclass
class Utterance:
    audio: np.ndarray
    started_at: float
    ended_at: float


class UtteranceSegmenter:
    def __init__(self, vad_prob: Callable[[np.ndarray], float], *,
                 threshold: float = 0.5, silence_ms: int = 400,
                 max_s: float = 30.0, pre_roll_ms: int = 200):
        self._vad = vad_prob
        self._th = threshold
        self._silence_chunks = max(1, int(silence_ms / 1000 * RATE / CHUNK))
        self._max_chunks = int(max_s * RATE / CHUNK)
        self._pre = deque(maxlen=max(1, int(pre_roll_ms / 1000 * RATE / CHUNK)))
        self._buf = np.empty(0, dtype=np.float32)   # acumulador hasta CHUNK
        self._voiced: list[np.ndarray] = []
        self._quiet = 0
        self._t0 = 0.0
        self.speaking = False

    def push(self, frame160: np.ndarray, now: float) -> Utterance | None:
        self._buf = np.concatenate([self._buf, frame160.astype(np.float32)])
        out: Utterance | None = None
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            out = self._chunk(chunk, now) or out
        return out

    def _chunk(self, chunk: np.ndarray, now: float) -> Utterance | None:
        voiced = self._vad(chunk) >= self._th
        if not self.speaking:
            if voiced:
                self.speaking = True
                self._voiced = list(self._pre) + [chunk]
                self._quiet = 0
                self._t0 = now
            else:
                self._pre.append(chunk)
            return None
        self._voiced.append(chunk)
        self._quiet = 0 if voiced else self._quiet + 1
        if self._quiet >= self._silence_chunks or len(self._voiced) >= self._max_chunks:
            u = Utterance(np.concatenate(self._voiced), self._t0, now)
            self.reset()
            return u
        return None

    def reset(self) -> None:
        self.speaking = False
        self._voiced = []
        self._quiet = 0
        self._pre.clear()
        self._buf = np.empty(0, dtype=np.float32)


def make_silero_vad() -> Callable[[np.ndarray], float]:
    from pysilero_vad import SileroVoiceActivityDetector
    det = SileroVoiceActivityDetector()

    def prob(chunk512: np.ndarray) -> float:
        pcm = (np.clip(chunk512, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return float(det(pcm))

    return prob
```

- [ ] **Step 4: Verificar que pasan** — Run: `pytest tests/test_vad.py -q` → `2 passed`.

- [ ] **Step 5: Smoke del VAD real** — Run: `python -c "from aurelius.ears.vad import make_silero_vad; import numpy as np; p=make_silero_vad(); print(p(np.zeros(512,dtype=np.float32)))"`
Expected: un float cercano a 0.0 (silencio). Si truena por modelo faltante: `pip show pysilero-vad` y reinstalar.

- [ ] **Step 6: Commit** — `git add -A; git commit -m "feat(ears): segmentador de utterances con VAD inyectable + Silero real"`

---

### Task 8: Capture, Player y el CLI `listen` — S1 completo

**Files:**
- Create: `src\aurelius\audio\capture.py`
- Create: `src\aurelius\audio\playback.py`
- Create: `src\aurelius\cli.py`

**Interfaces:**
- Produces:
  - `Capture(device: int | None = None, rate: int = 48000, blocksize: int = 1440)`: atributos `queue` (`queue.SimpleQueue` de bloques float32 mono), `xruns: int`, `dropped: int`; métodos `start()`, `stop()`. Backpressure: si `queue.qsize() > 32`, descarta el bloque e incrementa `dropped`.
  - `Player(farend: FarEndRing, device: int | None = None, rate: int = 48000)`: `play(pcm: np.ndarray)` (encola sin bloquear), `flush()` (barge-in), `idle() -> bool`, `start()`, `stop()`. El callback escribe TODO lo que suena al `farend`.
  - CLI typer `app` con comandos `devices` y `listen`.
- Consumes: `Reframer`, `Chain`, `Passthrough`, `HighpassBiquad`, `Decimator48to16`, `UtteranceSegmenter`, `make_silero_vad`, `EventLog`, `FarEndRing`.

- [ ] **Step 1: Implementar Capture**

```python
"""Captura WASAPI: el callback solo copia y suelta. Jamás bloquea (spec §4)."""
from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd


class Capture:
    def __init__(self, device: int | None = None, rate: int = 48000, blocksize: int = 1440):
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.xruns = 0
        self.dropped = 0
        self._stream = sd.InputStream(
            device=device, samplerate=rate, blocksize=blocksize,
            channels=1, dtype="float32", callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self.xruns += 1
        if self.queue.qsize() > 32:      # ponytail: backpressure por descarte, no por bloqueo
            self.dropped += 1
            return
        self.queue.put_nowait(indata[:, 0].copy())

    def start(self) -> None: self._stream.start()
    def stop(self) -> None: self._stream.stop(); self._stream.close()
```

- [ ] **Step 2: Implementar Player**

```python
"""Playback con cola de PCM y copia al ring far-end (la semilla del AEC)."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np
import sounddevice as sd

from aurelius.audio.farend import FarEndRing


class Player:
    def __init__(self, farend: FarEndRing, device: int | None = None, rate: int = 48000):
        self._farend = farend
        self._q: deque[np.ndarray] = deque()
        self._cur: np.ndarray | None = None
        self._pos = 0
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            device=device, samplerate=rate, blocksize=1440,
            channels=1, dtype="float32", callback=self._cb,
        )

    def _cb(self, outdata, frames, time_info, status) -> None:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._cur is None or self._pos >= len(self._cur):
                    if not self._q:
                        break
                    self._cur, self._pos = self._q.popleft(), 0
                take = min(frames - filled, len(self._cur) - self._pos)
                out[filled:filled + take] = self._cur[self._pos:self._pos + take]
                self._pos += take
                filled += take
        outdata[:, 0] = out
        self._farend.write(out)

    def play(self, pcm: np.ndarray) -> None:
        with self._lock:
            self._q.append(np.asarray(pcm, dtype=np.float32).reshape(-1))

    def flush(self) -> None:
        with self._lock:
            self._q.clear()
            self._cur = None

    def idle(self) -> bool:
        with self._lock:
            return not self._q and (self._cur is None or self._pos >= len(self._cur))

    def start(self) -> None: self._stream.start()
    def stop(self) -> None: self._stream.stop(); self._stream.close()
```

- [ ] **Step 3: CLI con `devices` y `listen`**

```python
"""CLI de aurelius. S1: devices + listen. run/enroll/bench llegan en S2/S3."""
from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def devices() -> None:
    """Lista dispositivos de audio con sus índices."""
    import sounddevice as sd
    print(sd.query_devices())


def _write_wav(path: Path, audio16k: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio16k, -1, 1) * 32767).astype("<i2").tobytes())


@app.command()
def listen(device: int = typer.Option(None), minutes: float = typer.Option(1.0),
           outdir: Path = typer.Option(Path("capturas"))) -> None:
    """S1: captura → DSP → decima → VAD → un WAV por utterance + eventos JSONL."""
    from aurelius.audio.capture import Capture
    from aurelius.audio.decimate import Decimator48to16
    from aurelius.audio.frames import Reframer
    from aurelius.dsp.chain import Chain
    from aurelius.dsp.stages import HighpassBiquad
    from aurelius.ears.vad import UtteranceSegmenter, make_silero_vad
    from aurelius.events import EventLog

    outdir.mkdir(exist_ok=True)
    log = EventLog(outdir / "eventos.jsonl")
    cap = Capture(device=device)
    reframer, chain, dec = Reframer(480), Chain([HighpassBiquad()]), Decimator48to16()
    seg = UtteranceSegmenter(make_silero_vad())
    n, t_end = 0, time.monotonic() + minutes * 60
    cap.start()
    print("escuchando… habla y calla; Ctrl+C para salir")
    try:
        while time.monotonic() < t_end:
            try:
                block = cap.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            for frame in reframer.push(block):
                u = seg.push(dec.process(chain.process(frame)), time.monotonic())
                if u is not None:
                    n += 1
                    _write_wav(outdir / f"utt_{n:03d}.wav", u.audio)
                    log.emit("speech_end", turn=n, dur_s=round(u.ended_at - u.started_at, 2),
                             chain_ms=round(chain.last_ms, 3))
                    print(f"  utterance {n}: {u.ended_at - u.started_at:.1f}s → utt_{n:03d}.wav")
    except KeyboardInterrupt:
        pass
    finally:
        log.emit("xrun_total", n=cap.xruns); log.emit("drop_total", n=cap.dropped)
        cap.stop(); log.close()
        print(f"xruns={cap.xruns} drops={cap.dropped}")
```

(`import queue` va junto a los imports de arriba del módulo.)

- [ ] **Step 4: Verificación manual (hardware real, día 1 de WASAPI)**

Run: `aurelius devices` → lista con índices; identifica tu micrófono.
Run: `aurelius listen --minutes 0.5` → habla dos frases con pausa entre ellas.
Expected: 2 archivos `capturas\utt_*.wav` audibles y limpios (ábrelos), `eventos.jsonl` con `speech_end` y `chain_ms < 1.0`, `xruns=0 drops=0` (si hay xruns constantes, sube `blocksize` a 2880 — criterio S1 del spec: < N/hora, no cero).

- [ ] **Step 5: Test de integración del bus sin hardware (spec §9)** — Create: `tests\test_pipeline.py`:

```python
"""El camino completo reframe → chain → decima → segmenta, con WAV sintético y VAD falso."""
import numpy as np

from aurelius.audio.decimate import Decimator48to16
from aurelius.audio.frames import Reframer
from aurelius.dsp.chain import Chain
from aurelius.dsp.stages import HighpassBiquad
from aurelius.ears.vad import UtteranceSegmenter


def test_bus_completo_extrae_una_utterance():
    rng = np.random.default_rng(3)
    quieto = np.zeros(48000, dtype=np.float32)                       # 1 s de silencio @48k
    voz = (rng.standard_normal(48000) * 0.3).astype(np.float32)      # 1 s de 'voz'
    señal = np.concatenate([quieto, voz, quieto])

    vad = lambda chunk: 0.9 if float(np.abs(chunk).mean()) > 0.01 else 0.0
    ref, chain, dec = Reframer(480), Chain([HighpassBiquad()]), Decimator48to16()
    seg = UtteranceSegmenter(vad, silence_ms=200)

    utts = []
    for i in range(0, len(señal) - 333, 333):                        # bloques que no dividen exacto
        for frame in ref.push(señal[i:i + 333]):
            u = seg.push(dec.process(chain.process(frame)), now=i / 48000.0)
            if u is not None:
                utts.append(u)
    assert len(utts) == 1
    assert 0.8 < len(utts[0].audio) / 16000 < 1.6                    # ~1 s de voz + colas
```

Run: `pytest tests/test_pipeline.py -q` → `1 passed`.

- [ ] **Step 6: Verificar la suite completa** — Run: `pytest -q` → todo verde (los tests no tocan hardware).

- [ ] **Step 7: Commit — cierre de S1**

```powershell
git add -A; git commit -m "feat(audio): captura WASAPI + playback con far-end + CLI listen (S1 completo)"
```

# FASE S2 — oye, entiende, responde en texto

### Task 9: WakeDetector + prueba de acento

**Files:**
- Create: `src\aurelius\ears\wake.py`

**Interfaces:**
- Consumes: frames de 160 muestras (vista 16 kHz).
- Produces: `WakeDetector(model_name: str = "hey_jarvis", threshold: float = 0.5)`: `push(frame160: np.ndarray) -> float | None` — devuelve el score cuando completa una ventana de 1 280 muestras (80 ms, lo que espera openWakeWord), `None` mientras acumula. El llamador compara contra `detector.threshold`.

- [ ] **Step 1: Implementación (wrapper fino; el modelo no se mockea — verificación por smoke)**

```python
"""Wake word con openWakeWord (modelo hey_jarvis preentrenado, Apache-2.0)."""
from __future__ import annotations

import numpy as np

WINDOW = 1280  # openWakeWord espera bloques de 80 ms @ 16 kHz en int16


class WakeDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5):
        from openwakeword.model import Model
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self._name = model_name
        self.threshold = threshold
        self._buf = np.empty(0, dtype=np.float32)

    def push(self, frame160: np.ndarray) -> float | None:
        self._buf = np.concatenate([self._buf, frame160.astype(np.float32)])
        if len(self._buf) < WINDOW:
            return None
        chunk, self._buf = self._buf[:WINDOW], self._buf[WINDOW:]
        pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        return float(self._model.predict(pcm16)[self._name])

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)
        self._model.reset()
```

- [ ] **Step 2: Descargar modelos base de openWakeWord**

Run: `python -c "import openwakeword.utils as u; u.download_models()"`
Expected: descarga sin error (melspectrogram, embedding y hey_jarvis a la caché del paquete).

- [ ] **Step 3: LA PRUEBA DE HUMO DEL ACENTO (riesgo #1 del spec §15 — hoy, no en la semana 3)**

Script temporal `scripts\wake_smoke.py`:

```python
"""Di 'hey jarvis' 10 veces con tu acento; imprime el score máximo por intento."""
import queue, time
import numpy as np
from aurelius.audio.capture import Capture
from aurelius.audio.decimate import Decimator48to16
from aurelius.audio.frames import Reframer
from aurelius.ears.wake import WakeDetector

cap, ref, dec, wd = Capture(), Reframer(480), Decimator48to16(), WakeDetector()
cap.start(); print("di 'hey jarvis' varias veces; Ctrl+C para salir")
best = 0.0
try:
    while True:
        try:
            block = cap.queue.get(timeout=1.0)
        except queue.Empty:
            continue
        for f in ref.push(block):
            s = wd.push(dec.process(f))
            if s is not None and s > 0.3:
                best = max(best, s)
                print(f"score={s:.2f}")
except KeyboardInterrupt:
    cap.stop(); print(f"máximo: {best:.2f}")
```

Run: `python scripts\wake_smoke.py` — di "hey jarvis" 10 veces, natural.
Expected: scores > 0.5 en ≥ 8 de 10 intentos. **Si no:** baja `threshold` a 0.35 y repite; si sigue fallando, activa el plan B del spec (entrenar wake word custom con el pipeline sintético de openWakeWord, ~1.5 h de Colab) ANTES de seguir con S2 — el proyecto entero depende de esta puerta. Anota el resultado en el commit.

- [ ] **Step 4: Commit** — `git add -A; git commit -m "feat(ears): WakeDetector openWakeWord + smoke de acento (resultado: <anota aquí>)"`

---

### Task 10: Transcriber — faster-whisper en CPU

**Files:**
- Create: `src\aurelius\ears\asr.py`
- Test: `tests\test_asr.py`

**Interfaces:**
- Produces: `Transcriber(model: str = "small", language: str = "es", cpu_threads: int = 8)`: `transcribe(audio16k: np.ndarray) -> str` (texto plano, strip; `""` si no reconoció nada). Carga perezosa del modelo en el primer uso (el arranque del CLI no debe pagar los ~2 s de carga si no se usa).

- [ ] **Step 1: Test que falla (usa fixture real si existe; si no, se salta)**

```python
import wave
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "hola.wav"  # grábalo con: aurelius listen


def _leer(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == 16000
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


@pytest.mark.skipif(not FIXTURE.exists(), reason="graba tests/fixtures/hola.wav con 'aurelius listen' diciendo 'hola aurelio'")
def test_transcribe_fixture():
    from aurelius.ears.asr import Transcriber
    texto = Transcriber().transcribe(_leer(FIXTURE)).lower()
    assert "hola" in texto
```

- [ ] **Step 2: Grabar el fixture** — Run: `aurelius listen --minutes 0.2 --outdir tests\fixtures_tmp`, di "hola aurelio", copia el `utt_001.wav` a `tests\fixtures\hola.wav`.

- [ ] **Step 3: Verificar que falla** — Run: `pytest tests/test_asr.py -q` → ImportError.

- [ ] **Step 4: Implementación mínima**

```python
"""ASR por utterance: faster-whisper small int8 en CPU (spec §5).

Streaming real en español no existe con calidad (verificado 2026-07-13):
el pseudo-streaming por VAD es la vía correcta, no un compromiso.
"""
from __future__ import annotations

import numpy as np


class Transcriber:
    def __init__(self, model: str = "small", language: str = "es", cpu_threads: int = 8):
        self._name, self._lang, self._threads = model, language, cpu_threads
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._name, device="cpu",
                                       compute_type="int8", cpu_threads=self._threads)
        return self._model

    def transcribe(self, audio16k: np.ndarray) -> str:
        segments, _info = self._load().transcribe(
            audio16k.astype(np.float32), language=self._lang,
            beam_size=5, vad_filter=False,      # el VAD ya lo hizo el segmentador
        )
        return "".join(s.text for s in segments).strip()
```

- [ ] **Step 5: Verificar que pasa** — Run: `pytest tests/test_asr.py -q` → `1 passed` (primera corrida descarga el modelo small, ~500 MB). Anota el tiempo de transcripción que reporta pytest — es tu primer dato real para la tabla del README.

- [ ] **Step 6: Commit** — `git add -A; git commit -m "feat(ears): Transcriber faster-whisper small int8 con carga perezosa"`

---

### Task 11: VoiceGate + `aurelius enroll` — solo la voz de Samuel

**Files:**
- Create: `src\aurelius\ears\gate.py`
- Modify: `src\aurelius\cli.py` (añadir comando `enroll`)
- Test: `tests\test_gate.py`

**Interfaces:**
- Consumes (de speechtotext, firmas verificadas contra el código real):
  - `speechtotext.speakers.registry.enroll(name: str, embedding: np.ndarray, *, seconds: float, model: str) -> None`
  - `speechtotext.speakers.registry.get_embeddings() -> dict[str, np.ndarray]`
  - `speechtotext.speakers.registry.list_voices() -> list[dict]` (cada dict trae `name`, `model`, …)
  - `speechtotext.speakers.identify.cosine(a, b) -> float`
  - El registro respeta `SPEECHTOTEXT_HOME` (clave para testear con tmpdir).
- Produces:
  - `GATE_MODEL_ID = "sherpa-wespeaker-en-voxceleb-resnet34-LM"` (constante; va al campo `model` del manifest — el swap de extractor con re-enrolamiento está soportado por diseño del registry).
  - `VoiceGate(owner: str, model_path: Path, threshold: float = 0.45, extractor=None)`: `check(audio16k: np.ndarray) -> tuple[bool, float]` (pasa/no-pasa + coseno), `embed(audio16k) -> np.ndarray`. `extractor` inyectable para tests (callable `audio16k -> np.ndarray`). Si el owner no está enrolado o su manifest tiene otro `model`, levanta `RuntimeError` con el comando exacto de re-enrolamiento.

- [ ] **Step 1: Tests que fallan (registry real en tmpdir, extractor falso — sin modelo ONNX)**

```python
import numpy as np
import pytest


def _gate(monkeypatch, tmp_path, extractor):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    from speechtotext.speakers import registry
    from aurelius.ears.gate import GATE_MODEL_ID, VoiceGate
    registry.enroll("samuel", np.array([1.0, 0.0], dtype=np.float32),
                    seconds=10.0, model=GATE_MODEL_ID)
    return VoiceGate("samuel", model_path=None, threshold=0.7, extractor=extractor)


def test_acepta_voz_parecida(monkeypatch, tmp_path):
    g = _gate(monkeypatch, tmp_path, lambda a: np.array([0.9, 0.1], dtype=np.float32))
    ok, score = g.check(np.zeros(16000, dtype=np.float32))
    assert ok and score > 0.7


def test_rechaza_voz_distinta(monkeypatch, tmp_path):
    g = _gate(monkeypatch, tmp_path, lambda a: np.array([0.0, 1.0], dtype=np.float32))
    ok, score = g.check(np.zeros(16000, dtype=np.float32))
    assert not ok and score < 0.7


def test_error_claro_si_modelo_distinto(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    from speechtotext.speakers import registry
    from aurelius.ears.gate import VoiceGate
    registry.enroll("samuel", np.array([1.0, 0.0]), seconds=10.0, model="pyannote-viejo")
    with pytest.raises(RuntimeError, match="aurelius enroll"):
        VoiceGate("samuel", model_path=None, extractor=lambda a: a)
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_gate.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Gate de identidad: una etapa, sobre la utterance completa, antes del LLM (spec §7).

Extractor: sherpa-onnx (decenas de ms en CPU). NO pyannote: embed_voice() del batch
corre el pipeline completo (segundos por utterance — falsificado en auditoría).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from speechtotext.speakers import registry
from speechtotext.speakers.identify import cosine

GATE_MODEL_ID = "sherpa-wespeaker-en-voxceleb-resnet34-LM"


def _sherpa_extractor(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=2)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(audio16k: np.ndarray) -> np.ndarray:
        s = ext.create_stream()
        s.accept_waveform(16000, audio16k.astype(np.float32))
        s.input_finished()
        return np.asarray(ext.compute(s), dtype=np.float32)

    return embed


class VoiceGate:
    def __init__(self, owner: str, model_path: Path | None,
                 threshold: float = 0.45,
                 extractor: Callable[[np.ndarray], np.ndarray] | None = None):
        self.owner = owner
        self.threshold = threshold
        self._extract = extractor or _sherpa_extractor(model_path)
        meta = {v["name"]: v for v in registry.list_voices()}.get(owner)
        if meta is None or meta.get("model") != GATE_MODEL_ID:
            raise RuntimeError(
                f"la voz '{owner}' no está enrolada con {GATE_MODEL_ID}; "
                f"corre: aurelius enroll {owner} <wav de >=10s de tu voz>"
            )
        self._ref = registry.get_embeddings()[owner]

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        return self._extract(audio16k)

    def check(self, audio16k: np.ndarray) -> tuple[bool, float]:
        score = cosine(self.embed(audio16k), self._ref)
        return score >= self.threshold, score
```

- [ ] **Step 4: Añadir `enroll` al CLI** (en `cli.py`, junto a los otros comandos):

```python
@app.command()
def enroll(name: str, wav: Path,
           model_path: Path = typer.Option(Path("models/wespeaker_en_voxceleb_resnet34_LM.onnx"))) -> None:
    """Registra una voz (>=10 s, una sola persona) para el gate de identidad."""
    import wave as wavmod
    from speechtotext.speakers import registry
    from aurelius.ears.gate import GATE_MODEL_ID, _sherpa_extractor

    with wavmod.open(str(wav), "rb") as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if rate != 16000:
        raise typer.BadParameter(f"el wav debe ser 16 kHz mono (es {rate}); usa los de 'aurelius listen'")
    emb = _sherpa_extractor(model_path)(audio)
    registry.enroll(name, emb, seconds=len(audio) / 16000, model=GATE_MODEL_ID)
    print(f"voz '{name}' enrolada ({len(audio)/16000:.1f}s, modelo {GATE_MODEL_ID})")
```

- [ ] **Step 5: Verificar** — Run: `pytest tests/test_gate.py -q` → `3 passed`.

- [ ] **Step 6: Enrolamiento real** — graba ≥10 s de tu voz con `aurelius listen` (lee cualquier párrafo), luego:
Run: `aurelius enroll samuel capturas\utt_001.wav`
Expected: `voz 'samuel' enrolada (…s, modelo sherpa-wespeaker-…)`. Verifica con `python -c "from speechtotext.speakers import registry; print(registry.list_voices())"`.

- [ ] **Step 7: Commit** — `git add -A; git commit -m "feat(ears): VoiceGate sherpa-onnx sobre el registry de speechtotext + CLI enroll"`

---

### Task 12: Brain — cliente streaming de llama-server

**Files:**
- Create: `src\aurelius\brain\llm.py`
- Test: `tests\test_brain.py`

**Interfaces:**
- Produces: `Brain(base_url: str = "http://127.0.0.1:8080", system_prompt: str | None = None, max_history: int = 12)`: `stream_reply(user_text: str) -> Iterator[str]` (deltas de texto; al agotarse, la respuesta completa quedó en el historial), `reset()` (borra historial). Payload SIEMPRE con `"cache_prompt": True` (sin cache: 7-15 s de prompt eval por turno — spec §5). Historial acotado a `max_history` mensajes (el system no cuenta). Errores de red levantan `BrainError`.

- [ ] **Step 1: Test que falla (servidor SSE falso en un thread — sin modelo)**

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aurelius.brain.llm import Brain, BrainError

RECIBIDO = {}


class _Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECIBIDO.update(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for tok in ["Hola", " Samuel", "."]:
            chunk = {"choices": [{"delta": {"content": tok}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a): pass


@pytest.fixture()
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_stream_y_cache_prompt(fake_server):
    b = Brain(base_url=fake_server)
    assert "".join(b.stream_reply("hola")) == "Hola Samuel."
    assert RECIBIDO["cache_prompt"] is True
    assert RECIBIDO["messages"][0]["role"] == "system"
    assert RECIBIDO["messages"][-1] == {"role": "user", "content": "hola"}


def test_error_de_red():
    b = Brain(base_url="http://127.0.0.1:1")   # puerto muerto
    with pytest.raises(BrainError):
        list(b.stream_reply("hola"))
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_brain.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Cerebro local: cliente streaming del /v1/chat/completions de llama-server."""
from __future__ import annotations

import json
from typing import Iterator

import requests

SYSTEM_PROMPT = (
    "Eres Aurelius, el asistente de voz local de Samuel. Respondes SIEMPRE en español, "
    "en una a tres oraciones cortas: tu texto se convierte en voz y la brevedad es una "
    "feature de latencia. Sin listas, sin markdown, sin emojis. Si no sabes algo, dilo."
)


class BrainError(RuntimeError):
    pass


class Brain:
    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 system_prompt: str | None = None, max_history: int = 12):
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._system = {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
        self._hist: list[dict] = []
        self._max = max_history

    def stream_reply(self, user_text: str) -> Iterator[str]:
        self._hist.append({"role": "user", "content": user_text})
        self._hist = self._hist[-self._max:]
        payload = {"messages": [self._system] + self._hist,
                   "stream": True, "cache_prompt": True, "temperature": 0.7}
        parts: list[str] = []
        try:
            with requests.post(self._url, json=payload, stream=True, timeout=(5, 120)) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {})
                    tok = delta.get("content")
                    if tok:
                        parts.append(tok)
                        yield tok
        except requests.RequestException as e:
            self._hist.pop()          # el turno no ocurrió
            raise BrainError(str(e)) from e
        self._hist.append({"role": "assistant", "content": "".join(parts)})

    def reset(self) -> None:
        self._hist.clear()
```

- [ ] **Step 4: Verificar que pasan** — Run: `pytest tests/test_brain.py -q` → `2 passed`.

- [ ] **Step 5: Smoke real (requiere llama-server corriendo — ver docs\setup-modelos.md)**

Run: `models\llama\llama-server.exe -m models\qwen2.5-7b-instruct-q4_k_m.gguf -t 9 -c 4096 --port 8080` (en otra terminal) y luego
`python -c "from aurelius.brain.llm import Brain; [print(t, end='', flush=True) for t in Brain().stream_reply('hola, ¿quién eres?')]"`
Expected: respuesta breve en español, streameada. Anota tokens/s del log de llama-server (dato para el README y para la decisión diferida #3 del spec).

- [ ] **Step 6: Commit** — `git add -A; git commit -m "feat(brain): cliente streaming llama-server con cache_prompt e historial acotado"`

---

### Task 13: FSM — la orquestadora pura

**Files:**
- Create: `src\aurelius\fsm.py`
- Test: `tests\test_fsm.py`

**Interfaces:**
- Produces (lógica pura, sin threads ni audio — todo inyectado por eventos):
  - `St` (Enum): `IDLE, AWAKE, LISTENING, THINKING, SPEAKING, HANGOVER`.
  - `Ev` (Enum): `WAKE, SPEECH_START, UTTERANCE, ASR_EMPTY, GATE_FAIL, SENTENCE, REPLY_DONE, PLAYBACK_IDLE, TICK`.
  - `Act` (Enum): `EARCON, TRANSCRIBE, SPEAK, FLUSH_SPEECH, RESET_LISTEN`.
  - `FSM(hangover_s: float = 0.4, awake_timeout_s: float = 6.0)`: `state: St`, `handle(ev: Ev, now: float, data=None) -> list[tuple[Act, object]]`. `data` viaja intacto en la acción (la `Utterance` en `TRANSCRIBE`, la oración en `SPEAK`).
- Reglas (spec §7): wake en SPEAKING = barge-in (`FLUSH_SPEECH` + `EARCON` → AWAKE); `PLAYBACK_IDLE` solo pasa a HANGOVER si ya llegó `REPLY_DONE`; HANGOVER dura `hangover_s` y al expirar (`TICK`) emite `RESET_LISTEN` → IDLE; AWAKE expira a IDLE por `awake_timeout_s`; `ASR_EMPTY`/`GATE_FAIL` en THINKING → IDLE directo.

- [ ] **Step 1: Tests que fallan**

```python
from aurelius.fsm import FSM, Act, Ev, St


def _acts(pairs):
    return [a for a, _ in pairs]


def test_turno_feliz():
    m = FSM()
    assert _acts(m.handle(Ev.WAKE, 0.0)) == [Act.EARCON] and m.state is St.AWAKE
    m.handle(Ev.SPEECH_START, 0.5); assert m.state is St.LISTENING
    out = m.handle(Ev.UTTERANCE, 2.0, data="u1")
    assert out == [(Act.TRANSCRIBE, "u1")] and m.state is St.THINKING
    out = m.handle(Ev.SENTENCE, 4.0, data="Hola.")
    assert out == [(Act.SPEAK, "Hola.")] and m.state is St.SPEAKING
    m.handle(Ev.SENTENCE, 4.5, data="¿Qué tal?")     # sigue en SPEAKING
    m.handle(Ev.REPLY_DONE, 5.0); assert m.state is St.SPEAKING
    m.handle(Ev.PLAYBACK_IDLE, 6.0); assert m.state is St.HANGOVER
    assert m.handle(Ev.TICK, 6.2) == []               # aún no expira (hangover 0.4)
    out = m.handle(Ev.TICK, 6.5)
    assert out == [(Act.RESET_LISTEN, None)] and m.state is St.IDLE


def test_barge_in():
    m = FSM(); m.state = St.SPEAKING
    out = m.handle(Ev.WAKE, 1.0)
    assert _acts(out) == [Act.FLUSH_SPEECH, Act.EARCON] and m.state is St.AWAKE


def test_playback_idle_sin_reply_done_no_cierra():
    m = FSM(); m.state = St.SPEAKING
    m.handle(Ev.PLAYBACK_IDLE, 1.0)                   # el LLM sigue generando
    assert m.state is St.SPEAKING


def test_gate_fail_y_asr_vacio_vuelven_a_idle():
    for ev in (Ev.GATE_FAIL, Ev.ASR_EMPTY):
        m = FSM(); m.state = St.THINKING
        m.handle(ev, 1.0); assert m.state is St.IDLE


def test_awake_expira():
    m = FSM(awake_timeout_s=6.0)
    m.handle(Ev.WAKE, 0.0)
    m.handle(Ev.TICK, 3.0); assert m.state is St.AWAKE
    m.handle(Ev.TICK, 6.5); assert m.state is St.IDLE


def test_eventos_fuera_de_estado_se_ignoran():
    m = FSM()
    assert m.handle(Ev.SENTENCE, 0.0, data="x") == [] and m.state is St.IDLE
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_fsm.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""FSM del loop (spec §7). Pura: recibe eventos con reloj externo, devuelve acciones."""
from __future__ import annotations

from enum import Enum, auto


class St(Enum):
    IDLE = auto(); AWAKE = auto(); LISTENING = auto()
    THINKING = auto(); SPEAKING = auto(); HANGOVER = auto()


class Ev(Enum):
    WAKE = auto(); SPEECH_START = auto(); UTTERANCE = auto()
    ASR_EMPTY = auto(); GATE_FAIL = auto(); SENTENCE = auto()
    REPLY_DONE = auto(); PLAYBACK_IDLE = auto(); TICK = auto()


class Act(Enum):
    EARCON = auto(); TRANSCRIBE = auto(); SPEAK = auto()
    FLUSH_SPEECH = auto(); RESET_LISTEN = auto()


class FSM:
    def __init__(self, hangover_s: float = 0.4, awake_timeout_s: float = 6.0):
        self.state = St.IDLE
        self._hangover_s = hangover_s
        self._awake_timeout_s = awake_timeout_s
        self._entered = 0.0
        self._reply_done = False

    def _go(self, state: "St", now: float) -> None:
        self.state = state
        self._entered = now

    def handle(self, ev: Ev, now: float, data=None) -> list[tuple[Act, object]]:
        s = self.state
        if ev is Ev.WAKE and s in (St.IDLE, St.SPEAKING, St.HANGOVER):
            acts = [(Act.FLUSH_SPEECH, None)] if s is St.SPEAKING else []
            self._go(St.AWAKE, now)
            self._reply_done = False
            return acts + [(Act.EARCON, None)]
        if s is St.AWAKE:
            if ev is Ev.SPEECH_START:
                self._go(St.LISTENING, now)
            elif ev is Ev.TICK and now - self._entered > self._awake_timeout_s:
                self._go(St.IDLE, now)
            return []
        if s is St.LISTENING and ev is Ev.UTTERANCE:
            self._go(St.THINKING, now)
            return [(Act.TRANSCRIBE, data)]
        if s is St.THINKING:
            if ev in (Ev.ASR_EMPTY, Ev.GATE_FAIL):
                self._go(St.IDLE, now)
                return []
            if ev is Ev.SENTENCE:
                self._go(St.SPEAKING, now)
                return [(Act.SPEAK, data)]
            if ev is Ev.REPLY_DONE:      # respuesta vacía: nada que decir
                self._go(St.IDLE, now)
                return []
        if s is St.SPEAKING:
            if ev is Ev.SENTENCE:
                return [(Act.SPEAK, data)]
            if ev is Ev.REPLY_DONE:
                self._reply_done = True
                return []
            if ev is Ev.PLAYBACK_IDLE and self._reply_done:
                self._go(St.HANGOVER, now)
                return []
        if s is St.HANGOVER and ev is Ev.TICK and now - self._entered > self._hangover_s:
            self._go(St.IDLE, now)
            return [(Act.RESET_LISTEN, None)]
        return []
```

- [ ] **Step 4: Verificar que pasan** — Run: `pytest tests/test_fsm.py -q` → `6 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(fsm): máquina de estados pura con barge-in, hangover y timeout de AWAKE"`

---

### Task 14: Config TOML + `aurelius run` en modo texto — hito de S2

**Files:**
- Create: `src\aurelius\config.py`
- Create: `aurelius.toml.example`
- Modify: `src\aurelius\cli.py` (añadir comando `run`)
- Test: `tests\test_config.py`

**Interfaces:**
- Produces: `Config` (dataclass) con `load(path: Path | None) -> Config` — si el path no existe, defaults. Campos exactos:

```python
input_device: int | None = None
output_device: int | None = None
wake_model: str = "hey_jarvis"
wake_threshold: float = 0.5
owner: str = "samuel"
gate_threshold: float = 0.45
gate_model_path: str = "models/wespeaker_en_voxceleb_resnet34_LM.onnx"
asr_model: str = "small"
language: str = "es"
brain_url: str = "http://127.0.0.1:8080"
piper_exe: str = "models/piper/piper.exe"
piper_voice: str = "models/voces/es_MX-claude-high.onnx"
piper_rate: int = 22050
events_path: str = "eventos.jsonl"
```

- Consumes: todo lo anterior. `run` en S2 imprime la respuesta en consola (el TTS llega en S3, Task 17 lo enchufa).

- [ ] **Step 1: Test de config que falla**

```python
from pathlib import Path
from aurelius.config import Config

def test_defaults_sin_archivo(tmp_path):
    c = Config.load(tmp_path / "no-existe.toml")
    assert c.owner == "samuel" and c.wake_threshold == 0.5

def test_carga_toml(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text('owner = "otro"\nwake_threshold = 0.35\n', encoding="utf-8")
    c = Config.load(p)
    assert c.owner == "otro" and c.wake_threshold == 0.35
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_config.py -q` → ImportError.

- [ ] **Step 3: Implementar Config**

```python
"""Config plana en TOML. Un archivo, una tabla, cero anidación (ponytail)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    input_device: int | None = None
    output_device: int | None = None
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5
    owner: str = "samuel"
    gate_threshold: float = 0.45
    gate_model_path: str = "models/wespeaker_en_voxceleb_resnet34_LM.onnx"
    asr_model: str = "small"
    language: str = "es"
    brain_url: str = "http://127.0.0.1:8080"
    piper_exe: str = "models/piper/piper.exe"
    piper_voice: str = "models/voces/es_MX-claude-high.onnx"
    piper_rate: int = 22050
    events_path: str = "eventos.jsonl"

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        if path is None or not Path(path).exists():
            return cls()
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
```

`aurelius.toml.example`: las mismas claves con sus defaults, comentadas línea a línea.

- [ ] **Step 4: Implementar `run` (modo texto) en cli.py**

Arquitectura del comando (los threads del spec §4): el hilo principal corre la FSM sobre una `queue.SimpleQueue` de eventos `(Ev, data)`; un thread `pipeline` mueve audio (captura → reframe → chain → decima → wake + segmentador) y publica eventos; un thread `worker` por turno hace ASR → gate → Brain y publica `SENTENCE`/`REPLY_DONE`.

```python
@app.command()
def run(config: Path = typer.Option(Path("aurelius.toml")), text_only: bool = typer.Option(True)) -> None:
    """El loop completo. En S2: text_only=True (respuestas a consola)."""
    import queue as q
    import threading
    import time as t

    from aurelius.audio.capture import Capture
    from aurelius.audio.decimate import Decimator48to16
    from aurelius.audio.frames import Reframer
    from aurelius.brain.llm import Brain, BrainError
    from aurelius.config import Config
    from aurelius.dsp.chain import Chain
    from aurelius.dsp.stages import HighpassBiquad
    from aurelius.ears.asr import Transcriber
    from aurelius.ears.gate import VoiceGate
    from aurelius.ears.vad import UtteranceSegmenter, make_silero_vad
    from aurelius.ears.wake import WakeDetector
    from aurelius.events import EventLog
    from aurelius.fsm import FSM, Act, Ev, St
    from aurelius.voice.split import SentenceSplitter

    cfg = Config.load(config)
    log = EventLog(cfg.events_path)
    events: q.SimpleQueue = q.SimpleQueue()

    cap = Capture(device=cfg.input_device)
    wake = WakeDetector(cfg.wake_model, cfg.wake_threshold)
    seg = UtteranceSegmenter(make_silero_vad())
    gate = VoiceGate(cfg.owner, Path(cfg.gate_model_path), cfg.gate_threshold)
    asr = Transcriber(cfg.asr_model, cfg.language)
    brain = Brain(cfg.brain_url)
    fsm = FSM()
    listen_gate = threading.Event()   # half-duplex: cerrado durante SPEAKING
    listen_gate.set()

    def pipeline() -> None:
        reframer, chain, dec = Reframer(480), Chain([HighpassBiquad()]), Decimator48to16()
        was_speaking = False
        while True:
            try:
                block = cap.queue.get(timeout=1.0)
            except q.Empty:
                events.put((Ev.TICK, None))
                continue
            for frame in reframer.push(block):
                f16 = dec.process(chain.process(frame))
                score = wake.push(f16)
                if score is not None and score >= wake.threshold:
                    events.put((Ev.WAKE, score))
                if not listen_gate.is_set():
                    continue                      # SPEAKING: VAD/ASR gateados, wake vivo
                u = seg.push(f16, t.monotonic())
                if seg.speaking and not was_speaking:
                    events.put((Ev.SPEECH_START, None))
                was_speaking = seg.speaking
                if u is not None:
                    events.put((Ev.UTTERANCE, u))
            events.put((Ev.TICK, None))

    def think(utt) -> None:
        texto = asr.transcribe(utt.audio)
        log.emit("asr_done", chars=len(texto))
        if not texto:
            events.put((Ev.ASR_EMPTY, None)); return
        ok, score = gate.check(utt.audio)
        log.emit("gate", ok=ok, score=round(score, 3))
        if not ok:
            events.put((Ev.GATE_FAIL, score)); return
        print(f"\n[tú] {texto}")
        splitter, first = SentenceSplitter(), True
        try:
            for delta in brain.stream_reply(texto):
                if first:
                    log.emit("first_token"); first = False
                for sent in splitter.push(delta):
                    log.emit("first_sentence")
                    events.put((Ev.SENTENCE, sent))
            resto = splitter.flush()
            if resto:
                events.put((Ev.SENTENCE, resto))
        except BrainError as e:
            events.put((Ev.SENTENCE, "Dame un segundo, se me cayó el cerebro."))
            print(f"[brain error] {e}")
        events.put((Ev.REPLY_DONE, None))

    threading.Thread(target=pipeline, daemon=True).start()
    cap.start()
    print("aurelius en modo texto — di 'hey jarvis'")
    try:
        while True:
            ev, data = events.get()
            if ev is Ev.UTTERANCE:
                log.emit("speech_end", dur_s=round(data.ended_at - data.started_at, 2))
            for act, payload in fsm.handle(ev, t.monotonic(), data):
                if act is Act.EARCON:
                    print("\n[!] te escucho")
                elif act is Act.TRANSCRIBE:
                    threading.Thread(target=think, args=(payload,), daemon=True).start()
                elif act is Act.SPEAK:
                    print(f"[aurelius] {payload}")
                    if text_only:
                        events.put((Ev.PLAYBACK_IDLE, None))   # sin TTS aún: cerrar el turno
                elif act is Act.RESET_LISTEN:
                    seg.reset(); wake.reset()
            listen_gate.clear() if fsm.state is St.SPEAKING and not text_only else listen_gate.set()
    except KeyboardInterrupt:
        cap.stop(); log.close(); print("\nchao")
```

> Nota: la Task 15 crea `SentenceSplitter` — al ejecutar este plan en orden con subagentes, implementa la Task 15 ANTES de correr este `run` (el import fallaría). El orden de commits puede ser 14→15 igual; solo no ejecutes `run` hasta tener 15. Alternativa: ejecuta Tasks 15 y 16 primero si prefieres estricta ejecutabilidad por commit.

- [ ] **Step 5: Verificar config** — Run: `pytest tests/test_config.py -q` → `2 passed`.

- [ ] **Step 6: HITO S2 (manual, con llama-server corriendo y Task 15 hecha)**

Run: `aurelius run` → di "hey jarvis", espera el `[!] te escucho`, pregunta "¿quién eres?".
Expected: `[tú] ¿quién eres?` + respuesta breve en `[aurelius]`. Pide a otra persona (o un audio de YouTube) decir "hey jarvis" y hablar: debe aparecer `gate` con `ok=false` en `eventos.jsonl` y ninguna respuesta. Mide en `eventos.jsonl`: `speech_end → first_token` (target < 2.5 s con cache caliente).

- [ ] **Step 7: Commit** — `git add -A; git commit -m "feat(cli): run en modo texto — wake, ASR, gate de identidad y cerebro streaming (S2)"`

# FASE S3 — habla y vitrina

### Task 15: SentenceSplitter — oraciones en streaming

**Files:**
- Create: `src\aurelius\voice\split.py`
- Test: `tests\test_split.py`

**Interfaces:**
- Produces: `SentenceSplitter()`: `push(delta: str) -> list[str]` (oraciones completas detectadas hasta ahora), `flush() -> str | None` (el resto al terminar el stream). Regla de corte: terminador `.!?…` seguido de whitespace. No corta decimales (`1.75` — el punto no va seguido de espacio) ni abreviaturas comunes del español.

- [ ] **Step 1: Tests que fallan**

```python
from aurelius.voice.split import SentenceSplitter


def test_corta_en_terminadores():
    s = SentenceSplitter()
    got = s.push("Hola Samuel. ¿Todo bien? Sí, ")
    assert got == ["Hola Samuel.", "¿Todo bien?"]
    assert s.flush() == "Sí,"


def test_no_corta_abreviaturas_ni_decimales():
    s = SentenceSplitter()
    got = s.push("El Sr. Gómez mide 1.75 metros. Fin")
    assert got == ["El Sr. Gómez mide 1.75 metros."]
    assert s.flush() == "Fin"


def test_streaming_por_deltas_de_un_token():
    s = SentenceSplitter()
    got = []
    for d in ["Ho", "la", ". ", "Chao", ". "]:
        got += s.push(d)
    assert got == ["Hola.", "Chao."]
```

- [ ] **Step 2: Verificar que fallan** — Run: `pytest tests/test_split.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""Splitter de oraciones en streaming: el TTS empieza con la primera oración (spec §5)."""
from __future__ import annotations

_TERM = ".!?…"
_ABREV = {"sr", "sra", "srta", "dr", "dra", "ing", "lic", "etc", "ud", "uds", "no"}


class SentenceSplitter:
    def __init__(self):
        self._buf = ""

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        i = 0
        while i < len(self._buf) - 1:
            if self._buf[i] in _TERM and self._buf[i + 1].isspace() and not self._falso_corte(i):
                out.append(self._buf[: i + 1].strip())
                self._buf = self._buf[i + 1:].lstrip()
                i = 0
            else:
                i += 1
        return [s for s in out if s]

    def _falso_corte(self, i: int) -> bool:
        if self._buf[i] != ".":
            return False
        palabra = self._buf[:i].rsplit(None, 1)[-1].lower() if self._buf[:i].strip() else ""
        return palabra in _ABREV

    def flush(self) -> str | None:
        s, self._buf = self._buf.strip(), ""
        return s or None
```

- [ ] **Step 4: Verificar que pasan** — Run: `pytest tests/test_split.py -q` → `3 passed`.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(voice): splitter de oraciones en streaming con abreviaturas y decimales"`

---

### Task 16: PiperTTS — subprocess por oración

**Files:**
- Create: `src\aurelius\voice\tts.py`
- Test: `tests\test_tts.py`

**Interfaces:**
- Produces: `PiperTTS(exe: Path, voice: Path, native_rate: int = 22050, out_rate: int = 48000)`: `synth(text: str) -> np.ndarray` — float32 mono a `out_rate` (48 kHz, el contrato del bus de salida). Un subprocess POR ORACIÓN: EOF del stdout = fin del audio de esa oración (la auditoría mató el pipe persistente sin framing). El comando queda en `self._cmd` (los tests lo sustituyen por un exe falso).

- [ ] **Step 1: Test que falla (piper falso: un script python que emite PCM conocido)**

```python
import sys
from pathlib import Path

import numpy as np

from aurelius.voice.tts import PiperTTS

FAKE = """
import sys
import numpy as np
sys.stdin.read()                                # el texto (se ignora)
pcm = (np.ones(22050) * 1000).astype("<i2")     # 1 s de 'audio' a 22050 Hz
sys.stdout.buffer.write(pcm.tobytes())
"""


def test_synth_devuelve_float32_a_48k(tmp_path):
    script = tmp_path / "fake_piper.py"
    script.write_text(FAKE, encoding="utf-8")
    tts = PiperTTS(exe=Path("ignorado"), voice=Path("ignorada"))
    tts._cmd = [sys.executable, str(script)]
    out = tts.synth("hola")
    assert out.dtype == np.float32
    assert abs(len(out) - 48000) < 500          # 1 s remuestreado a 48 kHz
    assert 0.02 < np.abs(out).max() < 0.05      # 1000/32768 ≈ 0.03
```

- [ ] **Step 2: Verificar que falla** — Run: `pytest tests/test_tts.py -q` → ImportError.

- [ ] **Step 3: Implementación mínima**

```python
"""TTS: Piper (GPL-3.0) como subprocess externo por oración. Jamás se importa."""
from __future__ import annotations

import subprocess
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


class PiperTTS:
    def __init__(self, exe: Path, voice: Path, native_rate: int = 22050, out_rate: int = 48000):
        self._cmd = [str(exe), "--model", str(voice), "--output_raw"]
        g = gcd(out_rate, native_rate)
        self._up, self._down = out_rate // g, native_rate // g

    def synth(self, text: str) -> np.ndarray:
        r = subprocess.run(self._cmd, input=text.encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        pcm = np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768.0
        if self._up != self._down:
            pcm = resample_poly(pcm, self._up, self._down).astype(np.float32)
        return pcm
```

- [ ] **Step 4: Verificar que pasa** — Run: `pytest tests/test_tts.py -q` → `1 passed`.

- [ ] **Step 5: Smoke con el Piper real** (requiere docs\setup-modelos.md hecho):
`python -c "from pathlib import Path; from aurelius.voice.tts import PiperTTS; import sounddevice as sd; t=PiperTTS(Path('models/piper/piper.exe'), Path('models/voces/es_MX-claude-high.onnx')); a=t.synth('Hola Samuel, soy Aurelius.'); sd.play(a, 48000); sd.wait()"`
Expected: lo escuchas. Si la voz no te gusta, baja las otras candidatas del spec §14 (es_ES-davefx, es_ES-sharvard, es_AR-daniela) y decide a oído — anota la elegida en `aurelius.toml`.

- [ ] **Step 6: Commit** — `git add -A; git commit -m "feat(voice): PiperTTS subprocess por oración con remuestreo al bus de 48k"`

---

### Task 17: SPEAKING completo — voz, earcon, barge-in y hangover

**Files:**
- Modify: `src\aurelius\cli.py` (comando `run`: enchufar TTS + playback; `text_only` pasa a `False` por default)

**Interfaces:**
- Consumes: `PiperTTS.synth`, `Player` (play/flush/idle), `FarEndRing`, `FSM` (Act.SPEAK/FLUSH_SPEECH/EARCON), `Ev.PLAYBACK_IDLE`.
- Produces: el loop completo hablado. Orden de reproducción garantizado por UN worker de TTS que consume una cola de oraciones en orden.

- [ ] **Step 1: Añadir al setup de `run` (tras crear `brain`)**

```python
    from aurelius.audio.farend import FarEndRing
    from aurelius.audio.playback import Player
    from aurelius.voice.tts import PiperTTS

    farend = FarEndRing()
    player = Player(farend, device=cfg.output_device)
    tts = PiperTTS(Path(cfg.piper_exe), Path(cfg.piper_voice), native_rate=cfg.piper_rate)
    dur = 0.08
    earcon = (0.2 * np.sin(2 * np.pi * 880 * np.arange(int(dur * 48000)) / 48000)).astype(np.float32)

    say_q: q.SimpleQueue = q.SimpleQueue()      # oraciones en orden hacia el TTS
    turno_sonando = threading.Event()

    def tts_worker() -> None:
        primera = True
        while True:
            sent = say_q.get()
            if sent is None:                     # señal de nuevo turno
                primera = True
                continue
            try:
                pcm = tts.synth(sent)
            except Exception as e:               # Piper falló: log + saltar oración (spec §8)
                log.emit("tts_error", error=str(e))
                continue
            if primera:
                log.emit("first_audio")
                primera = False
            player.play(pcm)
            turno_sonando.set()

    threading.Thread(target=tts_worker, daemon=True).start()
    player.start()
```

- [ ] **Step 2: Reemplazar las ramas de acciones del loop principal**

```python
            for act, payload in fsm.handle(ev, t.monotonic(), data):
                if act is Act.EARCON:
                    player.play(earcon)
                    say_q.put(None)              # resetea 'primera oración' del turno
                elif act is Act.TRANSCRIBE:
                    threading.Thread(target=think, args=(payload,), daemon=True).start()
                elif act is Act.SPEAK:
                    print(f"[aurelius] {payload}")
                    say_q.put(payload)
                elif act is Act.FLUSH_SPEECH:    # barge-in
                    while not say_q.empty():
                        say_q.get_nowait()
                    player.flush()
                    turno_sonando.clear()
                elif act is Act.RESET_LISTEN:
                    seg.reset(); wake.reset()
            if fsm.state is St.SPEAKING:
                listen_gate.clear()              # half-duplex: solo el wake sigue vivo
                if turno_sonando.is_set() and player.idle() and say_q.empty():
                    turno_sonando.clear()
                    events.put((Ev.PLAYBACK_IDLE, None))
            else:
                listen_gate.set()
```

Y en el cierre (`KeyboardInterrupt`): `player.stop()` junto a `cap.stop()` y `log.close()`.

- [ ] **Step 3: HITO S3 — conversación multivuelta sin manos (manual)**

Con llama-server corriendo: `aurelius run`
1. "hey jarvis" → earcon → pregunta algo → te responde HABLANDO, empezando por la primera oración mientras el LLM sigue generando.
2. Conversa 3+ vueltas seguidas. Expected: tras cada respuesta vuelve solo a IDLE (hangover) y NO se transcribe a sí mismo (revisa que no haya `speech_end` fantasma en `eventos.jsonl` justo tras `first_audio`).
3. Barge-in: mientras habla, di "hey jarvis" → se calla a mitad de oración y suena el earcon.
4. Gate: otra voz dice "hey jarvis" y pregunta → earcon sí (el wake no discrimina), pero `gate ok=false` y silencio.

Si el paso 2 falla (se oye a sí mismo), sube el hangover en `FSM(hangover_s=0.6)` — documentado como limitación v0.1 (sin AEC hasta F2).

- [ ] **Step 4: Suite completa** — Run: `pytest -q` → todo verde.

- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat(voice): loop hablado completo — earcon, TTS por oración, barge-in y hangover (S3)"`

---

### Task 18: bench, README vitrina y publicación

**Files:**
- Modify: `src\aurelius\cli.py` (comando `bench`)
- Modify: `README.md` (la vitrina)

**Interfaces:**
- Produces: `aurelius bench` — lee `eventos.jsonl` y reporta p50 por frontera. Las latencias del README salen de aquí: **medidas, no prometidas**.

- [ ] **Step 1: Implementar `bench`**

```python
@app.command()
def bench(events: Path = typer.Option(Path("eventos.jsonl"))) -> None:
    """Latencias p50 por turno desde el log JSONL. La tabla del README sale de aquí."""
    import json
    import statistics as st

    turns, marks = [], {}
    for line in events.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["kind"] == "speech_end":
            marks = {"t0": e["ts"]}
        elif e["kind"] in ("asr_done", "first_token", "first_sentence", "first_audio") and "t0" in marks:
            marks.setdefault(e["kind"], e["ts"] - marks["t0"])   # solo la PRIMERA ocurrencia por turno
            if e["kind"] == "first_audio":
                turns.append(marks); marks = {}
    if not turns:
        print("sin turnos completos en el log"); raise typer.Exit(1)
    print(f"turnos completos: {len(turns)}")
    for k in ("asr_done", "first_token", "first_sentence", "first_audio"):
        vals = [t[k] for t in turns if k in t]
        if vals:
            print(f"  fin de habla → {k:14s} p50 = {st.median(vals):.2f}s")
```

- [ ] **Step 2: Verificar bench** — conversa 5+ turnos con `aurelius run` y corre `aurelius bench`.
Expected: tabla con p50; `first_audio` debería caer en 3.5-5 s (spec §10). Si no, revisa que llama-server tenga `-t 9` y el cache caliente (segundo turno en adelante).

- [ ] **Step 3: Escribir el README vitrina** (reemplaza el stub; pega los números reales de `aurelius bench`):

```markdown
# aurelius

Asistente de voz 100% local. Tu audio nunca sale de tu máquina.

Di "hey jarvis". Aurelius solo despierta con TU voz (identificación por
embedding, no por contraseña), te transcribe localmente, piensa con un LLM
local y te responde hablando en ~N s [pega aquí el p50 de first_audio].

[GIF de demo: grábalo con ScreenToGif — una conversación de 2 vueltas + un barge-in]

## Cómo funciona

[pega aquí el diagrama del bus del spec §4: mic → 48 kHz/10 ms → cadena DSP
→ decimador → wake + VAD → ASR → gate de identidad → LLM → TTS por oración]

El bus corre a 48 kHz con frames de 10 ms y una cadena DSP enchufable: la
fase 2 (supresión de ruido, EQ de voz, compresor, AEC) se enchufa como
etapas sin tocar la topología. Hoy la cadena lleva un highpass de 80 Hz.

## Latencias medidas (Ryzen 9 5900X, CPU, cache caliente)

[pega aquí la salida de `aurelius bench`]

Turno con cache frío: bastante más lento (se documenta, no se esconde).

## Setup

1. `pip install -e .`  (Python 3.11+, Windows; ffmpeg no hace falta)
2. Modelos y binarios: docs/setup-modelos.md (una sola vez, ~6 GB)
3. Enrola tu voz: `aurelius listen` (lee un párrafo) → `aurelius enroll tu_nombre capturas\utt_001.wav`
4. Arranca llama-server (comando en docs/setup-modelos.md) y `aurelius run`

## Límites conocidos del v0.1

- Half-duplex sin AEC: con altavoz fuerte, el barge-in por voz pierde
  sensibilidad. El AEC llega en la fase 2 sobre el ring far-end ya instalado.
- El wake word no discrimina voces (el gate de identidad sí, justo después).
- ASR por utterance (~1 s tras callarte), no palabra a palabra: streaming
  real en español no existe con calidad hoy.

## Licencias

| Componente | Licencia |
|---|---|
| aurelius | MIT |
| numpy / scipy | BSD |
| sounddevice (PortAudio) | MIT |
| pysilero-vad + modelo Silero VAD | MIT |
| openWakeWord + hey_jarvis | Apache-2.0 |
| faster-whisper + pesos Whisper | MIT |
| sherpa-onnx + wespeaker resnet34 | Apache-2.0 |
| llama.cpp | MIT |
| Qwen2.5-7B-Instruct | Apache-2.0 |
| Piper (binario externo, no incluido) | GPL-3.0 |
| Voz es_MX [verifica MODEL_CARD y anótala] | [según MODEL_CARD] |
| speechtotext (registry de voces) | MIT |

Ningún modelo se redistribuye en este repo: docs/setup-modelos.md los
descarga de sus fuentes.
```

- [ ] **Step 4: Publicar**

```powershell
git add -A; git commit -m "docs: README vitrina con latencias medidas y tabla de licencias"
gh repo create sssamuelll/aurelius --public --source . --push
```

Expected: repo público en GitHub. El hito respira.

---

## Notas de ejecución

- **Orden estricto recomendado:** 1→13, luego 15 y 16 ANTES de correr el `run` de la Task 14 (su import de `SentenceSplitter` lo exige; los commits pueden ir en el orden del plan).
- **Cláusula de degradación (spec §6):** si al cerrar S1 los xruns no bajan de criterio con blocksize 2880, congela el bus a 16 kHz: `Capture(rate=16000)`, elimina el `Decimator48to16` del pipeline (la vista ML ES el bus) e instancia `HighpassBiquad(80, rate=16000)`. Nada de ears/brain/voice cambia.
- **Los tests jamás requieren hardware ni modelos** (los de ASR/fixture se saltan si no hay fixture). `pytest -q` debe correr en cualquier máquina.
- **pyannote no aparece en este repo.** Si un import lo trae por accidente vía speechtotext, es un bug: aurelius solo usa `speechtotext.speakers.registry` e `identify`.
- **Caída de llama-server:** cada turno es un request nuevo, así que la recuperación es automática cuando el server vuelve; mientras tanto suena la respuesta enlatada de `BrainError`. Un supervisor del proceso (spec §8, "reinicio con backoff") solo tiene sentido cuando aurelius arranque el server él mismo — se difiere a F2 y se anota como limitación v0.1.
- **Desconexión del device de audio:** sounddevice detiene el stream sin crash del proceso; reconectar requiere reiniciar `aurelius run`. Limitación v0.1 documentada, no se maneja en el esqueleto.
