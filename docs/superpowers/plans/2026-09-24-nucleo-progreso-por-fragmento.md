# Núcleo: progreso, cancelación y texto por fragmento — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que `transcribe()` informe del progreso y atienda la cancelación dentro de cada trozo, fragmento a fragmento, y entregue el texto de cada fragmento para una vista previa.

**Architecture:** `AsrBackend.transcribe` gana dos argumentos por palabra clave, `on_segment` y `cancel`. faster-whisper los atiende recorriendo su generador perezoso; whisper.cpp, leyendo en vivo el stdout de whisper-cli y terminando el proceso si se cancela. `core.transcribe` suma el avance de los trozos en paralelo con un contador protegido por un cerrojo, emite `Progress("transcribe")` en segundos de audio y pasa cada fragmento, en tiempo global, a un callback nuevo, `on_segment(PartialSegment)`. El CLI pasa a pintar minutos de audio.

**Tech Stack:** Python ≥ 3.11, stdlib (`threading`, `subprocess`, `re`, `tempfile`), faster-whisper, whisper-cli v1.9.1, typer/rich, pytest. Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-09-24-app-escritorio-cimientos-design.md` — §6.2 (notificaciones `progress` y `partial`) y §6.4 (cambios en el núcleo). Este es el paso 1 de §10.

## Global Constraints

- Python `>=3.11`. CI en ubuntu, macOS y Windows con 3.11 y 3.14; los tests no usan red, GPU ni modelos.
- Sin dependencias nuevas.
- El núcleo nunca imprime: ni `print` ni escrituras en stdout en `src/speechtotext/core/` ni en `src/speechtotext/asr/`.
- El árbol público habla inglés: código, comentarios, docstrings, nombres de tests y mensajes (`tests/test_public_tree.py` lo vigila). Una palabra en español que sea dato lleva el marcador `# spanish-is-data: <razón>` en su línea.
- `Progress`, `transcribe()` y `AsrBackend` son contrato: cada cambio va a `docs/api.md` y al `CHANGELOG.md`; lo que rompe, bajo "Changed — breaking".
- La cancelación es siempre `AsrError("cancelled", True, "transcription cancelled")`: el mismo código y mensaje que hoy.
- Una señal que el motor no emite está ausente, no a cero: no se inventa progreso ni texto.
- TDD: todo cambio de producción empieza con una prueba que falla por la razón esperada.
- El audio de las pruebas reales es privado: ningún texto transcrito sale de la máquina que lo transcribe, ni a la terminal compartida, ni a un commit, ni al PR.
- Rama: `feat/core-progress-per-fragment`, creada desde `main`. Un commit por tarea, con las líneas de atribución que indique la sesión que lo ejecuta.

## Review Focus

1. **Un callback del consumidor que lanza una excepción** (un bug en `on_segment` o en `on_progress`): tiene que salir tal cual, no disfrazada de `backend_failed`, y whisper-cli no puede quedarse corriendo. Tests: Task 1 `test_a_callback_error_is_not_reported_as_an_engine_failure`; Task 2 `test_a_callback_error_kills_whisper_cli_and_propagates`.
2. **Audio sin voz** (el motor no devuelve ningún segmento): la barra tiene que llegar igual al total. Test: Task 3 `test_silence_still_reaches_the_total`.
3. **Un fantasma sobre el relleno del último tramo** (Whisper narrando pasado el final del trozo): ni el progreso puede pasar del total ni la vista previa puede enseñar ese texto. Test: Task 3 `test_a_phantom_past_the_chunk_end_neither_overshoots_nor_previews`.
4. **La salida real de whisper-cli en Windows:** CRLF, una línea vacía al principio y bytes que no son UTF-8 válido. Test: Task 2 `test_live_lines_tolerate_crlf_blank_lines_and_bad_bytes`.
5. **Trozos en paralelo que terminan en otro orden:** el avance que ve el consumidor nunca retrocede. Test: Task 3 `test_parallel_chunks_sum_and_never_go_back`.

---

## Contexto para quien llega sin él

- **Tests.** En el Mac: `.venv/bin/python -m pytest -q` desde la raíz del repo. En el desktop (Windows, `ssh desktop`): `D:\Desktop\projects\speechtotext\.venv\Scripts\python.exe -m pytest -q`; ese entorno tiene además torch, transformers y el SDK de MCP.
- **Trozos.** Por encima de 20 minutos (`CHUNK_THRESHOLD = 1200`) el audio se corta en trozos de ~10 minutos (`target_len = 600`), en silencios. Con faster-whisper, hasta `jobs` trozos (4 por defecto) corren en paralelo, en hilos; con whisper.cpp, de uno en uno. Solo cuando hay archivo y trozos se escriben checkpoints.
- **Tiempos.** El backend devuelve tiempos locales al trozo que recibe; `_run_span` los desplaza con `shift_segments` y los recorta con `clip_to_end`. Regla de `clip_to_end`: un segmento con más relleno que audio (`end - fin > fin - start`) se descarta; uno que sobresale poco se recorta al final del trozo.
- **Hoy** el progreso de `transcribe` cuenta trozos (`done` = trozos terminados, `total` = número de trozos) y `cancel` solo se mira entre trozos. Un audio de menos de 20 minutos es un solo trozo: la barra salta de 0 a 1 y cancelar espera al final.
- **whisper-cli, medido el 2026-09-24 en el desktop** (v1.9.1, `large-v3-q5_0`, 60 s de audio, stdout redirigido a un pipe y `-np`): escribe cada segmento en stdout mientras decodifica, en una ráfaga por ventana de 30 s (a los 7,9 s, 12,6 s y 14,1 s; el proceso salió a los 14,2 s). Formato de línea: `[00:00:01.920 --> 00:00:04.060]  texto`, con CRLF en Windows y una primera línea vacía.
- **faster-whisper** devuelve un generador perezoso: decodifica una ventana de 30 s cada vez que se le piden segmentos. Esa es la granularidad de progreso y de cancelación en los dos motores.

## Mapa de archivos

| Archivo | Cambio |
|---|---|
| `src/speechtotext/asr/base.py` | `raise_if_cancelled()`; `AsrBackend.transcribe` gana `on_segment` y `cancel` |
| `src/speechtotext/asr/faster_whisper.py` | Recorre el generador segmento a segmento; `_segment()` y `_optional_float()` |
| `src/speechtotext/asr/whispercpp.py` | `parse_live_line()`, `_timeout_for()`, `_Watchdog`; `Popen` en lugar de `run`, con el seam `popen` |
| `src/speechtotext/core/transcribe.py` | `PartialSegment`, `SegmentCallback`, `_Meter`; `transcribe(on_segment=…)`; progreso en segundos |
| `src/speechtotext/cli/app.py` | Progreso en minutos de audio; el aviso de trozos sale del evento de decodificación |
| `tests/test_faster_whisper_backend.py`, `tests/test_whispercpp_backend.py`, `tests/test_transcribe.py`, `tests/test_cli.py`, `tests/test_api_contract.py` | Pruebas |
| `docs/api.md`, `CHANGELOG.md` | Contrato |

---

### Task 1: el contrato del backend gana `on_segment` y `cancel`; faster-whisper los atiende

**Files:**
- Modify: `src/speechtotext/asr/base.py`
- Modify: `src/speechtotext/asr/faster_whisper.py` (el método `transcribe`, líneas 144-255, y dos funciones nuevas de módulo)
- Test: `tests/test_faster_whisper_backend.py`
- Docs: `docs/api.md` (sección "The `asr/` layer")

**Interfaces:**
- Consumes: nada nuevo.
- Produces:
  - `speechtotext.asr.base.raise_if_cancelled(cancel: threading.Event | None) -> None` — lanza `AsrError("cancelled", True, "transcription cancelled")` si `cancel` está puesto.
  - `AsrBackend.transcribe(samples: np.ndarray, request: TranscriptionRequest, *, on_segment: Callable[[TranscriptionSegment], None] | None = None, cancel: threading.Event | None = None) -> TranscriptionResult`. Cada segmento llega a `on_segment` en cuanto el motor lo tiene, en orden, con tiempos locales a `samples`; `cancel` se mira entre segmentos.

- [ ] **Step 1: crear la rama**

```bash
git switch main && git pull --ff-only && git switch -c feat/core-progress-per-fragment
```

- [ ] **Step 2: escribir las pruebas que fallan**

En `tests/test_faster_whisper_backend.py`, cambiar la línea de import de `speechtotext.asr` y añadir `threading`:

```python
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from speechtotext.asr import AsrBackend, AsrError, Caps, TranscriptionRequest
from speechtotext.asr.faster_whisper import FasterWhisperBackend, FasterWhisperConfig
```

Añadir al final del archivo:

```python
def _raw(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text, words=None,
                           no_speech_prob=None, avg_logprob=None, compression_ratio=None)


def _lazy(segments, pulled):
    """Like faster-whisper's generator: it records each segment as the engine hands it over."""
    for segment in segments:
        pulled.append(segment.text)
        yield segment


def test_each_segment_reaches_the_callback_before_the_next_is_decoded():
    pulled, seen = [], []
    backend = _backend("large-v3", _lazy([_raw(0.0, 1.0, " one"), _raw(1.0, 2.5, " two")], pulled),
                       _INFO, {})
    result = backend.transcribe(
        _samples(3.0), TranscriptionRequest(),
        on_segment=lambda s: seen.append((s.start, s.end, s.text, list(pulled))),
    )
    assert seen == [(0.0, 1.0, " one", [" one"]), (1.0, 2.5, " two", [" one", " two"])]
    assert [s.text for s in result.segments] == [" one", " two"]


def test_cancel_stops_between_segments_and_decodes_no_further():
    pulled, stop = [], threading.Event()
    segments = [_raw(0.0, 1.0, " one"), _raw(1.0, 2.0, " two"), _raw(2.0, 3.0, " three")]
    backend = _backend("large-v3", _lazy(segments, pulled), _INFO, {})
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(3.0), TranscriptionRequest(),
                           on_segment=lambda s: stop.set(), cancel=stop)
    assert (ei.value.code, ei.value.recoverable) == ("cancelled", True)
    assert str(ei.value) == "transcription cancelled"
    assert pulled == [" one"]


def test_cancel_before_starting_never_reaches_the_model():
    stop = threading.Event()
    stop.set()

    class Untouchable:
        def transcribe(self, audio, **opts):
            pytest.fail("decoded after cancel")

    backend = FasterWhisperBackend("large-v3", model_factory=lambda path, **kw: Untouchable())
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(), TranscriptionRequest(), cancel=stop)
    assert ei.value.code == "cancelled"


def test_a_callback_error_is_not_reported_as_an_engine_failure():
    backend = _backend("large-v3", iter([_raw(0.0, 1.0, " one")]), _INFO, {})

    def broken(segment):
        raise ValueError("bug in the caller")

    with pytest.raises(ValueError, match="bug in the caller"):
        backend.transcribe(_samples(), TranscriptionRequest(), on_segment=broken)


def test_an_error_while_decoding_is_still_an_engine_failure():
    # Guard: passes today and must keep passing once the loop reads one segment at a time.
    def failing():
        yield _raw(0.0, 1.0, " one")
        raise RuntimeError("CUDA error: out of range")

    backend = _backend("large-v3", failing(), _INFO, {})
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(), TranscriptionRequest())
    assert ei.value.code == "backend_failed" and "CUDA error" in str(ei.value)
```

- [ ] **Step 3: comprobar que fallan por la razón esperada**

Run: `.venv/bin/python -m pytest tests/test_faster_whisper_backend.py -q`
Expected: 4 FAIL con `TypeError: ... got an unexpected keyword argument 'on_segment'` (o `'cancel'`); `test_an_error_while_decoding_is_still_an_engine_failure` PASA (es de guarda).

- [ ] **Step 4: el contrato en `src/speechtotext/asr/base.py`**

Sustituir el archivo entero por:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np

from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult, TranscriptionSegment

# Capability contract per engine. Guiding rule: degrade with a warning when the result
# is still what was requested with less precision; reject when the knob would be inert;
# never silence or substitution.
Cap = Literal["honored", "degraded", "rejected"]


@dataclass(frozen=True)
class Caps:
    hotwords: Cap
    vad: Cap
    word_timestamps: Cap


class AsrError(RuntimeError):
    def __init__(self, code: str, recoverable: bool, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


def raise_if_cancelled(cancel: threading.Event | None) -> None:
    """The one way to stop: the same code and message wherever the check happens."""
    if cancel is not None and cancel.is_set():
        raise AsrError("cancelled", True, "transcription cancelled")


@runtime_checkable
class AsrBackend(Protocol):
    """A speech-to-text engine. It accepts float32 mono at 16 kHz; nothing else. The caller
    resamples. The object is the model cache: warm() loads it once."""

    backend_id: str
    caps: Caps

    @property
    def model_id(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    @property
    def quant(self) -> str: ...

    @property
    def device(self) -> str: ...

    def warm(self) -> None:
        ...

    def transcribe(
        self,
        samples: np.ndarray,
        request: TranscriptionRequest,
        *,
        on_segment: Callable[[TranscriptionSegment], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> TranscriptionResult:
        """Each segment goes to `on_segment` as soon as the engine has it, in order, with
        times local to `samples`. `cancel` is checked between segments: once it is set,
        this raises AsrError("cancelled") and returns nothing partial."""
        ...
```

- [ ] **Step 5: faster-whisper segmento a segmento**

En `src/speechtotext/asr/faster_whisper.py`:

1. Imports: añadir `import threading` junto a `import time`, y cambiar `from speechtotext.asr.base import AsrError, Caps` por `from speechtotext.asr.base import AsrError, Caps, raise_if_cancelled`.
2. Justo antes de `class FasterWhisperBackend:`, añadir:

```python
def _optional_float(source, name: str) -> float | None:
    value = getattr(source, name, None)
    return float(value) if value is not None else None


def _segment(raw) -> TranscriptionSegment:
    """One faster-whisper segment in the contract's types; a signal it lacks stays None."""
    words = tuple(
        TranscriptionWord(text=word.word, start=float(word.start), end=float(word.end),
                          confidence=_optional_float(word, "probability"))
        for word in (getattr(raw, "words", None) or ())
    )
    signals = SegmentNativeSignals(
        no_speech=_optional_float(raw, "no_speech_prob"),
        avg_logprob=_optional_float(raw, "avg_logprob"),
        compression_ratio=_optional_float(raw, "compression_ratio"),
    )
    return TranscriptionSegment(float(raw.start), float(raw.end), raw.text, words, signals)
```

3. Sustituir el método `transcribe` entero (de `def transcribe(` hasta el `return TranscriptionResult(...)` final) por:

```python
    def transcribe(
        self,
        samples: np.ndarray,
        request: TranscriptionRequest,
        *,
        on_segment: Callable[[TranscriptionSegment], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> TranscriptionResult:
        self.warm()
        raise_if_cancelled(cancel)
        started = self._clock()
        try:
            raw_segments, info = self._model.transcribe(
                samples,
                language=None if request.language == "auto" else request.language,
                beam_size=request.beam_size,
                vad_filter=request.vad,
                hotwords=", ".join(request.hotwords) or None,
                initial_prompt=request.context,
                condition_on_previous_text=False,
                word_timestamps=request.word_timestamps,
            )
            pending = iter(raw_segments)
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError("backend_failed", True, str(exc)) from exc
        segments: list[TranscriptionSegment] = []
        # The generator decodes one 30 s window per request, so each segment reaches the
        # caller while the rest of the audio is still ahead. The callback runs outside the
        # try: its own error is not the engine's and propagates as it is.
        while True:
            try:
                raw = next(pending, None)
            except AsrError:
                raise
            except Exception as exc:
                raise AsrError("backend_failed", True, str(exc)) from exc
            if raw is None:
                break
            segment = _segment(raw)
            segments.append(segment)
            if on_segment is not None:
                on_segment(segment)
            raise_if_cancelled(cancel)
        elapsed_ms = round((self._clock() - started) * 1000)
        no_speech: list[float] = []
        logprobs: list[tuple[float, float]] = []
        compression: list[float] = []
        for segment in segments:
            signals = segment.native_signals
            weight = max(0.001, segment.end - segment.start)
            if signals.no_speech is not None:
                no_speech.append(signals.no_speech)
            if signals.avg_logprob is not None:
                logprobs.append((signals.avg_logprob, weight))
            if signals.compression_ratio is not None:
                compression.append(signals.compression_ratio)
        avg_logprob = (
            sum(value * weight for value, weight in logprobs)
            / sum(weight for _, weight in logprobs)
            if logprobs
            else None
        )
        text = "".join(segment.text for segment in segments).strip()
        warnings: list[str] = []
        if not text:
            warnings.append("empty_transcript")
        language = str(getattr(info, "language", None) or request.language)
        if request.language != "auto" and language != request.language:
            warnings.append("language_mismatch")
        return TranscriptionResult(
            text=text,
            language=language,
            words=tuple(word for segment in segments for word in segment.words),
            segments=tuple(segments),
            backend=self.backend_id,
            model=self._model_id,
            model_version=self._model_version,
            latency_ms=elapsed_ms,
            native_signals=NativeSignals(
                no_speech=max(no_speech) if no_speech else None,
                avg_logprob=avg_logprob,
                compression_ratio=max(compression) if compression else None,
                language_probability=(
                    float(info.language_probability)
                    if getattr(info, "language_probability", None) is not None
                    else None
                ),
            ),
            warnings=tuple(warnings),
        )
```

- [ ] **Step 6: comprobar que pasan**

Run: `.venv/bin/python -m pytest tests/test_faster_whisper_backend.py tests/test_asr_contract.py -q`
Expected: todo PASS.

- [ ] **Step 7: documentar el contrato**

En `docs/api.md`, sección "The `asr/` layer", sustituir el bloque `class AsrBackend(Protocol):` por:

```python
class AsrBackend(Protocol):
    backend_id: str          # "faster-whisper" | "whispercpp"
    caps: Caps               # hotwords / vad / word_timestamps -> honored | degraded | rejected
    model_id: str; model_version: str; engine_version: str; quant: str; device: str
    def warm(self) -> None                                  # loads (once); the object is the cache
    def transcribe(self, samples: np.ndarray, request: TranscriptionRequest, *,
                   on_segment: Callable[[TranscriptionSegment], None] | None = None,
                   cancel: threading.Event | None = None) -> TranscriptionResult
```

Y justo debajo del bloque, antes del párrafo que empieza por "`TranscriptionResult.segments` are", añadir:

```markdown
`on_segment` receives each segment as soon as the engine has it, in order,
with times local to `samples`. `cancel` is checked between segments: once it
is set, the call raises `AsrError("cancelled")` instead of returning a partial
result. Both engines decode 30-second windows, and that is the granularity of
both.
```

- [ ] **Step 8: la suite entera**

Run: `.venv/bin/python -m pytest -q`
Expected: todo PASS (el núcleo todavía no pasa los argumentos nuevos; ningún falso se entera).

- [ ] **Step 9: commit**

```bash
git add src/speechtotext/asr/base.py src/speechtotext/asr/faster_whisper.py tests/test_faster_whisper_backend.py docs/api.md
git commit -m "feat(asr): backends take on_segment and cancel; faster-whisper honors them per segment"
```

---

### Task 2: whisper.cpp atiende `on_segment` y `cancel`

**Files:**
- Modify: `src/speechtotext/asr/whispercpp.py`
- Test: `tests/test_whispercpp_backend.py`
- Docs: `docs/api.md` (la viñeta de `WhisperCppBackend`)

**Interfaces:**
- Consumes: `raise_if_cancelled` (Task 1).
- Produces:
  - `speechtotext.asr.whispercpp.parse_live_line(line: bytes) -> TranscriptionSegment | None`.
  - `speechtotext.asr.whispercpp._timeout_for(n_samples: int) -> float`, con la fórmula de siempre: `max(120, 4 * n_samples / 16000)`.
  - `WhisperCppBackend(model, *, exe=None, model_path=None, popen=subprocess.Popen, clock=time.perf_counter, poll_s=0.2)`: el seam `run` desaparece y lo sustituye `popen`.

- [ ] **Step 1: el falso de whisper-cli pasa a ser un proceso**

En `tests/test_whispercpp_backend.py`:

1. Imports: añadir `import threading` y `from speechtotext.asr import AsrError`, y ampliar el import del backend a `from speechtotext.asr.whispercpp import WhisperCppBackend, parse_live_line, parse_ojf`.
2. Sustituir `_run_stub` entero por:

```python
def _popen_stub(lines=(), rc=0, stderr_bytes=b"", write="fixture", hang=False):
    """A fake whisper-cli. It writes the -ojf JSON as `write` says ('fixture' copies the
    synthetic fixture, a dict writes that JSON, None writes nothing, a str writes garbage),
    prints `lines` on stdout and exits with `rc`; with `hang` it waits instead until it is
    terminated or killed. Returns (popen, seen)."""
    seen = {}

    class FakeProc:
        def __init__(self, cmd, stdout=None, stderr=None):
            seen.update(cmd=list(cmd), proc=self, base=cmd[cmd.index("-of") + 1],
                        wav=cmd[cmd.index("-f") + 1])
            seen["wav_existed"] = os.path.exists(seen["wav"])
            base = seen["base"]
            if write == "fixture":
                shutil.copyfile(FIXTURE, base + ".json")
            elif isinstance(write, dict):
                Path(base + ".json").write_text(json.dumps(write), encoding="utf-8")
            elif isinstance(write, str):
                Path(base + ".json").write_text(write, encoding="utf-8")
            stderr.write(stderr_bytes)
            self.returncode = None
            self.terminated = self.killed = False
            self._ended = threading.Event()
            self.stdout = self._print()

        def _print(self):
            yield from lines
            if hang:
                self._ended.wait(5)
            else:
                self._end(rc)

        def _end(self, code):
            if self.returncode is None:
                self.returncode = code
            self._ended.set()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self._ended.wait(5)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self._end(-15)

        def kill(self):
            self.killed = True
            self._end(-9)

    return FakeProc, seen
```

3. Sustituir `_backend` por:

```python
def _backend(popen, **kw):
    # cycle, not iter: some tests call transcribe() twice on the same backend.
    ticks = itertools.cycle([10.0, 10.5])
    return WhisperCppBackend(
        "large-v3", exe=Path("C:/wcpp/Release/whisper-cli.exe"),
        model_path=Path("C:/wcpp/models/ggml.bin"), popen=popen, clock=lambda: next(ticks),
        poll_s=0.01, **kw,
    )
```

4. En los tests existentes, reemplazos mecánicos:
   - `run, seen = _run_stub(` → `popen, seen = _popen_stub(`;
   - `run, _ = _run_stub(` → `popen, _ = _popen_stub(`;
   - `_backend(run)` → `_backend(popen)`;
   - en `test_the_temp_wav_is_pcm16_mono_16k_and_gets_deleted`, `return run(cmd, **kw)` → `return popen(cmd, **kw)` y `seen["wav_existia"]` → `seen["wav_existed"]`;
   - en `test_a_nonzero_rc_fails_with_the_tail_of_stderr`, `_run_stub(write=None, rc=3, stderr=stderr)` → `_popen_stub(write=None, rc=3, stderr_bytes=stderr)`.
5. Sustituir `test_timeout_scales_proportionally_with_a_floor` por:

```python
def test_timeout_scales_proportionally_with_a_floor():
    assert whispercpp._timeout_for(100 * 16000) == 400   # 4x the duration
    assert whispercpp._timeout_for(10 * 16000) == 120    # floor for a cold JIT
```

- [ ] **Step 2: escribir las pruebas nuevas**

Añadir al final de `tests/test_whispercpp_backend.py`:

```python
def test_live_lines_become_segments_with_local_times():
    segment = parse_live_line(b"[00:01:02.340 --> 01:00:00.000]  hello\r\n")
    assert (segment.start, segment.end, segment.text) == (62.34, 3600.0, " hello")
    assert segment.words == () and segment.native_signals.no_speech is None


def test_live_lines_tolerate_crlf_blank_lines_and_bad_bytes():
    assert parse_live_line(b"\r\n") is None                                  # the blank first line
    assert parse_live_line(b"whisper_init_from_file: loading model\n") is None
    assert parse_live_line(b"[00:00:00.000 --> 00:00:01.000]   \r\n") is None  # no text
    segment = parse_live_line(b"[00:00:01.000 --> 00:00:02.000]  caf\xe9\r\n")  # not UTF-8
    assert segment is not None and segment.text.startswith(" caf")


def test_each_live_line_reaches_the_callback_and_the_json_is_still_the_result():
    popen, _ = _popen_stub(lines=[b"\r\n",
                                  b"[00:00:00.000 --> 00:00:01.500]  Segment one\r\n",
                                  b"[00:00:01.500 --> 00:00:03.000]  Segment two\r\n"])
    got = []
    result = _backend(popen).transcribe(_samples(), TranscriptionRequest(), on_segment=got.append)
    assert [(s.start, s.end, s.text) for s in got] == [(0.0, 1.5, " Segment one"),
                                                       (1.5, 3.0, " Segment two")]
    assert len(result.segments) == 6        # the fixture's six: the JSON decides the result


def test_cancel_terminates_whisper_cli_and_reports_cancelled():
    stop = threading.Event()
    popen, seen = _popen_stub(lines=[b"[00:00:00.000 --> 00:00:01.000]  one\r\n"], hang=True)
    with pytest.raises(AsrError) as ei:
        _backend(popen).transcribe(_samples(), TranscriptionRequest(),
                                   on_segment=lambda s: stop.set(), cancel=stop)
    assert (ei.value.code, str(ei.value)) == ("cancelled", "transcription cancelled")
    assert seen["proc"].terminated
    assert not os.path.exists(seen["wav"]) and not os.path.exists(seen["base"] + ".json")


def test_cancel_before_starting_never_launches_whisper_cli():
    stop = threading.Event()
    stop.set()
    backend = _backend(lambda *a, **k: pytest.fail("launched after cancel"))
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(), TranscriptionRequest(), cancel=stop)
    assert ei.value.code == "cancelled"


def test_a_callback_error_kills_whisper_cli_and_propagates():
    popen, seen = _popen_stub(lines=[b"[00:00:00.000 --> 00:00:01.000]  one\r\n"], hang=True)

    def broken(segment):
        raise ValueError("bug in the caller")

    with pytest.raises(ValueError, match="bug in the caller"):
        _backend(popen).transcribe(_samples(), TranscriptionRequest(), on_segment=broken)
    assert seen["proc"].killed
    assert not os.path.exists(seen["wav"])


def test_a_run_past_its_timeout_is_killed(monkeypatch):
    monkeypatch.setattr(whispercpp, "_timeout_for", lambda n_samples: 0.05)
    popen, seen = _popen_stub(hang=True)
    with pytest.raises(RuntimeError, match="did not finish"):
        _backend(popen).transcribe(_samples(), TranscriptionRequest())
    assert seen["proc"].killed
```

- [ ] **Step 3: comprobar que fallan por la razón esperada**

Run: `.venv/bin/python -m pytest tests/test_whispercpp_backend.py -q`
Expected: la colección falla con `ImportError: cannot import name 'parse_live_line'`.

- [ ] **Step 4: implementar**

En `src/speechtotext/asr/whispercpp.py`:

1. Imports: añadir `import re` y `import threading` (en orden alfabético con los demás de stdlib), y cambiar `from speechtotext.asr.base import AsrError, Caps` por `from speechtotext.asr.base import AsrError, Caps, raise_if_cancelled`.
2. Después de `_NO_SIGNALS = ...`, añadir:

```python
# What whisper-cli prints per segment while it decodes, -np included. Measured on v1.9.1
# (2026-09-24): one burst per 30 s window, CRLF on Windows, a blank first line.
_LIVE_LINE = re.compile(
    r"^\[(\d+):(\d{2}):(\d{2})\.(\d{3}) --> (\d+):(\d{2}):(\d{2})\.(\d{3})\]\s?(.*)$"
)


def _seconds(hours: str, minutes: str, secs: str, millis: str) -> float:
    return (((int(hours) * 60 + int(minutes)) * 60 + int(secs)) * 1000 + int(millis)) / 1000


def parse_live_line(line: bytes) -> TranscriptionSegment | None:
    """One stdout line, `[00:00:01.920 --> 00:00:04.060]  text`, as a segment with times
    local to the input. Anything else is None: the blank first line, notices, a segment
    with no text. Only a preview: the result is still read from the -ojf JSON."""
    match = _LIVE_LINE.match(line.decode("utf-8", errors="replace").rstrip("\r\n"))
    if match is None:
        return None
    *stamps, text = match.groups()
    if not text.strip():
        return None
    return TranscriptionSegment(_seconds(*stamps[:4]), _seconds(*stamps[4:]), text, (), _NO_SIGNALS)


def _timeout_for(n_samples: int) -> float:
    # 4x covers the measured warm case 17 times (19.3x real time); 120 s floor for cold JIT.
    return max(120, 4 * n_samples / SAMPLE_RATE)


class _Watchdog(threading.Thread):
    """Ends whisper-cli when the caller cancels or the run outlives its timeout. The reader
    blocks on stdout for a whole 30 s window; this thread is what makes cancelling prompt."""

    def __init__(self, proc, cancel: threading.Event | None, timeout: float, poll_s: float) -> None:
        super().__init__(daemon=True)
        self._proc, self._cancel, self._timeout, self._poll_s = proc, cancel, timeout, poll_s
        self.cancelled = False
        self.timed_out = False

    def run(self) -> None:
        deadline = time.monotonic() + self._timeout
        while self._proc.poll() is None:
            if self._cancel is not None and self._cancel.is_set():
                self.cancelled = True
                self._proc.terminate()
                return
            if time.monotonic() > deadline:
                self.timed_out = True
                self._proc.kill()
                return
            time.sleep(self._poll_s)
```

3. En `WhisperCppBackend.__init__`, sustituir el parámetro `run: Callable = subprocess.run,` por `popen: Callable = subprocess.Popen,`, añadir `poll_s: float = 0.2,` después de `clock`, y en el cuerpo cambiar `self._run = run` por `self._popen = popen` y añadir `self._poll_s = poll_s`.
4. En `transcribe`:
   - la firma pasa a ser `def transcribe(self, samples: np.ndarray, request: TranscriptionRequest, *, on_segment: Callable[[TranscriptionSegment], None] | None = None, cancel: threading.Event | None = None) -> TranscriptionResult:`;
   - después de `self.warm()`, añadir `raise_if_cancelled(cancel)`;
   - sustituir las líneas que calculan `timeout`, llaman a `self._run(...)` y comprueban `proc.returncode` (el comentario "4x covers…" incluido; ahora vive en `_timeout_for`) por:

```python
            self._run_cli(cmd, _timeout_for(len(samples)), on_segment, cancel)
```

5. Añadir el método, después de `transcribe`:

```python
    def _run_cli(self, cmd: list[str], timeout: float,
                 on_segment: Callable[[TranscriptionSegment], None] | None,
                 cancel: threading.Event | None) -> None:
        """Run whisper-cli, handing each live line to `on_segment`. stderr goes to a file, not
        a pipe: nothing reads it until the end, and a full pipe would stall the process."""
        with tempfile.TemporaryFile() as err:
            proc = self._popen(cmd, stdout=subprocess.PIPE, stderr=err)
            watchdog = _Watchdog(proc, cancel, timeout, self._poll_s)
            watchdog.start()
            try:
                for line in proc.stdout:
                    segment = parse_live_line(line)
                    if segment is not None and on_segment is not None:
                        on_segment(segment)
                returncode = proc.wait()
            finally:
                if proc.poll() is None:      # the callback raised: never leave whisper-cli running
                    proc.kill()
                    proc.wait()
                watchdog.join()
            if watchdog.cancelled:
                raise AsrError("cancelled", True, "transcription cancelled")
            if watchdog.timed_out:
                raise RuntimeError(f"whisper-cli did not finish in {timeout:.0f} s")
            if returncode != 0:
                err.seek(0)
                tail = "\n".join(err.read().decode("utf-8", errors="replace").splitlines()[-10:])
                raise RuntimeError(f"whisper-cli rc={returncode}: {tail}")
```

- [ ] **Step 5: comprobar que pasan**

Run: `.venv/bin/python -m pytest tests/test_whispercpp_backend.py -q`
Expected: todo PASS.

- [ ] **Step 6: documentar**

En `docs/api.md`, en la viñeta de `speechtotext.asr.whispercpp.WhisperCppBackend(model)`, añadir al final de la viñeta:

```markdown
  It reads whisper-cli's output as it decodes, one line per segment, for
  `on_segment`; the result still comes from its JSON. Cancelling ends the
  process.
```

- [ ] **Step 7: la suite entera y commit**

Run: `.venv/bin/python -m pytest -q`
Expected: todo PASS.

```bash
git add src/speechtotext/asr/whispercpp.py tests/test_whispercpp_backend.py docs/api.md
git commit -m "feat(asr): whisper.cpp streams its segments and ends the process on cancel"
```

---

### Task 3: `transcribe()` cuenta segundos de audio, cancela dentro del trozo y entrega `PartialSegment`; el CLI lo pinta

**Files:**
- Modify: `src/speechtotext/core/transcribe.py`
- Modify: `src/speechtotext/cli/app.py` (import y la función `on_progress` de `transcribe_file`, líneas ~256-287)
- Test: `tests/test_transcribe.py`, `tests/test_cli.py`, `tests/test_api_contract.py`
- Docs: `docs/api.md` (sección "`transcribe()`"), `CHANGELOG.md`

**Interfaces:**
- Consumes: `AsrBackend.transcribe(..., *, on_segment, cancel)` y `raise_if_cancelled` (Task 1); los dos backends reales los atienden (Tasks 1 y 2).
- Produces:
  - `speechtotext.core.transcribe.PartialSegment(start: float, end: float, text: str)`, frozen: segundos de toda la grabación, texto sin espacios en los extremos y todavía sin hablante.
  - `speechtotext.core.transcribe.SegmentCallback = Callable[[PartialSegment], None]`.
  - `transcribe(..., on_progress=None, on_segment: SegmentCallback | None = None, cancel=None)`.
  - `Progress("transcribe", done, total, detail)`: `done` y `total` en segundos de audio; `done` nunca retrocede; el `detail` de un trozo terminado acaba en `(new)` o `(cache)`.

- [ ] **Step 1: el falso de `test_transcribe.py` se porta como un motor real**

En `tests/test_transcribe.py`, en la sección "orchestration", añadir `from speechtotext.asr.base import raise_if_cancelled` junto a los otros imports de `speechtotext.asr`, y sustituir el método `transcribe` de `FakeBackend` por:

```python
    def transcribe(self, samples, request, *, on_segment=None, cancel=None):
        self.calls.append((len(samples), request))
        if self.boom is not None:
            if self.wrap:
                raise AsrError("backend_failed", True, str(self.boom))
            raise self.boom
        segs = tuple(
            TranscriptionSegment(s, e, t, (TranscriptionWord(t, s, e, None),) if request.word_timestamps else (),
                                 SegmentNativeSignals(None, None, None))
            for s, e, t in self.segments
        )
        for seg in segs:          # like a real engine: hand each one over, then look at cancel
            if on_segment is not None:
                on_segment(seg)
            raise_if_cancelled(cancel)
        return TranscriptionResult(
            text="".join(t for _, _, t in self.segments).strip(), language=self.language,
            words=(), segments=segs, backend=self.backend_id, model=self.model_id,
            model_version=self.model_version, latency_ms=1,
            native_signals=NativeSignals(None, None, None, self.probability), warnings=(),
        )
```

Y en las tres subclases que redefinen `transcribe` (las dos `WarningBackend` y `CancellingBackend`), cambiar la firma a `def transcribe(self, samples, request, **kw):` y la llamada a `super().transcribe(samples, request, **kw)`.

- [ ] **Step 2: ajustar las dos pruebas de progreso que ya existen**

En `test_progress_uses_callback_and_decodes_once`, sustituir las aserciones sobre `events` por:

```python
    assert [e.stage for e in events] == ["decode", "decode", "load", "transcribe", "transcribe"]
    assert (events[0].done, events[0].total) == (0, None)
    assert (events[1].done, events[1].total, events[1].detail) == (10.0, 10.0, "a.wav")
    assert (events[3].done, events[3].total) == (2.0, 10.0)     # the 1-2 s segment, as it lands
    assert (events[-1].done, events[-1].total) == (10.0, 10.0)
    assert events[-1].detail.endswith("(new)")
```

En `test_multiple_chunks_reassemble_in_order_with_global_times`, sustituir las dos líneas de `details` por:

```python
    finished = [e.detail for e in events if e.stage == "transcribe" and e.detail.endswith("(new)")]
    assert len(finished) == 2
```

- [ ] **Step 3: escribir las pruebas nuevas del núcleo**

Añadir al final de `tests/test_transcribe.py`:

```python
# --- progress, cancel and text per fragment (spec 2026-09-24, §6.4) ---------------------

def _steps(events):
    return [e for e in events if e.stage == "transcribe"]


def _two_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "largo.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    return audio


def test_progress_advances_in_audio_seconds_within_a_single_chunk():
    events = []
    backend = FakeBackend([(0.0, 2.0, " a"), (2.0, 5.0, " b"), (5.0, 8.0, " c")])
    core.transcribe(_zeros(10.0), backend=backend, chunk=False, on_progress=events.append)
    assert [e.done for e in _steps(events)] == [2.0, 5.0, 8.0, 10.0]
    assert {e.total for e in _steps(events)} == {10.0}
    assert _steps(events)[-1].detail.endswith("(new)")


def test_parallel_chunks_sum_and_never_go_back(tmp_path, monkeypatch):
    audio = _two_chunks(tmp_path, monkeypatch)
    events = []
    core.transcribe(audio, backend=FakeBackend([(1.0, 2.0, " t"), (2.0, 300.0, " u")]),
                    chunk=True, jobs=2, on_progress=events.append)
    dones = [e.done for e in _steps(events)]
    assert dones == sorted(dones) and dones[-1] == 1200.0
    assert {e.total for e in _steps(events)} == {1200.0}


def test_partial_segments_arrive_in_global_time_without_speakers(tmp_path, monkeypatch):
    audio = _two_chunks(tmp_path, monkeypatch)
    partials = []
    core.transcribe(audio, backend=FakeBackend([(1.0, 2.0, " t")]), chunk=True, jobs=2,
                    on_segment=partials.append)
    assert sorted(partials, key=lambda p: p.start) == [
        core.PartialSegment(1.0, 2.0, "t"), core.PartialSegment(601.0, 602.0, "t"),
    ]


def test_cancel_reaches_the_engine_in_the_middle_of_a_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(600.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0)])
    stop, seen = threading.Event(), []

    def stop_after_the_first(partial):
        seen.append(partial.text)
        stop.set()

    backend = FakeBackend([(0.0, 1.0, " a"), (1.0, 2.0, " b"), (2.0, 3.0, " c")])
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=backend, chunk=True, cancel=stop,
                        on_segment=stop_after_the_first)
    assert ei.value.code == "cancelled"
    assert seen == ["a"]                                    # stopped before the second segment
    assert not list((tmp_path / "chunks").glob("*.json"))   # a cancelled chunk leaves no checkpoint


def test_silence_still_reaches_the_total():
    events, partials = [], []
    core.transcribe(_zeros(10.0), backend=FakeBackend(segments=()), chunk=False,
                    on_progress=events.append, on_segment=partials.append)
    assert [(e.done, e.total) for e in _steps(events)] == [(10.0, 10.0)]
    assert partials == []


def test_a_phantom_past_the_chunk_end_neither_overshoots_nor_previews():
    events, partials = [], []
    backend = FakeBackend([(1.0, 2.0, " real"), (9.9, 39.9, " Thanks for watching.")])
    t = core.transcribe(_zeros(10.0), backend=backend, chunk=False,
                        on_progress=events.append, on_segment=partials.append)
    assert [p.text for p in partials] == ["real"]
    assert max(e.done for e in _steps(events)) == 10.0
    assert [s.text for s in t.segments] == [" real"]


def test_cached_chunks_count_as_done_and_preview_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(600.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0)])
    backend = FakeBackend()
    request, _ = core._apply_caps(backend, core._request(language="auto", vad=False, hotwords=(),
                                                         beam_size=5, word_timestamps=False))
    chunked.chunk_path(core._identity(audio, backend, request), 0.0, 600.0).write_text(
        json.dumps({"language": "es", "segments": [{"start": 1.0, "end": 2.0, "text": " cache"}]}),
        encoding="utf-8")
    events, partials = [], []
    core.transcribe(audio, backend=backend, chunk=True, on_progress=events.append,
                    on_segment=partials.append)
    assert [(e.done, e.total) for e in _steps(events)] == [(600.0, 600.0)]
    assert _steps(events)[0].detail.endswith("(cache)")
    assert partials == [] and backend.calls == []
```

- [ ] **Step 4: comprobar que fallan por la razón esperada**

Run: `.venv/bin/python -m pytest tests/test_transcribe.py -q`
Expected: FAIL. Las pruebas que pasan `on_segment` a `transcribe()` fallan con `TypeError: transcribe() got an unexpected keyword argument 'on_segment'`; las de progreso, porque `done` sigue contando trozos (`1`, no `10.0`) y el detalle acaba en `(nuevo)`.

- [ ] **Step 5: implementar en `src/speechtotext/core/transcribe.py`**

1. Imports: cambiar `from speechtotext.asr.base import AsrBackend, AsrError` por `from speechtotext.asr.base import AsrBackend, AsrError, raise_if_cancelled` y `from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult` por `from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult, TranscriptionSegment`.
2. Justo después de `ProgressCallback = Callable[[Progress], None]`, añadir:

```python
@dataclass(frozen=True)
class PartialSegment:
    """A segment the engine has just produced, in seconds of the whole recording and before
    diarization: no speaker yet, and the text trimmed. For a live preview; the Transcript is
    the result."""
    start: float
    end: float
    text: str


SegmentCallback = Callable[[PartialSegment], None]
```

3. Justo después de `def _mmss(...)`, añadir:

```python
class _Meter:
    """Seconds of audio transcribed so far, summed over the chunks running in parallel.
    Workers report from their own threads; the lock hands the callbacks one event at a time,
    in order, so `done` never goes back."""

    def __init__(self, total: float, emit: ProgressCallback,
                 on_segment: SegmentCallback | None) -> None:
        self._total = total
        self._emit = emit
        self._on_segment = on_segment
        self._lock = threading.Lock()
        self._done: dict[int, float] = {}

    def hook(self, span: int, start: float, end: float) -> Callable[[TranscriptionSegment], None]:
        """The engine callback for one chunk: local times in, global progress and text out."""
        length = end - start
        detail = f"{_mmss(start)}-{_mmss(end)}"

        def on_local(segment: TranscriptionSegment) -> None:
            if segment.end - length > length - segment.start:
                return   # more padding than audio: clip_to_end drops it, so nobody sees it
            reached = min(segment.end, length)
            text = segment.text.strip()
            with self._lock:
                if self._on_segment is not None and text:
                    self._on_segment(PartialSegment(round(start + segment.start, 3),
                                                    round(start + reached, 3), text))
                self._advance(span, reached, detail)

        return on_local

    def finish(self, span: int, length: float, detail: str) -> None:
        with self._lock:
            self._advance(span, length, detail)

    def _advance(self, span: int, seconds: float, detail: str) -> None:
        self._done[span] = max(self._done.get(span, 0.0), seconds)
        done = min(self._total, round(sum(self._done.values()), 2))
        self._emit(Progress("transcribe", done, self._total, detail))
```

4. Sustituir `_run_span` por:

```python
def _run_span(backend, samples, request, start, end, identity, cancel, on_segment=None):
    """(global segments, from_cache, language, probability, engine warnings). An old
    checkpoint can contain a phantom over the padding: clip_to_end also runs on read."""
    raise_if_cancelled(cancel)
    path = chunk_path(identity, start, end) if identity is not None else None
    if path is not None and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            segs = clip_to_end([seg_from_dict(d) for d in data["segments"]], end)
            return segs, True, data.get("language"), None, ()
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt checkpoint -> recompute
    a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    result = backend.transcribe(samples[a:b], request, on_segment=on_segment, cancel=cancel)
    segs = clip_to_end(shift_segments(_timed(result), start), end)
    if path is not None:
        path.write_text(json.dumps({"language": result.language,
                                    "segments": [seg_to_dict(s) for s in segs]},
                                   ensure_ascii=False), encoding="utf-8")
    return segs, False, result.language, result.native_signals.language_probability, result.warnings
```

5. En la firma de `transcribe()`, añadir `on_segment: SegmentCallback | None = None,` entre `on_progress` y `cancel`, y añadir al final de su docstring:

```python
    During the transcribe stage `on_progress` counts seconds of audio and `on_segment` gets
    each PartialSegment as the engine produces it. Both may run on worker threads, one call
    at a time."""
```

6. En el cuerpo de `transcribe()`: borrar la función interna `check_cancel` y sustituir sus dos llamadas, `check_cancel()`, por `raise_if_cancelled(cancel)`.
7. Sustituir el bloque `with ThreadPoolExecutor(max_workers=workers) as pool:` entero por:

```python
    meter = _Meter(duration, emit, on_segment)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_span, backend, samples, eff, s, e, identity, cancel,
                            meter.hook(i, s, e)): i
                for i, (s, e) in enumerate(spans)}
        for fut in as_completed(futs):
            i = futs[fut]
            s, e = spans[i]
            try:
                results[i], cached, langs[i], probs[i], span_warnings = fut.result()
            except RuntimeError as exc:   # AsrError is also a RuntimeError
                # On the first failure, pending work is canceled: with a broken engine and SLOW
                # failures (paging, timeout), draining 17 chunks would take hours.
                pool.shutdown(wait=False, cancel_futures=True)
                if isinstance(exc, AsrError) and exc.code != "backend_failed":
                    raise                                   # cancelled, unsupported_option: as-is
                translated = _oom(exc, backend.backend_id)  # the wrapped message preserves the allocator text
                if translated is not exc:
                    raise translated from exc
                if len(spans) == 1:
                    raise
                raise AsrError("backend_failed", True,
                               f"engine {backend.backend_id} failed on chunk {i + 1}/{len(spans)} "
                               f"({_mmss(s)}-{_mmss(e)}): {exc}") from exc
            for w in span_warnings:
                warning = f"{backend.backend_id}: {w}"
                if warning not in extra:
                    extra.append(warning)
            span = e - s
            cov = 100 * sum(x.end - x.start for x in results[i]) / span if span > 0 else 0.0
            meter.finish(i, span, f"{_mmss(s)}-{_mmss(e)} {cov:.0f}% "
                                  f"({'cache' if cached else 'new'})")
```

- [ ] **Step 6: comprobar que pasan**

Run: `.venv/bin/python -m pytest tests/test_transcribe.py -q`
Expected: todo PASS.

- [ ] **Step 7: el CLI, prueba primero**

En `tests/test_cli.py`, dentro de `_fake_transcribe`, sustituir el método `transcribe` de su `FakeBackend` por:

```python
        def transcribe(self, samples, request, *, on_segment=None, cancel=None):
            if calls is not None:
                calls.append((samples, request))
            if boom is not None:
                raise boom
            result = _result(segments, info)
            for segment in result.segments:
                if on_segment is not None:
                    on_segment(segment)
            return result
```

Sustituir `test_chunking_is_announced_and_each_chunk_is_listed_outside_a_tty` por:

```python
def test_chunking_is_announced_and_progress_is_listed_in_audio_minutes_outside_a_tty(tmp_path, monkeypatch):
    from speechtotext.core import transcribe as core_transcribe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(1.0, 2.0)], _info(1200.0))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: True)
    monkeypatch.setattr(core_transcribe, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = _invoke(audio, tmp_path, "-j", "2")
    assert result.exit_code == 0, result.stdout
    output = _flattened(result.stdout)
    assert "Chunked (jobs=2)" in output
    # The chunks run in parallel: the first line can read 10:02, the other chunk's first
    # segment already counted. What is fixed: at least one line before the end, and the end.
    assert len(re.findall(r"\d\d:\d\d / 20:00", output)) >= 2 and "20:00 / 20:00" in output
    assert "(new)" in output and "[1/2]" not in output


def test_redirected_progress_prints_one_line_per_minute_of_audio(tmp_path, monkeypatch):
    segments = [_seg(float(s), float(s + 10)) for s in range(0, 300, 10)]   # 30 segments, 5 min
    audio = _fake_transcribe(monkeypatch, tmp_path, segments, _info(300.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0, result.stdout
    printed = re.findall(r"(\d\d:\d\d) / 05:00", _flattened(result.stdout))
    assert printed == ["01:00", "02:00", "03:00", "04:00", "05:00"]
```

Run: `.venv/bin/python -m pytest tests/test_cli.py -q -k "chunking or redirected"`
Expected: FAIL. Hoy el CLI imprime `[done/total]` para cualquier evento con `total > 1`, así que salen líneas como `[2/1200]`, y el aviso "Chunked" depende de ese mismo evento.

- [ ] **Step 8: el CLI, implementación**

En `src/speechtotext/cli/app.py`, añadir `from speechtotext.core import transcribe as core_module` debajo de `from speechtotext.core import probe as core_probe`. En `transcribe_file`, sustituir desde `notified = False` hasta el final de la función `on_progress` por:

```python
        printed_at = 0.0

        def on_progress(p):
            nonlocal printed_at
            if p.stage == "decode" and p.total:
                dur_min = p.total / 60
                if route.eta_factor:
                    eta_min = max(1, round(dur_min * route.eta_factor))
                    source = "estimated" if route.estimated else "measured with bench"
                    console.print(f"Duration {dur_min:.1f} min · ETA ~{eta_min} min ({source})")
                else:
                    console.print(f"Duration {dur_min:.1f} min · ETA not measured for this route")
                # The same decision the core makes, on the same inputs.
                if core_module.should_chunk(p.total, chunk):
                    console.print(f"[bold]Chunked[/bold] (jobs={jobs}) · {model}")
                return
            if p.stage == "transcribe" and p.total:
                line = f"{_fmt(p.done)} / {_fmt(p.total)} {p.detail}".rstrip()
                if console.is_terminal:
                    progress.update(task, description=line, total=p.total, completed=p.done)
                elif p.done - printed_at >= 60 or p.done >= p.total > printed_at:
                    # Redirected to a file, Live does not refresh. One line per minute of audio:
                    # hours of silence is the failure the 2026-07-08 spec names; a line per
                    # segment is the opposite one.
                    printed_at = p.done
                    console.print(f"  {line}", markup=False)
            else:
                progress.update(task, description=f"{p.stage} {p.detail}".strip())
```

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: todo PASS.

- [ ] **Step 9: el contrato, prueba primero**

En `tests/test_api_contract.py`, añadir `"speechtotext.core.transcribe.PartialSegment",` a `CONTRACT`, justo después de `"speechtotext.core.transcribe.Progress",`.

Run: `.venv/bin/python -m pytest tests/test_api_contract.py -q`
Expected: FAIL en `test_every_documented_name_exists_and_imports[speechtotext.core.transcribe.PartialSegment]`: `docs/api.md` todavía no lo nombra.

- [ ] **Step 10: documentar**

En `docs/api.md`, sección "`transcribe()`":

1. En el bloque de ejemplo, cambiar la línea de import por `from speechtotext.core.transcribe import transcribe, Transcript, Progress, PartialSegment, AsrError`.
2. Sustituir el final de la frase larga del primer párrafo, desde `; \`cancel\` is a` hasta `checked between chunks.`, por un punto, y añadir a continuación este párrafo:

```markdown
During `transcribe`, `done` and `total` are seconds of audio: `done` grows as
the engine finishes each segment, summed across the chunks that run in
parallel, and never goes back; `total` is the duration. `on_segment` receives
each `PartialSegment(start, end, text)` as the engine produces it, in seconds
of the whole recording, trimmed and with no speaker yet: a live preview, not
the result. Audio read from a checkpoint produces none. Both callbacks may be
called from worker threads, never two at once. `cancel` is a
`threading.Event` checked between segments: once it is set, the call raises
`AsrError("cancelled")` at the end of the 30-second window the engine is
decoding.
```

En `CHANGELOG.md`, sección "Unreleased":

- Al principio de "### Added":

```markdown
- `transcribe(on_segment=...)` receives each `PartialSegment(start, end, text)`
  as the engine produces it, in seconds of the whole recording and before
  diarization: a live preview. Audio read from a checkpoint produces none.
- **New contract**: `core.transcribe.PartialSegment`.
```

- Al principio de "### Changed":

```markdown
- The CLI shows progress in minutes of audio, `mm:ss / mm:ss`. Redirected to a
  file, it prints one line per minute of audio instead of one per chunk.
```

- Al principio de "### Changed — breaking":

```markdown
- `Progress("transcribe")` counts seconds of audio, not chunks. `done` grows
  segment by segment, summed across the chunks running in parallel, and
  `total` is the duration; before, a file under 20 minutes jumped from 0 to 1.
  A finished chunk's `detail` ends in `(new)` instead of `(nuevo)`. <!-- # spanish-is-data: the old detail string being replaced -->
- `AsrBackend.transcribe` takes two keyword arguments, `on_segment` and
  `cancel`, and `transcribe()` passes both: a backend of your own has to accept
  them. The two built-in engines honor them between segments, so `cancel` now
  stops `transcribe()` inside a chunk, not only between chunks. Callbacks run
  on worker threads, one at a time.
```

Run: `.venv/bin/python -m pytest tests/test_api_contract.py tests/test_public_tree.py -q`
Expected: PASS.

- [ ] **Step 11: la suite entera y commit**

Run: `.venv/bin/python -m pytest -q`
Expected: todo PASS.

```bash
git add src/speechtotext/core/transcribe.py src/speechtotext/cli/app.py tests/test_transcribe.py tests/test_cli.py tests/test_api_contract.py docs/api.md CHANGELOG.md
git commit -m "feat(core): progress in seconds of audio, cancel within a chunk, PartialSegment"
```

---

### Task 4: comprobación con los motores reales y PR

Sin cambios de código. Confirma en el desktop, con los dos motores de verdad, lo que las pruebas afirman con falsos, y deja las cifras en el PR.

**Files:** ninguno del repo. Un script desechable en `D:\bench-diar\verify_progress.py`.

**Interfaces:**
- Consumes: todo lo anterior, a través de `speechtotext.core.transcribe.transcribe`.

- [ ] **Step 1: subir la rama**

```bash
git push -u origin feat/core-progress-per-fragment
```

- [ ] **Step 2: un clon aparte en el desktop, sin tocar la copia de trabajo del usuario**

```powershell
git clone --branch feat/core-progress-per-fragment https://github.com/sssamuelll/speechtotext D:\bench-diar\stt-verify
$env:PYTHONPATH = "D:\bench-diar\stt-verify\src"
Set-Location D:\bench-diar\stt-verify
D:\Desktop\projects\speechtotext\.venv\Scripts\python.exe -m pytest -q
```

Expected: todo PASS, incluidas las pruebas que en el Mac se saltan por falta de torch, transformers o el SDK de MCP.

- [ ] **Step 3: el script de comprobación**

Guardar en `D:\bench-diar\verify_progress.py`. Solo imprime cuentas y tiempos: el audio es privado y su texto no sale de la máquina.

```python
"""Real-engine check of progress, cancel and partials. Prints counts and times only."""
import threading
import time

from speechtotext.asr.base import AsrError
from speechtotext.core.transcribe import transcribe

WAV = r"D:\bench-diar\e2e10.wav"   # 10 minutes: a single chunk, the case that used to jump 0 -> 1


def run(engine, model, device, cancel_after=None):
    events, partials, stop = [], [], threading.Event()
    t0 = time.monotonic()
    if cancel_after is not None:
        threading.Timer(cancel_after, stop.set).start()
    try:
        transcribe(WAV, model=model, engine=engine, device=device, chunk=False,
                   on_progress=lambda p: events.append((time.monotonic() - t0, p)),
                   on_segment=lambda s: partials.append(time.monotonic() - t0),
                   cancel=stop)
        outcome = "finished"
    except AsrError as exc:
        outcome = f"AsrError({exc.code}) {time.monotonic() - t0 - (cancel_after or 0):.1f} s after cancel"
    steps = [(t, p.done, p.total) for t, p in events if p.stage == "transcribe"]
    gaps = [b[0] - a[0] for a, b in zip(steps, steps[1:])]
    print(f"{engine}/{model}/{device}: {outcome}; events={len(steps)} partials={len(partials)} "
          f"monotonic={all(a[1] <= b[1] for a, b in zip(steps, steps[1:]))} "
          f"last={steps[-1][1:] if steps else None} max_gap={max(gaps, default=0):.1f}s")


run("faster-whisper", "small", "cpu")
run("whispercpp", "large-v3", "cuda")
run("faster-whisper", "small", "cpu", cancel_after=20)
run("whispercpp", "large-v3", "cuda", cancel_after=20)
```

Run (misma sesión de PowerShell, con `PYTHONPATH` puesto): `D:\Desktop\projects\speechtotext\.venv\Scripts\python.exe D:\bench-diar\verify_progress.py`

Expected, por motor:
- Sin cancelar: `finished`, `events` > 1, `partials` > 0, `monotonic=True`, `last` igual a `(duración, duración)`.
- Cancelando: `AsrError(cancelled)`, con un retraso tras la orden no mayor que lo que tarda una ventana de 30 s en ese motor.

Anotar las cuatro líneas: van al PR.

- [ ] **Step 4: el CLI con la salida redirigida**

```powershell
D:\Desktop\projects\speechtotext\.venv\Scripts\python.exe -c "from speechtotext.cli.app import app; app()" transcribe D:\bench-diar\e2e10.wav -m small --engine faster-whisper -d cpu -f json -o D:\bench-diar\verify-out > D:\bench-diar\verify-cli.log
Select-String -Path D:\bench-diar\verify-cli.log -Pattern " / 10:00" | Measure-Object | Select-Object -ExpandProperty Count
```

Expected: unas 10 líneas de progreso, una por minuto de audio, en lugar de una por segmento o ninguna. Borrar después `D:\bench-diar\verify-out*`: contiene texto transcrito.

- [ ] **Step 5: abrir el PR**

Escribir el cuerpo en `"${TMPDIR:-/tmp}/pr-core-progress.md"`, pegando en el bloque de código, tal cual, las cuatro líneas que imprimió el paso 3 y, en la última viñeta, el número del paso 4. Después, añadir al final las líneas de atribución de la sesión:

````markdown
## Summary

- `AsrBackend.transcribe` takes `on_segment` and `cancel`. faster-whisper reads its generator one segment at a time; whisper.cpp reads whisper-cli's live output and ends the process on cancel.
- `transcribe()` reports `Progress("transcribe")` in seconds of audio, within each chunk and summed across parallel chunks, and hands each `PartialSegment` to `on_segment`. `cancel` now stops inside a chunk.
- The CLI shows `mm:ss / mm:ss`, and one line per minute of audio when redirected.

Spec: `docs/superpowers/specs/2026-09-24-app-escritorio-cimientos-design.md`, §6.4.

## Real engines (desktop: Ryzen 9 5900X, GTX 980; a private 10-minute recording, counts only)

```
(the four lines printed by verify_progress.py)
```

- CLI redirected to a file: (count) progress lines for 10 minutes of audio.

## Test plan

- [x] Full suite on macOS (no torch) and on the desktop (torch, transformers, mcp).
- [x] Both engines, real models: progress within the single chunk, partials, monotonic `done`, cancel.
- [ ] CI: the 7 jobs.
````

```bash
gh pr create --base main --head feat/core-progress-per-fragment \
  --title "feat(core): progress, cancel and text per fragment" --body-file "${TMPDIR:-/tmp}/pr-core-progress.md"
```

Esperar a CI (los 7 jobs) y dejarlo abierto: el merge lo decide el usuario.
