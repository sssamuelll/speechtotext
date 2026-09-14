# speechtotext solo — diseño del lanzamiento público

**Fecha:** 2026-09-12 · **Estado:** aprobado en brainstorming, pendiente de plan
**Alcance:** este repo. La app de escritorio es un repo aparte que consume por tag y queda fuera.
**Idioma de este documento:** español — es documento de trabajo y sale purgado del árbol público con todo `docs/superpowers/` (§9.5).

---

## 1. Objetivo

speechtotext se publica como proyecto independiente, en un repo nuevo de GitHub y en PyPI, con historia completa desde 2021, en inglés, con CI en tres sistemas. Meta declarada: un repo que compite por miles de estrellas por sí solo, con historial impecable, y que no nombra ni depende de ningún consumidor privado.

La propuesta de valor, medida y no inventada (README, "Lo medido"): **una grabación entra y sale un texto del que se pueden abrir issues sin volver al vídeo, 100 % local.** El origen del proyecto —un script para transcribir entrevistas que no podían salir de la máquina— es la razón de ser del "100 % local" y se cuenta en el README **sin identificar a nadie** (§9.7).

## 2. Decisiones cerradas

| Decisión | Elegido | Descartado y por qué |
|---|---|---|
| Para quién | Cuatro personas: A terminal, B `import`, C no técnica (desktop), D agentes (MCP) | — |
| Descomposición | (1) núcleo = este repo; (2) MCP = subcomando aquí, si cabe; (3) desktop = repo aparte por tag | Desktop en el mismo repo: cargaría el repo público con un Electron/Tauri y su ciclo de releases |
| El 58 % privado (`evaluation`, `security`, `models`, `confidence`) | Se muda a aurelius, precedente `api/`→klara en v0.4.0 | Paquete privado aparte (tercer repo para un consumidor); quedarse con razón pública (Windows-only, ACLs, peso muerto) |
| Default | `large-v3`, sin VAD, sin hotwords, reloj dicho de frente | `small` por default: cambia lo que se dijo (17× más errores, frases invertidas) |
| Sondeo de la máquina | Sí, barato, antes de cargar; nunca cambia el modelo en silencio | — |
| whisper.cpp | Primera clase: descarga con checksum donde hay release, `PATH`/brew donde no | "Trae tu binario": la persona C no puede |
| Plataformas | Windows + macOS + Linux con CI en los tres | Solo Windows: se lee como proyecto personal |
| Historia | Completa desde 2021, reescrita con `filter-repo` | Empezar de cero o desde v0.4.0: Samuel quiere contar la historia entera |
| Forma del núcleo | Un solo backend (`asr/`) debajo de todo | Dos entradas coexistiendo (la trampa sigue); borrar `asr/` (tira una capa validada) |
| Idioma | Inglés, todo, con solace-wren para la prosa | — |
| Nombre | Nuevo, sesión aparte con solace-wren; `speechtotext` y `speech-to-text` están tomados en PyPI | Mantener el nombre: imposible en PyPI |
| Repo | Nuevo en GitHub; este queda privado como archivo | Publicar este: los PR/issues no se reescriben (PR #21 se llama "spec-jarvis-voice-core" para siempre) |

## 3. Frontera del producto

**Regla:** se queda lo que hace falta para "archivo entra, transcripción sale", o lo que mide el audio sin emitir veredicto. Lo demás se va.

| Se queda (público) | Se va a aurelius | Cambia de forma |
|---|---|---|
| `core/` (transcribe, chunked, formats, segments, finder, benchmark, enginepin, probe, models), `cli/`, `audio/` entero, `speakers/`, `asr/` | `evaluation/`, `security/`, `models/`, `confidence/`, `statistics.py`, `docs/audio-evaluation.md`, sus tests (`test_evaluation_*`, `test_private_artifacts*`, `test_model_*`, `test_calibration*`, `test_confidence_*`, `test_environment`, `test_statistics_*`) | `asr/FasterWhisperBackend(model: str \| Path, ...)`: recibe nombre o ruta, no `VerifiedModelArtifact`. Los Protocols `CalibratedAsrBackend` y `VerifiedLocalAsrBackend` se van con lo verificado |
| | | `TranscriptionResult` pierde `confidence_target`, `calibrated_confidence`, `calibrator_version`. aurelius envuelve el resultado público con su calibración |
| | | `core/engines.py` desaparece: `CAPS`, `quant_for`, `effective_opts` y el adaptador de subprocess pasan a `asr/whispercpp.py` |

`audio/` se queda entero (1 057 líneas): `compute_voice_evidence` ya es contrato documentado, y `AudioClip`/`AudioView`/proveniencia siguen siendo la entrada por clip para la persona B. El backend, sin embargo, ya no exige `AudioClip` (§4.1).

**Costo declarado:** aurelius sube el pin una vez con cambios en su contract test — tipos sin calibración, backend sin artefacto, imports de lo que absorbió. Un PR allá, uno acá, mismo día. Ninguna feature de aurelius se pierde: el código se muda completo con sus tests.

## 4. El núcleo

### 4.1 El backend — `asr/`

```python
Cap = Literal["honrado", "degradado", "rechazado"]

@dataclass(frozen=True)
class Caps:
    hotwords: Cap
    vad: Cap
    word_timestamps: Cap

@runtime_checkable
class AsrBackend(Protocol):
    backend_id: str          # "faster-whisper" | "whispercpp"
    model_id: str            # "large-v3"
    caps: Caps
    def warm(self, on_progress: ProgressCallback | None = None) -> None: ...
    def transcribe(self, samples: np.ndarray, request: TranscriptionRequest) -> TranscriptionResult: ...
```

- **Entrada:** `samples` float32 mono a **16 000 Hz**, nada más. Quien llama resamplea. Es exactamente lo que ambos caminos ya hacían por debajo (el wav temporal 16k mono de `chunked`, y `clip.view("asr").samples` del backend).
- **`warm()`** carga el modelo (y lo descarga si falta, reportando progreso). El objeto es la caché: no hay singleton ni registro global. Quien quiera el modelo caliente entre llamadas conserva el objeto.
- **`FasterWhisperBackend(model, device="cpu", compute_type="int8", cpu_threads=0, num_workers=1)`** — lo de hoy sin `VerifiedModelArtifact`. `model` es un nombre (`"large-v3"`, resuelto por faster-whisper vía HF Hub) o una ruta a un directorio CTranslate2. Se mantiene el seam `model_factory` para tests.
- **`WhisperCppBackend(model, device="cuda")`** — nuevo. `warm()` = `enginepin.ensure_engine()` + `ensure_model(model)`. `transcribe()` escribe un wav temporal con `wave` (stdlib), corre `whisper-cli` con los flags de hoy (`-mc 0`, `-bs`, `-l`, `-np -ojf`), parsea con `_parse_ojf`. Señales nativas que whisper.cpp no emite: `None`, nunca fabricadas. `Caps(hotwords="rechazado", vad="degradado", word_timestamps="degradado")`.
- **`TranscriptionRequest`** gana `vad: bool = False`. `language` admite `"auto"` (→ `None` hacia el motor). Lo demás igual: `hotwords`, `word_timestamps`, `beam_size`, `context`.
- **`TranscriptionResult`**: `text, language, words, segments, backend, model, model_version, latency_ms, native_signals, warnings`. Sin campos de calibración.
- **Validación de CAPS** ocurre en `core/transcribe` antes de construir nada: `rechazado` con el knob activo → `AsrError("unsupported_option", recoverable=False)`; `degradado` → warning en el `Transcript`. Jamás silencio, jamás sustitución callada (regla madre heredada de `engines.py`).

### 4.2 Una decodificación — `audio/io.py`

`core/transcribe` abre el archivo **una vez** con PyAV (dependencia existente) y obtiene `samples` 16 kHz mono float32 + duración. Hoy el mismo audio se decodifica hasta tres veces (duración por `probe_duration`, cada trozo por `ffmpeg -ss/-t`, y entero otra vez para diarizar). Después:

- Los trozos son *slices* del array: `samples[int(s*16000):int(e*16000)]`.
- La diarización recibe las mismas muestras: `speakers.diarize(samples, sample_rate, num_speakers)` — `_load_waveform` ya construye `{"waveform", "sample_rate"}` para pyannote; solo cambia la firma.
- `plan_chunks` sigue usando `ffmpeg silencedetect` sobre el archivo para elegir cortes: funciona, tiene tests, y ffmpeg decodifica en segundos contra minutos de inferencia. Pasarlo a energía sobre las muestras es follow-up (§10).

### 4.3 Una función — `core/transcribe.py`

```python
def transcribe(
    audio: Path | np.ndarray,            # archivo, o muestras 16 kHz mono ya decodificadas
    *,
    model: str = "large-v3",
    language: str = "auto",
    engine: Literal["auto", "faster-whisper", "whispercpp"] = "auto",
    device: Literal["auto", "cpu", "cuda"] = "auto",
    compute_type: str = "auto",
    vad: bool = False,
    beam_size: int = 5,
    hotwords: tuple[str, ...] = (),
    word_timestamps: bool = False,
    diarize: bool = False,
    speakers: int | None = None,
    identify: bool = True,
    threshold: float = 0.5,
    chunk: bool | None = None,           # None = automático por encima de 20 min
    jobs: int = 4,
    backend: AsrBackend | None = None,   # reutilizar un modelo caliente
    on_progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> Transcript
```

**Flujo, un solo camino:** sondeo (§5.1, salvo `backend=` dado) → decodificar → `plan_chunks` si toca (si no, un trozo `[0, duración]`) → por trozo: checkpoint si existe, si no `backend.transcribe(slice, request)` → `shift_segments` + `clip_to_end` → reensamblar → huecos, `speech_s`, `is_suspect` → diarizar e identificar si se pidió → `Transcript`.

El archivo corto y el largo son **el mismo camino con n = 1**: desaparece la doble orquestación de `cli/app.py`. Paralelismo por `ThreadPoolExecutor(jobs)` para faster-whisper; `jobs = 1` estructural para whispercpp (N subprocesos contra una GPU paginan, medido). Checkpoints en `~/<app>/chunks` con la misma llave por contenido de hoy (archivo, modelo, motor, cuantización, device, opts efectivos).

```python
@dataclass
class Transcript:
    segments: list[LabeledSegment]        # start, end, text, words?, speaker?, suspect, señales nativas
    language: str
    language_probability: float | None
    duration: float
    speech_s: float
    gaps: list[list[float]]
    engine: EngineInfo                    # engine, model, quant, device, backend_id
    route_reason: str                     # por qué el sondeo eligió esto (vacío si backend= dado)
    warnings: tuple[str, ...]             # degradaciones de CAPS, idioma con baja probabilidad, etc.

@dataclass(frozen=True)
class Progress:
    stage: Literal["probe", "download", "load", "decode", "transcribe", "diarize"]
    done: float
    total: float | None                   # None = indeterminado
    detail: str                           # "trozo 3/7", "large-v3 · 2,9 GB", ...

ProgressCallback = Callable[[Progress], None]
```

- **El núcleo nunca imprime.** El CLI pinta con rich desde el callback; el MCP lo ignora; la desktop dibuja una barra. Si un mensaje no puede existir en la desktop, no existe en el CLI.
- **`cancel`** se consulta entre trozos y antes de diarizar; al dispararse, `transcribe` levanta `AsrError("cancelled", recoverable=True)` sin dejar temporales.
- **Errores:** `AsrError(code, recoverable, message)` es el error del núcleo. Códigos: `unsupported_option`, `model_integrity`, `backend_failed`, `out_of_memory` (el `RuntimeError` del allocator que hoy caza el CLI, con el consejo dentro del mensaje), `cancelled`, `insufficient_resources` (§5.1). `AudioDecodeError` para archivos que no se pueden abrir.
- **Los escritores** (`write_txt/srt/vtt/json`) reciben un `Transcript`. El esquema JSON no cambia respecto a `docs/api.md` salvo el bloque `engine`, que gana `backend_id` y `route_reason`.

## 5. Sondeo, defaults y modelos

### 5.1 Sondeo — `core/probe.py`

Reutiliza lo que `core/benchmark.py` ya tiene (`_nvidia_smi`, `_ram_gb`, `machine_info`, el exe pinneado); `bench` pasa a importar de aquí.

```python
@dataclass(frozen=True)
class Machine:
    platform: str; cpu_count: int; ram_gb: float | None
    cuda: bool; gpu_name: str | None; vram_free_gb: float | None
    whispercpp: Path | None               # binario ya instalado/verificado, o None

@dataclass(frozen=True)
class Route:
    engine: str; device: str; compute_type: str
    reason: str                           # una frase, para imprimir
    eta_factor: float | None              # × duración del audio; None = no medido
    estimated: bool                       # True si el factor sale de la tabla y no de bench.json

def machine() -> Machine                  # < 1 s, sin cargar modelos
def choose_route(m: Machine, model: str, *, engine="auto", device="auto", compute_type="auto") -> Route
```

**Tabla (para `engine="auto"`):**

| Condición | Ruta | Razón impresa |
|---|---|---|
| CUDA y `vram_free_gb ≥ VRAM_FW_GB` (5) | faster-whisper · cuda · float16 | "GPU con N GB libres" |
| CUDA y `VRAM_WCPP_GB` (2) ≤ libre < 5, y whisper.cpp disponible o descargable (Windows) | whispercpp · cuda · q5_0 | "GPU con N GB libres: whisper.cpp cuantizado" |
| lo demás | faster-whisper · cpu · int8 | "sin GPU utilizable: CPU" |

Reglas duras:
- **El sondeo nunca cambia el modelo.** Si `ram_gb < RAM_MIN_GB[model]` corta con `AsrError("insufficient_resources")` cuyo mensaje sugiere `-m small`. No degrada callado.
- `engine`/`device` explícitos se respetan; el sondeo solo rellena lo que falta (`compute_type="auto"` → int8 en CPU, float16 en CUDA, q5_0 en whispercpp — regla existente).
- Los umbrales `VRAM_FW_GB = 5`, `VRAM_WCPP_GB = 2` y `RAM_MIN_GB = {"large-v3": 6, "small": 3}` (techo sobre el pico medido de 3,6 GB en int8) son constantes con comentario `ponytail:` — medidas en una máquina (5900X + GTX 980), no leyes. `bench` es quien las afina.
- **ETA:** `eta_factor` sale de `bench.json` si esta máquina ya se midió para esa ruta; si no, de una tabla corta de factores medidos aquí (CPU int8 large-v3 ≈ 1,3× tiempo real; whisper.cpp CUDA ≈ 0,13×; faster-whisper CUDA: sin medir → `None`) con `estimated=True`. El CLI imprime "~11 min (estimado)" antes de empezar.

### 5.2 Defaults del CLI

`-m large-v3` · `--no-vad` · `--engine auto` · `-d auto` · `--compute-type auto` · `-l auto` · `--chunk` automático sobre 20 min · `--jobs 4` · sin hotwords · `--beam-size 5`.

`-l auto`: Whisper detecta bien con ≥ 30 s; el CLI imprime idioma y probabilidad, y si la probabilidad es baja (< 0,5) sugiere `-l xx`. `-m small` queda documentado como "borrador rápido". Hotwords: documentados con la medición (apagones en bloque, n = 3) y la recomendación de comparar contra una corrida sin ellos.

### 5.3 Modelos como API — `core/models.py`

```python
@dataclass(frozen=True)
class ModelInfo: engine: str; name: str; path: Path; size_bytes: int; verified: bool

def installed(engine: str | None = None) -> list[ModelInfo]
def ensure(engine: str, name: str, on_progress: ProgressCallback | None = None) -> Path
def remove(engine: str, name: str) -> None
def data_dir() -> Path       # %LOCALAPPDATA%/<app> · ~/Library/Application Support/<app> · ~/.local/share/<app>
```

- faster-whisper: `huggingface_hub.snapshot_download` (dependencia transitiva ya presente) con progreso por callback; `installed()` lista el caché HF filtrando los repos `faster-whisper-*`.
- whispercpp: `enginepin` tal cual, con `install_root()` movido a `data_dir()/whisper-cpp/<version>`.
- Rutas por sistema con la stdlib; ninguna dependencia nueva. `~/.speechtotext` (registro de voces, chunks, bench.json) se renombra con el paquete (§9.1) y se documenta la migración.
- Las descargas son de gigas: el CLI anuncia tamaño antes y pinta barra; nunca en silencio.

### 5.4 whisper.cpp de primera clase — `core/enginepin.py`

Ya existe para Windows (zip cublas pinneado con sha256, GGML pinneados). Se extiende:
- `ensure_engine()` en macOS/Linux: busca `whisper-cli` en `PATH` (incluye brew); si no está, `AsrError("backend_failed")` con instrucciones de instalación. No se intenta compilar ni descargar binarios sin release oficial.
- `ensure_model()` idéntico en los tres sistemas (HF Hub).
- El pin de versión sube solo con commit consciente y re-benchmark (regla existente).

## 6. Superficies

### 6.1 CLI

`transcribe` · `find` · `enroll` / `voices` / `forget` · `bench` · `probe` · `models [pull|rm]` · `mcp`.

- `transcribe` ≈ 100 líneas: parsear flags → `core.transcribe(...)` con callback rich → escribir formatos → resumen (idioma, cobertura, huecos, ruta elegida). Todo lo que imprime nace del `Transcript` o del callback.
- `probe`: imprime `Machine` y la `Route` para `large-v3` — lo que se pide pegar en un issue.
- `models`: lista / descarga / borra sobre §5.3.
- `find` reusa `transcribe(audio=samples[slice], ...)` para extraer regiones.
- Formatos de salida sin cambios: `txt`, `srt`, `vtt`, `json`.

### 6.2 MCP — `speechtotext mcp`

Extra opcional `[mcp]` (SDK oficial `mcp`). Transporte stdio. Cuatro herramientas: `transcribe(path, language?, model?, diarize?)` → texto y ruta del JSON; `find(path, query)`; `voices()`; `probe()`. ~100 líneas envolviendo `core.transcribe` con `on_progress=None`. **Última tarea del plan; si no cabe en el lanzamiento, cae sin dolor.**

### 6.3 Contrato para la persona B — `docs/api.md` (inglés)

Documenta exactamente esto, y nada más es público:
- `transcribe()` y `Transcript`, `Progress`, `AsrError` y sus códigos.
- `AsrBackend`, `Caps`, `FasterWhisperBackend`, `WhisperCppBackend`, `TranscriptionRequest`, `TranscriptionResult` y sus tipos.
- Los 18 símbolos de `audio.__all__` (hoy 2 documentados).
- `speakers`: `registry` (enroll / get_embeddings(model) / remove / voices), `identify.assign_names`, `diarization.diarize`.
- `probe.machine`, `probe.choose_route`, `models.*`.
- El esquema JSON de salida.

Regla existente que ahora se cumple: lo que cambie en `api.md` sale en el CHANGELOG bajo *breaks*.

### 6.4 El mapa se rompe con ruido — `tests/test_api_contract.py`

Para cada módulo público, cada nombre en su `__all__` tiene que aparecer en `docs/api.md` como encabezado o `code span`; si no, el test falla con el nombre. Y al revés: cada símbolo que `api.md` nombra en un bloque de firma tiene que existir e importarse. Hoy fallaría (18 exportados, 2 documentados). Es el mecanismo que impide que el contrato vuelva a pudrirse en silencio.

## 7. Pruebas y CI

- **La suite se parte con el código.** Los ~300 tests de lo que se va viajan a aurelius con sus módulos. Quedan ~350 y llegan: `test_transcribe.py` (un trozo y varios, orden de eventos de progreso, `cancel`, CAPS rechazado/degradado, `backend=` reutilizado), `test_probe.py` (cada fila de la tabla, umbrales, "nunca cambia el modelo", `insufficient_resources`), `test_models.py` (rutas por sistema, `installed`, `ensure` con descarga falsa), `test_whispercpp_backend.py` (adaptador con fixture sintética), `test_api_contract.py`. TDD.
- **Cero habla humana real en fixtures.** `tests/fixtures/whispercpp_ojf.json` se reemplaza por una sintética con la misma forma; el audio de prueba es seno y ruido generados en el test. Las fixtures que hoy dicen "hey Jarvis" o usan "Aurelius" como hotword cambian de texto.
- **CI:** `.github/workflows/tests.yml` activado, matriz `ubuntu-latest × windows-latest × macos-latest` en Python 3.11 y 3.13. Nada en CI descarga modelos ni toca red: los tests que lo necesitan van marcados `@pytest.mark.network` y se saltan; el resto usa los seams existentes (`model_factory`, fakes de `urlopen`/`subprocess`). Un job `package` construye el wheel, lo instala en un venv limpio y corre `<app> --help` y `<app> probe`.
- Gate local: `pytest -q`. La suite entera corre sin GPU, sin red y sin modelos.

## 8. Traducción a inglés

| Qué | Quién | Verificación |
|---|---|---|
| README, `docs/api.md`, `docs/README.md`, `docs/design.md` (nuevo, una página con las decisiones que dieron forma al producto), CHANGELOG, ayuda y mensajes del CLI, etiquetas del gráfico | Claude con solace-wren | lectura completa; `test_api_contract` |
| Comentarios, docstrings, nombres de tests, `ponytail:` y demás anotaciones | delegado (reasonix) con brief: conservar el *por qué*, no aplanar; nada de slop | suite verde; lectura de muestra aleatoria del 10 % |
| Mensajes de commit (121) | tabla generada, aplicada por `filter-repo` (§9.5) | lectura completa de la tabla antes de aplicar |

`docs/superpowers/` (specs y planes ejecutados, en español, con nombre de herramienta) sale del árbol público entero; su racional se condensa en `docs/design.md`. `docs/plan-*.md` y `docs/benchmark-turboscribe.md` se traducen o se condensan en `design.md` — decisión por documento durante la traducción, con `docs/README.md` como índice único.

## 9. Lanzamiento — orden y verificación

1. **Nombre.** Sesión con solace-wren. Prueba: libre en PyPI (`GET /pypi/<n>/json` → 404) y en GitHub. Rename mecánico una sola vez: paquete, `pyproject`, entry point, `data_dir()`, rutas de instalación, textos. La suite es la verificación.
2. **aurelius absorbe el 58 %.** PR allá con módulos y tests; su contract test se adapta (§3). Pinnea el tag público al final del proceso.
3. **Refactor del núcleo** (§4–6), PR por PR en este repo privado, cada uno con gate local verde. Primera versión pública: **`0.6.0`** — la secuencia sigue; `1.0` cuando la API sobreviva un mes de uso ajeno.
4. **Traducción** (§8).
5. **Reescritura de historia**, sobre un clon fresco, una sola vez:
   - Antes: borrar en `origin` las 5 ramas stale pre-squash (`docs/lo-medido`, `docs/registro-al-dia`, `feat/senales-nativas`, `feat/voice-evidence`, `fix/trozos-recortan-a-su-ventana`).
   - `git filter-repo` purga de **toda** la historia: `docs/superpowers/**`, `src/static/audio.wav`, `src/static/texto.txt`, `src/.vscode/**`, `src/avconv.exe`, `src/ffmpeg.exe`, `docs/audio-evaluation.md`.
   - `--replace-text` con tabla: nombres privados (`aurelius`, `jarvis`, `klara`), rutas de máquina (`C:\Users\simon`, `D:\AudioBench\...`, `D:\Desktop\...`, `c:\caminantes`), y el nombre del familiar y la emisora en `docs/benchmark-turboscribe.md`.
   - Mensajes: callback con tabla para los 121 commits (traducción + los 13 con nombres privados o ramas `claude/…-XXXXX`).
   - **Verificación obligatoria antes de publicar:** `git grep -i` de cada palabra prohibida (nombres privados, el político de la entrevista, COPEI, Vielma, caminantes, el familiar, `simon`) sobre `$(git rev-list --all)` = **cero**; `git cat-file -e` de los SHAs de `audio.wav` y `texto.txt` falla; `git fsck` limpio; tamaño del repo sin el wav.
6. **Repo nuevo en GitHub** con el nombre nuevo: push de historia y tags; CI encendido; descripción honesta (no "real-time"); topics; **crear `LICENSE` MIT** — hoy no existe en el árbol aunque `pyproject` lo declara. Este repo queda privado, archivado.
7. **PyPI** por *trusted publishing* desde Actions al taggear — sin tokens. Prueba: `pip install <nombre>` en un venv limpio, `<app> transcribe --help`, `<app> probe`.
8. **README**: la historia de origen en un párrafo, **sin nombres de personas, medio ni ciudad**. Badge de CI. La sección "Lo medido" traducida con el gráfico.
9. aurelius sube el pin al tag público del repo nuevo.

## 10. No bloquea el lanzamiento

- La app de escritorio (repo aparte; consume `transcribe()`, `Progress`, `cancel` y `models.*`, que este diseño deja listos).
- El MCP, si se resbala.
- `plan_chunks` por energía sobre las muestras (hoy ffmpeg silencedetect: funciona).
- Medir `eta_factor` de faster-whisper CUDA (sin GPU de ≥ 5 GB aquí).
- Streaming / micrófono (la entrada por clip existe; el endpointer causal no).

## 11. Riesgos declarados

- **`chunked.py` cambia de costura** (wav temporal → slice de muestras): siete monkeypatches de tests dependen de `chunked.WhisperModel`; se reescriben contra `backend.transcribe`. Riesgo acotado y conocido.
- **`TranscriptionResult` sin calibración rompe a aurelius** una vez; cubierto por su contract test y un PR coordinado.
- **`-l auto` sobre clips cortos** puede detectar mal; mitigado por imprimir probabilidad y sugerir `-l`. Reversible a `es` sin tocar arquitectura.
- **Descarga de whisper.cpp en Windows** pesa (zip con DLLs CUDA); se anuncia con tamaño. En mac/Linux depende de que el usuario lo instale: el mensaje de error lo dice.
- **La reescritura de historia es irreversible en el repo nuevo**; por eso se hace sobre un clon, con la verificación de §9.5 como gate, y este repo queda intacto como archivo.
