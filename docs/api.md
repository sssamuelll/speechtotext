# Contrato para consumidores

Lo que una app que depende de `speechtotext` puede dar por garantizado, y lo que
no. Si algo de este documento cambia, el cambio se anota en
[`CHANGELOG.md`](../CHANGELOG.md) y sale en un tag nuevo — nadie pinnea `@main`.

Lo que no está aquí es implementación: puede cambiar sin aviso.

---

## Esquema del JSON

Lo produce `speechtotext transcribe -f json` y lo escribe
`core.formats.write_json`. Las claves opcionales **se omiten** cuando no hay
valor; ninguna aparece en `null`.

```json
{
  "language": "es",
  "language_probability": 0.9987,
  "duration": 1843.2,
  "speech_s": 1502.7,
  "gaps": [[312.4, 340.1], [905.0, 913.8]],
  "speakers": ["Samuel", "Hablante 2"],
  "engine": {
    "name": "faster-whisper",
    "version": "1.2.0",
    "model": "small",
    "quant": "int8",
    "device": "cpu",
    "selection": "explicit",
    "diarization": "segment"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.42,
      "text": "Hola, ¿cómo estás?",
      "speaker": "Samuel",
      "no_speech": 0.0142,
      "avg_logprob": -0.1877,
      "compression_ratio": 1.2044,
      "suspect": true
    }
  ]
}
```

### Nivel superior

| Clave | Tipo | Presencia |
|---|---|---|
| `language` | `str` | Siempre. |
| `language_probability` | `float` | Omitida si el motor no la reporta (pasa con `--language auto` en la ruta troceada). |
| `duration` | `float` | Siempre. Duración del archivo, en segundos. |
| `speech_s` | `float` | Siempre. Suma de la duración de los segmentos con voz. |
| `gaps` | `[[float, float], …]` | Siempre. Huecos sin voz de 5 s o más, como pares `[inicio, fin]`. |
| `speakers` | `[str, …]` | Solo si la corrida produjo hablantes. |
| `engine` | `object` | Solo si el CLI lo informa. Incluye `diarization: "segment"` o `"word"` cuando se usó `--diarize`. `selection` es `"auto"` si el motor lo eligió el sondeo (`--engine auto`) y `"explicit"` si lo pidió el usuario. |
| `segments` | `[object, …]` | Siempre. |

### Cada segmento

| Clave | Tipo | Presencia |
|---|---|---|
| `id` | `int` | Siempre. |
| `start`, `end` | `float` | Siempre. Segundos. |
| `text` | `str` | Siempre. |
| `speaker` | `str` | Solo con diarización. |
| `no_speech` | `float` | Omitida si el motor no la midió. Redondeada a 4 decimales. |
| `avg_logprob` | `float` | Igual. |
| `compression_ratio` | `float` | Igual. |
| `suspect` | `true` | Solo cuando dispara la heurística. **Nunca aparece como `false`.** |

`whisper.cpp` no emite ninguna de las tres señales nativas: bajo
`--engine whispercpp` esas claves nunca están.

### `suspect`

Marca un segmento para que lo revises, no dictamina que esté mal. Dispara si:

- `no_speech > 0.6`, o
- el segmento dura 10 s o más y tiene **menos de un carácter por segundo**.

El segundo criterio es una heurística de densidad sin calibrar. Trátalo como una
sugerencia de revisión; si necesitas una decisión con precisión medida, usa el
arnés de evaluación de tu consumidor, no este campo.

---

## Capa `audio/`: medidas sobre la señal, sin veredicto

`speechtotext.audio` mide y transforma audio; **nunca decide si transcribir**. Todo lo
que exporta es inmutable y se valida al construirse: un tipo mal formado revienta donde
se crea, no tres capas más abajo.

```python
from speechtotext.audio import decode_audio, AudioDecodeError

with open("reunion.m4a", "rb") as stream:
    view = decode_audio(stream, sample_rate=16000)   # -> AudioView
samples = view.samples                               # float32 mono, de solo lectura
```

### Decodificación

- **`decode_audio(stream, *, sample_rate, av_module=None) -> AudioView`** — entra un
  stream binario *seekable* (no una ruta: quien abre, cierra), sale un `AudioView` mono
  float32 al `sample_rate` pedido. `av_module` es el seam de los tests.
- **`AudioDecodeError(RuntimeError)`** — el archivo no se pudo abrir o no trae audio.

### Un clip y sus vistas

El mismo audio se mide, se identifica y se transcribe con preprocesados distintos.
`AudioClip` los guarda juntos con su proveniencia, para que cada número se pueda rastrear
hasta las muestras de las que salió.

- **`AudioClip(started_at, ended_at, source_id, speech_regions, quality, views)`** —
  `.view(name)` devuelve la vista pedida y lanza `KeyError` si no está o no existe.
  Valida al construirse que las duraciones de las vistas y `quality.duration_ms` cuadren
  con el clip, y que las regiones de habla no se solapen ni se salgan.
- **`AudioViews(capture, analysis, asr, identity=None, spoof=None)`** — las tres primeras
  son obligatorias.
- **`AudioViewName`** — `Literal["capture", "analysis", "identity", "spoof", "asr"]`.
- **`AudioView`** — `samples` (float32 mono, respaldado por `bytes`: no se puede escribir),
  `sample_rate`, `provenance`. **Sin constructor público**: se crea con
  `AudioView.capture(samples, sample_rate, *, step)` o con
  `AudioView.derive(parent, samples, *, sample_rate=None, steps, models=(), thresholds=None)`.
  Propiedades: `duration_s` y `pipeline_fingerprint`.
- **`SpeechRegion(start_s, end_s)`** — ordenable; exige tiempos finitos y `0 <= start_s < end_s`.

### Calidad y ganancia

```python
from speechtotext.audio import apply_fixed_gain, compute_audio_quality

gain = apply_fixed_gain(samples, 6.0)
report = compute_audio_quality(samples, gain.samples, 16000, regions,
                               requested_gain_db=6.0, applied_gain_db=gain.applied_gain_db)
```

- **`compute_audio_quality(capture, processed, sample_rate, speech_regions, *, requested_gain_db, applied_gain_db, dropped_frames=0, discontinuities=0) -> AudioQualityReport`**
- **`AudioQualityReport`** — `duration_ms`, `effective_voice_ms`, `input_rms_dbfs`,
  `processed_rms_dbfs`, `peak_dbfs`, `clipping_ratio`, `noise_floor_dbfs`, `snr_db`,
  `requested_gain_db`, `applied_gain_db`, `dropped_frames`, `discontinuities`, `warnings`.
  `noise_floor_dbfs` es `None` cuando el clip es habla de punta a punta y no queda ni una
  muestra de silencio que medir; `snr_db` es `None` cuando falta cualquiera de los dos
  lados de la resta — sin silencio, o sin ninguna región de habla. **`None` no es cero**:
  la misma regla que gobierna la evidencia de voz.
- **`apply_fixed_gain(samples, gain_db, *, max_gain_db=18.0, peak_limit_dbfs=-1.0) -> GainResult`**
- **`GainResult(samples, requested_gain_db, applied_gain_db, limited)`** — `limited` es
  `True` cuando el limitador tuvo que recortar la ganancia pedida.

### Puerta de calidad

Decide si un clip merece inferencia, contra umbrales que pone quien llama. Los cuatro
umbrales de señal no traen valor por defecto a propósito — medir es de la librería,
decidir es de quien la usa. Los dos contadores de transporte sí lo traen, en `0`: un
fotograma perdido no es aceptable por omisión.

- **`QualityThresholds(min_effective_voice_ms, min_processed_rms_dbfs, min_snr_db, max_clipping_ratio, max_dropped_frames=0, max_discontinuities=0)`**
- **`evaluate_pre_inference(report, thresholds) -> PreInferenceDecision`**
- **`PreInferenceDecision(eligible, reason_codes)`**
- **`QualityReason`** — un `Literal` con ocho códigos:
  `"silence"`, `"too_short"`, `"level_too_low"`, `"snr_unavailable"`, `"snr_too_low"`,
  `"clipping"`, `"dropped_audio"`, `"discontinuous_audio"`. La puerta acumula
  **todas** las razones que aplican, no la primera.

### Proveniencia

La huella de un pipeline de audio: qué pasos, qué modelos, qué umbrales. Sirve para afirmar
que dos resultados salieron del mismo tratamiento sin guardar el audio.

- **`PipelineStep(name, version, parameters)`** — `parameters` tiene que ser JSON
  serializable; se congela al construirse.
- **`ModelRef(model_id, fingerprint)`** — `fingerprint` es un sha256 en hex minúsculas de
  64 caracteres. La proveniencia no sabe de manifiestos ni de sistemas de archivos: quien
  tenga un modelo verificado lo reduce a esto, y así la huella se puede calcular en
  cualquier máquina.
- **`PipelineProvenance`** — `sample_rate`, `parent_fingerprint`, `steps`,
  `model_fingerprints`, `thresholds`. **Sin constructor público**:
  `PipelineProvenance.capture(*, sample_rate, step, models=(), thresholds=None)` y
  `PipelineProvenance.derive(parent, *, sample_rate, steps, models=(), thresholds=None)`.
  Propiedad `fingerprint` (sha256 del payload ordenado), más `to_dict()` y
  `from_dict(data, *, parent, models)`.

---

## Evidencia de voz

`speechtotext.audio.compute_voice_evidence` describe una señal de audio. **No
decide nada**: no clasifica voz contra no-voz, no es un VAD, y no combina sus
propias medidas en un booleano. El umbral y la decisión son de quien llama.

```python
from speechtotext.audio import compute_voice_evidence

evidencia = compute_voice_evidence(ventana, sample_rate=16000)

if evidencia.voice_band_ratio is None:
    ...  # no se pudo medir: la ventana está por debajo de -80 dBFS
elif evidencia.voice_band_ratio >= 0.4:
    ...  # hay energía donde vive la voz
```

### Entrada

`compute_voice_evidence(samples: np.ndarray, sample_rate: int) -> VoiceEvidence`

- `samples`: mono, 1-D, cualquier dtype convertible a `float64`, todo finito.
  Estéreo, `NaN` o `inf` levantan `ValueError`.
- `sample_rate`: entero positivo, y suficientemente alto para resolver F0 en
  70-350 Hz. Una tasa demasiado baja levanta `ValueError`.
- Duración mínima: un tramo de 64 ms. Por debajo de eso devuelve todas las
  medidas en `None` con `frames = 0`, sin error.

### Campos de `VoiceEvidence`

| Campo | Tipo | Qué mide |
|---|---|---|
| `voice_band_ratio` | `float \| None` | Energía entre 300 y 3400 Hz sobre la energía total, ponderada por tramo (FFT con ventana de Hann). La banda telefónica donde vive la inteligibilidad. |
| `voiced_ratio` | `float \| None` | Fracción de **todos** los tramos donde se detectó F0 entre 70 y 350 Hz. |
| `f0_median_hz` | `float \| None` | Mediana de F0 sobre los tramos sonoros. `None` si ninguno lo fue. |
| `spectral_flatness` | `float \| None` | Media geométrica sobre media aritmética del espectro. 0 es tonal puro, 1 es ruido blanco. |
| `frames` | `int` | Tramos analizados. Solo es 0 si el audio dura menos que un tramo. |

Las tres razones van en `[0, 1]`; `f0_median_hz` es positivo.

### `None` no es cero

Las cuatro medidas son `None` **a la vez** cuando ningún tramo supera -80 dBFS:
eso es *no pude medir*, y es distinto de *medí y no hay voz*, que se reporta como
`0.0`. Un consumidor que trate `None` como cero convierte silencio en un veredicto
negativo, y ese es exactamente el error que el módulo existe para no cometer.

### Constantes internas

| Constante | Valor |
|---|---|
| `VOICE_BAND_HZ` | `(300, 3400)` |
| `F0_RANGE_HZ` | `(70, 350)` |
| `FRAME_S` | `0.064` (cuatro periodos de 70 Hz) |
| `HOP_S` | `0.016` |
| `VOICING_THRESHOLD` | `0.5` autocorrelación normalizada |
| `OCTAVE_COST` | `0.05` por octava de lag |
| `SILENCE_RMS` | `1e-4` (-80 dBFS) |

Están expuestas para que puedas razonar sobre las medidas, no para que las
sintonices: cambiarlas cambia el significado de los números que ya guardaste.

### Límites conocidos

- **Un tono puro dentro de banda engaña a dos medidas**: da `voice_band_ratio`
  cercano a 1 y `voiced_ratio` cercano a 1, con un `f0_median_hz` subarmónico
  inventado. Lo delata `spectral_flatness` por debajo de `1e-6`. Si tu entrada
  puede traer tonos (timbres, acoples, tonos de llamada), mira la planitud.
- **F0 sin interpolación**: la resolución es de lag entero, así que la mediana se
  cuantiza más a frecuencias altas.
- **Ventana y salto fijos**: 64 ms y 16 ms, no configurables.
- La ganancia no afecta las razones: medir a -20 dB o a -40 dB da lo mismo.

### Garantías

- **Determinista entre procesos**: los tests fijan valores golden en hexadecimal
  y los verifican en un proceso distinto, así que hasta un cambio de backend de
  FFT se detecta.
- **Memoria acotada por lotes**, no por duración: el pico se mantiene por debajo
  de 80 MB sobre 20 s, y crece menos de 1.5× al pasar a 120 s.

> El umbral `voice_band_ratio >= 0.4` que aparece en el ejemplo es el que usa
> un consumidor en producción para decidir si hubo voz bajo un texto transcrito. Es un
> valor calibrado contra su micrófono y su caso, no una constante de esta
> librería. Calibra el tuyo con `scripts/validar_evidencia_de_voz.py`, que compara
> un tramo de habla real contra uno de cuarto vacío e imprime el solape.

---

## Registro de voces

`speechtotext.speakers.registry`. Vive en `~/.speechtotext` o en lo que apunte
`SPEECHTOTEXT_HOME`.

```python
from speechtotext.speakers import registry

registry.enroll("Samuel", embedding, seconds=12.4, model="pyannote/speaker-diarization-community-1")
registry.list_voices()                                    # todas, de todos los modelos
vectores = registry.get_embeddings("pyannote/speaker-diarization-community-1")
registry.remove("Samuel")                                 # en todos los modelos
```

| Función | Contrato |
|---|---|
| `home() -> Path` | `SPEECHTOTEXT_HOME` si está definida; si no, `~/.speechtotext`. |
| `enroll(name, embedding, *, seconds, model) -> None` | `model` es obligatorio y keyword-only. |
| `list_voices(model=None) -> list[dict]` | Filas `{"name", "model", **meta}`, ordenadas por nombre. Sin `model`, mezcla todos. |
| `get_embeddings(model) -> dict[str, np.ndarray]` | `model` es obligatorio y posicional. |
| `remove(name, *, model=None) -> bool` | Sin `model`, borra esa persona en todos los modelos. |

### El modelo es parte de la identidad

Cada vector se archiva bajo el modelo que lo produjo y solo se compara contra
vectores del mismo modelo: el coseno entre dos espacios vectoriales distintos no
significa nada. Por eso `get_embeddings` exige el modelo y no tiene default.

**Trampa**: pedir un modelo que no tiene voces enroladas devuelve `{}`, no un
error. Un nombre de modelo mal escrito se ve igual que un registro vacío — nadie
identificará a nadie y no habrá ninguna señal de por qué. Si tu app depende de
identificar, comprueba que el diccionario no venga vacío antes de seguir.

Las entradas cuyo `.npy` ya no existe en disco se ignoran en silencio.

### Formato en disco

```
~/.speechtotext/voices/
├── manifest.json
└── pyannote_speaker-diarization-community-1/
    └── samuel.npy
```

`manifest.json` es un diccionario anidado, `{modelo: {nombre: metadatos}}`, con
`file`, `seconds` y `enrolled_at` por voz. Las rutas se guardan con `/` fijo para
que el registro sea portable entre plataformas. Los vectores son `float32`.

Los manifiestos planos de la `0.4.x` (sin agrupar por modelo) se siguen leyendo:
se reagrupan en memoria al cargar, y el disco se reescribe al formato nuevo en el
siguiente `enroll` o `remove`.

### Producir embeddings

Hoy solo hay un productor: `speakers.diarization.embed_voice`, con pyannote, y
corre el pipeline completo de diarización para sacar un vector — **segundos por
muestra**. Sirve para lote, no para un camino en vivo.

El registro en cambio sí es agnóstico: `model` es una cadena libre, y archivar
vectores de otro extractor funciona hoy. Lo que está atado a pyannote es el CLI
(`enroll` y la ruta de identificación), no el almacén.

---

## Identificación

`speechtotext.speakers.identify` es el módulo — el paquete `speakers` no reexporta nada,
se importa por submódulo. `assign_names(clusters, enrolled, threshold) -> dict[str, str]`
mapea cada `speaker_id`
anónimo a un nombre registrado, con un greedy: ordena todos los pares posibles de mayor a
menor coseno, corta por debajo del umbral, y asigna sin reusar ni un hablante ni un
nombre. El umbral por defecto del CLI es `0.5`. `cosine` está expuesto pero es
implementación: puede cambiar sin aviso.

### Diarizar

`speakers.diarization.diarize(samples, sample_rate, num_speakers=None)` devuelve
`(turns, embeddings)`: `turns` es una lista de `(start, end, speaker_id)` y `embeddings`
un vector por `speaker_id`, en el mismo espacio que `embed_voice` — comparables contra lo
registrado. `num_speakers` es una pista; si se omite, el pipeline decide cuántos hay. Un
hablante cuyo embedding salga con `NaN`, o que el pipeline no devuelva, aparece en `turns`
pero no en `embeddings`. Requiere el extra `[diarize]`; el pipeline se carga una vez por
proceso.

---

## Señales no finitas: dos políticas, a propósito

El mismo trío de señales (`no_speech`, `avg_logprob`, `compression_ratio`) se
trata distinto según por dónde entre, y la diferencia es deliberada:

| Camino | Ante `NaN` o `inf` |
|---|---|
| `core.segments.native_signals` (el del CLI y el JSON) | Lo convierte en ausencia de medida: la clave se omite del JSON. |
| `asr.types.NativeSignals` / `SegmentNativeSignals` | Levanta `ValueError`. |

A mitad del pipe, "esta señal no existe" ya es una respuesta correcta, y obligar a
cada llamador a atrapar una excepción por eso no compra nada. Dentro de un
`TranscriptionResult`, en cambio, un no-finito es corrupción de datos y debe
abortar.

Consecuencia práctica: en el JSON del CLI, *ausente* y *inválido* se ven igual; en
los tipos de `asr/`, lo inválido no llega a existir.

---

## Capa `asr/`: el único contrato de motor

`speechtotext.asr` define el motor de voz a texto: `AsrBackend`, `Caps`,
`TranscriptionRequest`, `TranscriptionResult`, `NativeSignals` y compañía, con
validación estricta y dataclasses inmutables. **Todo camino pasa por aquí**: el CLI,
`bench`, `find` y `transcribe()` construyen un backend y le hablan igual.

```python
class AsrBackend(Protocol):
    backend_id: str          # "faster-whisper" | "whispercpp"
    caps: Caps               # hotwords / vad / word_timestamps -> honrado | degradado | rechazado
    model_id: str; model_version: str; engine_version: str; quant: str; device: str
    def warm(self) -> None                                  # carga (una vez); el objeto es la caché
    def transcribe(self, samples: np.ndarray, request: TranscriptionRequest) -> TranscriptionResult
```

`TranscriptionResult.segments` son `TranscriptionSegment(start, end, text, words, native_signals)`,
y sus `words` son `TranscriptionWord(text, start, end, confidence)` — `confidence` es `None`
o un número entre 0 y 1, nunca fabricado. `Cap` es el `Literal["honrado", "degradado", "rechazado"]`
del que se arman los `Caps`.

Entra **float32 mono a 16 000 Hz** y nada más; quien llama resamplea (`core.transcribe.load_audio`
lo hace desde cualquier archivo). El texto de segmentos y palabras se devuelve tal como lo
emite el motor, con su espacio inicial: recortar es de quien presenta.

Dos implementaciones, ninguna reexportada (importar `speechtotext.asr` no carga motores):

- `speechtotext.asr.faster_whisper.FasterWhisperBackend(model, config=None, *, model_version="unpinned")`
  — `model` es un nombre (lo resuelve faster-whisper desde HF Hub) o una ruta a un
  directorio CTranslate2 (solo local). Honra hotwords, VAD y palabras.
- `speechtotext.asr.whispercpp.WhisperCppBackend(model)` — binario y modelo GGML pinneados
  por SHA-256 (`core/enginepin.py`), CUDA siempre, `q5_0`. Rechaza hotwords (el prompt es
  inerte bajo `-mc 0`), degrada VAD y palabras, y no emite señales nativas.

`TranscriptionRequest(language="es", hotwords=(), word_timestamps=True, beam_size=5, context=None, vad=False)`;
`language="auto"` deja detectar. El `fingerprint` incluye `vad`.

---

## `transcribe()`

```python
from speechtotext.core.transcribe import transcribe, Transcript, Progress, AsrError

t = transcribe("reunion.mp4", model="large-v3", on_progress=print)
```

Un archivo (o muestras 16 kHz mono) entra, un `Transcript` sale: `segments` (con hablante y
señales nativas; la marca `suspect` la calcula el escritor JSON con `is_suspect`), `language`,
`language_probability`, `duration`, `speech_s`,
`gaps`, `engine` (un `EngineInfo`, con `.to_dict()` para el JSON), `request` (la efectiva, tras CAPS), `warnings` y `diarization` (un `DiarizationReport` o `None`). Una sola
decodificación; el archivo corto y el largo son el mismo camino con n trozos; los trozos
dejan checkpoint por contenido en `~/.speechtotext/chunks`. El núcleo nunca imprime:
`on_progress` recibe `Progress(stage, done, total, detail)` con etapas
`decode → load → transcribe → diarize` (con archivo, `decode` se emite dos veces: antes
con `total=None` y después con `done = total = duración`; `download` la emite
`models.ensure`); `cancel` es un
`threading.Event` que se mira entre trozos.

Errores: `AsrError(code, recoverable, message)` con `code` en `unsupported_option`,
`out_of_memory`, `insufficient_resources`, `backend_failed`, `cancelled`, `diarize_unavailable`, `diarize_failed`;
`AudioDecodeError` si el archivo no se puede abrir. Los avisos del motor (p. ej. `empty_transcript`)
llegan a `warnings` como `"<motor>: <aviso>"`. `backend=` permite reutilizar un
modelo caliente entre llamadas. `route=` recibe una `Route` ya resuelta (el CLI sondea,
imprime la razón y la pasa: la máquina se mira una sola vez).

---

## Sondeo y modelos

```python
from speechtotext.core import probe, models

m = probe.machine()                       # < 1 s, sin cargar modelos
r = probe.choose_route(m, "large-v3")     # engine="auto", device="auto", compute_type="auto"
```

`Machine(platform, cpu_count, ram_gb, cuda, gpu_name, vram_free_gb, whispercpp)`: lo que
hay, con `None` donde no se pudo medir.
`Route(engine, device, compute_type, reason, eta_factor, estimated)`: la elección;
`reason` es una frase para imprimir (vacía si no
hay nada que avisar); `eta_factor` multiplica la duración del audio (`None` = sin medir)
y `estimated` es `False` solo si salió del `bench.json` de esta máquina. Reglas: lo
explícito se respeta, el sondeo solo rellena `auto`; **nunca cambia el modelo** — si no
cabe, `AsrError("insufficient_resources")`. Fuera de Windows, whisper.cpp se etiqueta
`device="native"`.

```python
models.data_dir() -> Path                                   # SPEECHTOTEXT_HOME o la ruta del sistema
models.installed(engine=None) -> list[ModelInfo]            # ModelInfo(engine, name, path, size_bytes, verified)
models.ensure(engine, name, on_progress=None) -> Path       # baja si falta; Progress("download", …)
models.remove(engine, name) -> None                         # FileNotFoundError si no está
models.remote_size(engine, name) -> int | None              # bytes que bajaría ensure; None sin red
```

`verified` es `True` solo para whisper.cpp (sha256 contra el pin); los de faster-whisper
los verifica Hugging Face por tamaño. Nombres válidos: `tiny`, `base`, `small`, `medium`,
`large-v3`, `distil-large-v3` (faster-whisper) y `large-v3`, `small` (whisper.cpp).

---

## Módulos que no son contrato de esta librería

- **`core/`** no tiene `__init__.py` público: se importa por submódulo
  (`from speechtotext.core.formats import write_json`). Lo que uses de ahí es
  interno y puede moverse entre versiones.

Los paquetes `evaluation/`, `security/`, `models/` y `confidence/` que existían
hasta la `0.5.1` se extrajeron en la `0.6.0`: eran el arnés de evaluación y la
cadena de custodia de un consumidor, no parte de transcribir audio.
