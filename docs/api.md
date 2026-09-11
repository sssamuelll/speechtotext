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
| `engine` | `object` | Solo si el CLI lo informa. Incluye `diarization: "segment"` o `"word"` cuando se usó `--diarize`. |
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
arnés de [evaluación](audio-evaluation.md), no este campo.

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
> aurelius en producción para decidir si hubo voz bajo un texto transcrito. Es un
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

`speechtotext.speakers.identify` compara por coseno y asigna nombres con un
greedy: ordena todos los pares posibles de mayor a menor puntaje, corta por
debajo del umbral, y asigna sin reusar ni un hablante ni un nombre. El umbral por
defecto del CLI es `0.5`.

---

## Verificación de modelos

`speechtotext.models` prueba que un directorio de pesos está intacto antes de que
un backend lo cargue. **Solo Windows.**

```python
from speechtotext.models import load_model_manifest, verify_model_files

manifest = load_model_manifest(ruta, model_root=raiz, expected_fingerprint=huella)
with verify_model_files(manifest, raiz) as artefacto:
    ...
```

Público: `ModelFile`, `ModelManifest`, `VerifiedModelArtifact`, `ModelIntegrityError`,
`load_model_manifest`, `parse_model_manifest_bytes`, `verify_model_files`.

El manifiesto usa el esquema `speechtotext.model/v1` y exige una `revision`
inmutable (SHA de git o `sha256:<hex>`). La verificación exige que el árbol sea
de solo lectura, que el inventario coincida exactamente — ni un archivo de más ni
de menos — y vuelve a comprobar el inventario después de hashear. Todo fallo
levanta `ModelIntegrityError`: nunca degrada, nunca continúa con una advertencia.

`VerifiedModelArtifact` es un context manager de un solo uso.

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
`TranscriptionResult` o de un artefacto de calibración, en cambio, un no-finito es
corrupción de datos y debe abortar.

Consecuencia práctica: en el JSON del CLI, *ausente* y *inválido* se ven igual; en
los tipos de `asr/`, lo inválido no llega a existir.

---

## Capa `asr/`: contratos sin cableado

`speechtotext.asr` define una interfaz provider-neutral — `AsrBackend`,
`TranscriptionRequest`, `TranscriptionResult`, `NativeSignals` y compañía — con
validación estricta y dataclasses inmutables.

> **Ningún camino del CLI la usa hoy.** `speechtotext transcribe` va directo a
> `core.engines`. Si la adoptas, estás construyendo sobre una capa paralela, no
> sobre el camino que ejercita la suite de transcripción.

`FasterWhisperBackend` es su única implementación y no se reexporta: se importa
desde `speechtotext.asr.faster_whisper`. Importar `speechtotext.asr` no carga
`faster_whisper` — la dependencia pesada entra solo si pides el backend.

---

## Módulos que no son contrato de esta librería

- **`confidence/`** — calibra la probabilidad de que un segmento sea usable, a
  partir de 13 features. Es infraestructura del arnés de evaluación
  (`python -m speechtotext.evaluation`), no algo que `speechtotext transcribe`
  active. No lo busques en la salida del CLI.
- **`evaluation/`** — corpus privado, métricas y gate de aceptación. Se opera por
  su CLI; ver [`audio-evaluation.md`](audio-evaluation.md). Solo Windows.
- **`security/`** — artefactos privados de runtime bajo `%LOCALAPPDATA%`, con ACL
  de un solo usuario y cifrado NTFS verificados por handle. Dentro de este repo su
  único llamador es su propio `python -m speechtotext.security promote`; el
  consumidor real vive en aurelius. Solo Windows, y falla cerrado en sesión
  elevada a propósito.
- **`core/`** no tiene `__init__.py` público: se importa por submódulo
  (`from speechtotext.core.formats import write_json`). Lo que uses de ahí es
  interno y puede moverse entre versiones.
