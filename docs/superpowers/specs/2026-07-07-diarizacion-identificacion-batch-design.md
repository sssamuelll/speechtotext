# Diarización + identificación de hablantes — Fase 1 (core + batch)

- **Fecha:** 2026-07-07
- **Estado:** diseño aprobado, listo para plan de implementación
- **Alcance:** core compartido de voces + superficie **batch** (archivo grabado)
- **Fuera de alcance:** tiempo real / streaming (Fase 2, spec propio)

## 1. Contexto y objetivo

`speechtotext` transcribe audio offline con faster-whisper. Se quiere que, sobre una
conversación entre dos (o más) personas, el transcript indique **quién dijo qué**
(diarización) y, cuando la persona esté registrada, **con su nombre** (identificación).

Todo **offline**: sin subir audio a terceros. El motor de diarización y embeddings es
`pyannote.audio` (modelos neuronales locales). La transcripción actual de faster-whisper
se **reusa intacta**; la diarización se suma encima.

### Por qué no DSP por frecuencias

Separar hablantes por pitch/frecuencia no funciona: el F0 de dos personas se solapa
(sobre todo mismo género) y el de una misma persona varía más al entonar que entre
hablantes distintos. El estado del arte son **embeddings neuronales de voz**
(ECAPA/wespeaker) + clustering, que es lo que usa pyannote. No se rueda DSP propio.

## 2. Decisiones tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Offline vs nube | **Offline** | Identidad del proyecto; privacidad. |
| Separar vs identificar | **Ambos** | Diarización + registro de voces por nombre. |
| Batch vs live | **Batch primero**, live en Fase 2 | Live es otro pipeline (online); reusa este core. |
| Motor | **pyannote directo** + faster-whisper actual | Reusa la transcripción; deps más contenidas; control del enrollment. |
| CLI | **Quiebre limpio a subcomandos** | Estructura correcta; herramienta nueva de un solo usuario. |

## 3. Arquitectura

```
src/speechtotext/
├── core/
│   ├── audio.py          (existe · reuso transcode_to_wav a 16 kHz mono)
│   ├── formats.py        (+ writers renderizan hablante)
│   └── segments.py       NUEVO · LabeledSegment(start, end, text, speaker)
├── speakers/             NUEVO · core compartido (lo reusa el live en Fase 2)
│   ├── __init__.py
│   ├── embedding.py      modelo de embeddings · audio → vector L2-normalizado
│   ├── registry.py       enrollment: guardar/listar/borrar voces
│   ├── identify.py       vector → nombre (coseno + umbral)
│   └── diarization.py    pyannote batch + asignación de segmentos a hablantes
├── cli/app.py            subcomandos: transcribe / enroll / voices / forget
└── api/                  (sin cambios)
```

El paquete `speakers/` aísla el corazón reutilizable. La Fase 2 (live) reemplaza solo
`diarization.py` por un `streaming.py`; `embedding.py`, `registry.py` e `identify.py`
sirven a las dos superficies sin cambios.

## 4. Componentes

### `core/segments.py` — `LabeledSegment`
```python
@dataclass
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
```
Motivo: el `Segment` de faster-whisper es un namedtuple inmutable; no se le puede pegar
el hablante. Al diarizar se envuelve cada segmento en `LabeledSegment`. La ruta sin
diarizar sigue pasando segmentos crudos; los writers leen `getattr(seg, "speaker", None)`
→ retrocompatible.

### `speakers/embedding.py`
- `load_embedding_model()` → carga y cachea el modelo de embeddings (el mismo que usa
  el pipeline de pyannote, para consistencia). Requiere `HF_TOKEN` + términos aceptados.
- `embed(wav_path, model, region=None) -> np.ndarray` → vector L2-normalizado de la
  ventana de audio indicada (o todo el archivo).
- Deps: pyannote.audio, torch, numpy.

### `speakers/registry.py`
- Almacén: `${SPEECHTOTEXT_HOME:-~/.speechtotext}/voices/` con `<nombre>.npy` por voz y
  un `manifest.json` (nombre → {archivo, fecha, segundos, modelo}).
- `enroll(name, embedding, meta)`, `list_voices() -> list[VoiceEntry]`,
  `get_embeddings() -> dict[str, np.ndarray]`, `remove(name)`.
- I/O de ficheros, testeable con `SPEECHTOTEXT_HOME` apuntando a un temp dir.

### `speakers/identify.py`
- `identify(vec, enrolled: dict[str, np.ndarray], threshold: float) -> str | None`
  → similitud coseno contra cada voz; devuelve el mejor nombre si ≥ umbral, si no `None`.
- `assign_names(clusters: dict[str, np.ndarray], enrolled, threshold) -> dict[str, str]`
  → mapea `SPEAKER_xx` → nombre. Si dos clusters matchean el mismo nombre, se lo queda el
  de mayor similitud; el otro queda anónimo. numpy puro, 100% unit-testeable.

### `speakers/diarization.py`
- `diarize(wav_path, num_speakers=None) -> turns` → pipeline
  `pyannote/speaker-diarization-3.1`; devuelve turnos `(start, end, speaker_id)`.
- `assign_segments(whisper_segments, turns) -> list[LabeledSegment]` → a cada segmento le
  asigna el hablante con **mayor solape temporal**. Función pura → testeable sin modelos.
- `cluster_embeddings(wav_path, turns, embed_model) -> dict[str, np.ndarray]` → un vector
  representativo por hablante (para identificar).

## 5. Flujo de datos (`transcribe --diarize`)

```
audio ─► faster-whisper ─► segmentos [(start, end, text)]         (intacto)
audio ─► pyannote        ─► turnos    [(start, end, SPEAKER_xx)]
        assign_segments(segmentos, turnos) → LabeledSegment(speaker=SPEAKER_xx)
   si registry no vacío y --identify:
        cluster_embeddings(audio, turnos) → {SPEAKER_xx: vec}
        assign_names(clusters, enrolled, threshold) → {SPEAKER_00: "Samuel", ...}
        relabel: SPEAKER_xx → nombre  (sin match → "Hablante N")
   ─► writers renderizan el hablante
```

## 6. CLI (subcomandos)

```
speechtotext transcribe FILE [--diarize] [--speakers N]
                             [--identify/--no-identify] [--threshold 0.5] [flags previos]
speechtotext enroll NAME SAMPLE.wav
speechtotext voices
speechtotext forget NAME
```

- **Quiebre:** `speechtotext audio.wav` → `speechtotext transcribe audio.wav`. Se actualiza
  el README y los ejemplos.
- Flags nuevos de `transcribe`:
  - `-D/--diarize` (bool, default off).
  - `--speakers N` — pista de cantidad exacta (opcional; auto si se omite). `--speakers 2`
    mejora precisión en conversación conocida de dos.
  - `--identify/--no-identify` — default: identifica si hay voces registradas.
  - `--threshold FLOAT` (default 0.5) — perilla de calibración del match de voz.

## 7. Enrollment

- `speechtotext enroll "Samuel" muestra.wav`: transcode a 16 kHz mono (reusa
  `core.audio`), embedding de toda la muestra, L2-normaliza, guarda `.npy` + manifest.
  Avisa si la muestra dura < ~10 s (embedding poco fiable).
- `speechtotext voices`: tabla (nombre, segundos, fecha, modelo).
- `speechtotext forget "Samuel"`: borra `.npy` + entrada del manifest.
- Home override: `SPEECHTOTEXT_HOME` (tests y no ensuciar el home real).

## 8. Formato de salida

- **txt:** agrupa turnos consecutivos del mismo hablante:
  ```
  Samuel: hola, ¿cómo estás?
  Ale: bien, ¿y tú?
  ```
- **srt / vtt:** prefija el cue → `Samuel: bien, ¿y tú?`
- **json:** cada segmento gana `"speaker"`, más un top-level `"speakers": [...]`.
- Sin `--diarize`: salida idéntica a la actual (speaker vacío → sin prefijo).

## 9. Manejo de errores y bordes

- pyannote/torch no instalado → error claro: `pip install -e ".[diarize]"`.
- Falta `HF_TOKEN` o términos del modelo sin aceptar → capturar el error de pyannote e
  imprimir la URL exacta para aceptar + cómo poner el token.
- Muestra de enrollment muy corta o en silencio → avisar / rechazar.
- `--diarize` sin voces registradas → etiquetas anónimas (`Hablante 1/2`), sin error.
- Match por debajo del umbral → cluster queda anónimo.
- Diarización detecta N ≠ 2 → respeta la cantidad real (no fuerza 2 salvo `--speakers`).
- ffmpeg ausente → ruta de error existente.

## 10. Dependencias / instalación

```toml
[project.optional-dependencies]
diarize = ["pyannote.audio>=3.1", "numpy>=1.24", "torch>=2.0"]
```
El base `pip install -e .` sigue liviano. Diarización con `pip install -e ".[diarize]"`.
Trae torch (~2 GB) y usa modelos gated de pyannote (aceptar términos + `HF_TOKEN` una vez).

## 11. Testing

Partes puras, sin modelos ni red:
- `assign_segments` — solape (segmento a caballo entre turnos → gana el de más solape).
- `identify` / `assign_names` — coseno + umbral + colisión de dos clusters al mismo nombre.
- `registry` — enroll/list/forget round-trip en `SPEECHTOTEXT_HOME` temporal.
- writers — renderizan hablante en txt/srt/vtt/json (y lo omiten si `speaker is None`).

Partes con modelos pesados (pipeline pyannote, embedding real) fuera del unit test:
smoke test manual con un audio de dos voces y una voz enrolada.

## 12. Límites conocidos

- Etiqueta a **nivel de segmento**, no de palabra: un cambio de turno a mitad de segmento
  se lo lleva un solo hablante. (whisperX daría nivel-palabra; se puede añadir después.)
- Precisión de identificación depende de la calidad del enrollment y del umbral; voces muy
  parecidas pueden confundirse.
- Primera corrida descarga modelos gated (requiere token HF).
- CPU funciona pero la diarización suma tiempo sobre la transcripción.

## 13. Continuidad (Fase 2 — live)

El live reusa `speakers/embedding.py`, `registry.py` e `identify.py` sin cambios; añade
`speakers/streaming.py` (diarización online con `diart` + ASR por chunks) y un
`transcribe --live` / comando `listen`. Spec aparte cuando se aborde.
