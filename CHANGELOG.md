# Changelog

Los consumidores pinnean un tag, así que este archivo existe para una sola
pregunta: **¿me conviene subir el pin, y qué se me rompe si lo hago?**

Lo marcado como **rompe** exige cambios en el código que consume la librería.

---

## Sin publicar

### Añadido

- `core.transcribe.transcribe()`: un archivo (o muestras) entra, un `Transcript` sale, con
  progreso por callback y cancelación. Una sola decodificación del audio; el archivo corto
  y el largo recorren el mismo camino. Contrato en [`docs/api.md`](docs/api.md#transcribe).
- `WhisperCppBackend` implementa el mismo contrato que `FasterWhisperBackend`; el CLI,
  `bench` y `find` construyen los motores por un solo sitio.

### Cambiado — rompe

- **Se extrajeron el arnés de evaluación y la cadena de custodia**: `evaluation/`,
  `security/`, `models/`, `confidence/`, `statistics.py` y el extra `[evaluation]`.
  Viven en su único consumidor, como pasó con `api/` en la `0.4.0`. Esta librería
  queda en lo que es: audio → texto, con o sin hablantes, y las medidas sobre la
  señal.
- `FasterWhisperBackend(model, config=None, *, model_version="unpinned")` recibe
  un nombre o una ruta, no un `VerifiedModelArtifact`. Los Protocols
  `CalibratedAsrBackend` y `VerifiedLocalAsrBackend` se fueron con la verificación.
- `TranscriptionResult` ya no trae `confidence_target`, `calibrated_confidence` ni
  `calibrator_version`, y `ConfidenceTarget` desaparece. Quien calibre envuelve el
  resultado.
- `PipelineProvenance` recibe `ModelRef(model_id, fingerprint)` en `models=`, no
  artefactos verificados. Las huellas no cambian: el payload solo guardaba
  `model_fingerprints`.

  Migración: importar lo extraído desde su nuevo paquete; envolver
  `FasterWhisperBackend` con la verificación propia pasando `model=artefacto.root`
  y `model_version=artefacto.manifest.revision`; convertir cada artefacto a
  `ModelRef(manifest.model_id, artefacto.fingerprint)` antes de derivar proveniencia.
  Con una ruta, el `model` del resultado pasa a ser el nombre del directorio
  (`Path.name`), no el `model_id` del manifiesto; si el envoltorio necesita
  conservar ese id, sobreescribe la propiedad `model_id` del backend.
- **`AsrBackend.transcribe` recibe muestras, no un `AudioClip`**: float32 mono a 16 kHz.
  Quien tenga un clip pasa `clip.view("asr").samples`. El Protocol declara además `caps`,
  `engine_version`, `quant` y `device`.
- `TranscriptionRequest` gana `vad: bool = False`; entra al `fingerprint`, así que las
  huellas de peticiones cambian una vez. `language="auto"` es válido.
- El texto de `TranscriptionSegment` y `TranscriptionWord` se devuelve crudo (con el
  espacio inicial del motor); `result.text` sigue recortado.
- `speakers.diarization.diarize(samples, sample_rate, num_speakers=None)` recibe muestras;
  `read_wav(path)` las lee de un wav. `embed_voice(wav_path)` no cambia.
- `core.chunked`: `chunk_path(identity, start, end)`; se van `run_chunked`,
  `transcribe_chunk` y `probe_duration`. Los checkpoints viejos dejan de coincidir y se
  recomputan.
- Se va `core/engines.py`: el adaptador de whisper.cpp es `asr.whispercpp.WhisperCppBackend`
  y los CAPS viven en cada backend.

---

## v0.5.1 — 2026-09-12

### Añadido

- Las señales nativas del motor llegan al JSON: cada segmento puede traer
  `no_speech`, `avg_logprob` y `compression_ratio` (#23). Un valor no finito se
  trata como ausencia de medida y la clave se omite, en vez de escribir `NaN`,
  que ni siquiera es JSON válido. Contrato en [`docs/api.md`](docs/api.md#esquema-del-json).
- Documentación del contrato para consumidores (`docs/api.md`) y este changelog.

### Arreglado

- Con `--chunk`, Whisper podía emitir un segmento sobre el relleno de ceros de la
  última ventana de un trozo (una despedida de YouTube con 30 s de marca falsa) y
  ese segmento caía encima del trozo siguiente. Ahora se recorta al final real del
  trozo, también al leer checkpoints viejos que lo traigan.
- `speechtotext.__version__` decía `0.3.0` desde hace dos tags, y el docstring del
  paquete seguía anunciando un servicio HTTP de pronunciación que se fue en la
  `0.4.0`.

---

## v0.5.0 — 2026-09-11

### Añadido

- **Evidencia de voz por DSP determinista** (#22): `compute_voice_evidence` mide
  energía en la banda de voz, proporción de tramos sonoros, F0 mediana y planitud
  espectral. Son medidas, no veredictos — el módulo no clasifica voz contra
  no-voz a propósito, y distingue "no pude medir" (`None`) de "medí y no hay"
  (`0.0`). Contrato en [`docs/api.md`](docs/api.md#evidencia-de-voz).

### Cambiado — rompe

- **El registro de voces archiva y filtra por modelo.** `registry.enroll` exige
  ahora `model`, y `registry.get_embeddings(model)` exige el modelo como
  argumento obligatorio: el coseno entre dos espacios vectoriales distintos no
  significa nada, así que una voz solo se compara contra vectores del mismo
  extractor.

  Migración: los manifiestos planos de la `0.4.x` se siguen leyendo — se
  reagrupan en memoria al cargar y el disco se reescribe al formato nuevo en el
  siguiente `enroll` o `remove`. Lo que sí hay que tocar es el código: cualquier
  `get_embeddings()` sin argumento levanta `TypeError`.

  Cuidado con el modo de fallo silencioso: pedir un modelo sin voces enroladas
  devuelve `{}`, no un error. Un nombre mal escrito se ve igual que un registro
  vacío.

---

## v0.4.0 — 2026-08-07

### Cambiado — rompe

- **Se extrajo el servicio HTTP de evaluación de pronunciación** (FastAPI +
  Azure). Vivía en `api/` y hoy vive adaptado dentro de su único consumidor
  (klara). Esta librería quedó como lo que es: voz a texto local, sin servidor y
  sin dependencias de nube.
- A partir de aquí el repo se consume como **librería versionada por tags**. Los
  consumidores fijan tag o SHA, nunca `@main`: cambio que un consumidor necesite
  entra por PR aquí, se taggea, y recién entonces se sube el pin allá.
