# Diseño: transcripción por trozos (resumible + paralela)

**Fecha:** 2026-07-08
**Issue:** #7 — *Audio largo: transcripción como pase único bloqueante (sin checkpoint, resume ni paralelismo)*
**Rama:** `feat/transcripcion-chunked`

## Problema

`transcribe_file()` materializa la transcripción completa de golpe (`segments = list(segments_iter)`) y no escribe nada hasta el final. En audio largo con `large-v3` (~1.1× tiempo real en CPU), un archivo de 2h+ es un pase único de 2h30m: cualquier corte pierde el 100% del trabajo, no hay resume, y un solo job de faster-whisper no satura una CPU multinúcleo. El incidente que lo destapó: un archivo de ~2h14m murió tres veces por teardown de entorno antes de producir un solo archivo.

Como `_transcribe_opts` ya fija `condition_on_previous_text=False` (las ventanas se tratan de forma independiente), **trocear en fronteras de silencio no degrada la transcripción** — solo desbloquea checkpointing, resume y paralelismo.

## Alcance

**Dentro (v1):** durabilidad (checkpoint por trozo + resume) **y** paralelismo (pool de workers), juntos — comparten la infra de troceo. Componer con diarización. Progreso legible off-TTY.

**Fuera (v1):** nada del modelo acústico ni del post-proceso (puntuación, hotwords, horas). Nada de la alineación palabra→hablante (ya resuelta, item #2 del backlog). Esto es **solo** durabilidad + throughput del pase de transcripción.

## Decisiones (brainstorming, 2026-07-08)

1. **Alcance v1:** durabilidad + paralelismo.
2. **Diarización:** componer — trocear solo la transcripción, reensamblar con timestamps globales, diarizar el audio completo como hoy.
3. **Fronteras:** conscientes del silencio (`ffmpeg silencedetect`), con fallback a corte fijo.

## Flujo

- Archivo **corto** (≤ umbral) o `--no-chunk`: ruta actual sin cambios.
- Archivo **largo** (> `CHUNK_THRESHOLD` ≈ 20 min = 1200 s) o `--chunk`: ruta troceada.
- Downstream idéntico en ambos casos: `diarize` → `normalize_hours` → writers.

```
dur = probe_duration(audio)
if chunk is None: chunk = dur > CHUNK_THRESHOLD
if chunk:
    segments, info = run_chunked(audio, opts, jobs)
else:
    segments_iter, info = whisper.transcribe(str(audio), **opts); segments = list(segments_iter)
# resto idéntico
```

## Módulo nuevo: `core/chunked.py`

### Representación de segmentos

Los `Segment` de faster-whisper son inmutables — no se pueden desplazar sus timestamps in-place. `chunked.py` produce dataclasses ligeros, drop-in para todos los consumidores (que usan `s.start/s.end/s.text` y `getattr(s, "words", None)`):

```python
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
```

El offset del trozo se aplica al **construirlos** — a `start/end` del segmento **y de cada palabra** (crítico: `assign_segments` es word-level; sin offset en palabras la diarización se rompe).

### `plan_chunks(audio, target_len=600.0) -> list[tuple[float, float]]`

- Una pasada de `ffmpeg -i audio -af silencedetect=noise=-30dB:d=0.5 -f null -`; parsear `silence_start`/`silence_end` de stderr.
- Punto de corte = el punto medio del silencio más cercano a cada múltiplo de `target_len`. Si no hay silencio dentro de una ventana de búsqueda (±~30 s) de una frontera → corte fijo en el múltiplo.
- Devuelve rangos `(start, end)` contiguos que cubren `[0, dur]`, en timeline global.
- Función pura sobre la lista de silencios: `pick_cuts(silences, dur, target_len, search) -> cuts` (testeable sin ffmpeg).

### `transcribe_chunk(audio, start, end, opts, model) -> list[TimedSegment]` (con checkpoint)

- **Clave de contenido** (patrón de `finder.index_path`): `sha1("{audio.resolve()}|{size}|{mtime}|{model}|{lang}|{beam}|{vad}|{hotwords}|{word_timestamps}|{start}|{end}")[:16]` → `~/.speechtotext/chunks/{digest}.json` (reusa `finder._home`). La clave incluye **todo** parámetro que afecte la salida, así un cambio invalida el checkpoint.
- Si el JSON existe y parsea → cargar (resume), sin extraer ni transcribir.
- Si no: `ffmpeg -y -ss {start} -t {end-start} -i audio -ar 16000 -ac 1 {tmp.wav}` → `model.transcribe(tmp.wav, **opts)` → construir `TimedSegment`/`TimedWord` con offset `+start` → escribir JSON → borrar `tmp.wav`.
- Espejo de `load_or_build_index`: fallback a recomputar si el JSON está corrupto.
- El JSON guarda segmentos con timestamps **globales** (offset ya aplicado) → reensamblar es concatenar.

### `run_chunked(audio, opts, jobs) -> (list[TimedSegment], info)`

- `plan_chunks` → lista de rangos.
- **Un solo** `WhisperModel(model, device, compute_type, cpu_threads=max(1, cpu_count()//jobs), num_workers=jobs)` — pesos compartidos (2-3 GB, no N copias). `num_workers` es el mecanismo de faster-whisper para `transcribe()` concurrente desde varios hilos.
- `ThreadPoolExecutor(max_workers=jobs)` somete `transcribe_chunk` por rango. Los trozos con checkpoint no se extraen ni transcriben.
- Reensamblar: concatenar los `TimedSegment` en orden de trozo (ya con timestamps globales).
- `info` sintético: `.language` (del 1er trozo; si `lang` es `auto`, se detecta en el 1er trozo y se **fija** para los demás, por consistencia), `.language_probability`, `.duration` = duración total.

### Progreso off-TTY

`run_chunked` detecta si stdout es TTY. Si **no** lo es (redirigido a log), emite una línea por trozo terminado: `[k/n] trozo mm:ss–mm:ss OK (cache|nuevo)`. En TTY, el spinner de `rich` como hoy. Resuelve el log mudo al redirigir salida.

## Wiring en `transcribe_file`

- `probe_duration(audio)` vía **PyAV** (`av.open(audio).duration` — instantáneo desde metadata del contenedor, sin subprocess; PyAV ya es dependencia del finder). Fallback a `ffprobe` si el contenedor no reporta duración.
- Nuevas flags en `transcribe`: `--chunk/--no-chunk` (default: auto por duración), `--jobs/-j` (default `min(4, n_trozos)`).
- La ruta troceada sustituye solo el bloque `segments = list(segments_iter)`. El resto de `transcribe_file` (diarización, `normalize_hours`, writers) no cambia.

## Diarización (componer)

Tras reensamblar, si `--diarize`: `_run_diarization(audio, segments, ...)` sobre el audio **completo**, igual que hoy. Ya es whole-file post-transcripción; solo requiere que los segmentos traigan timestamps (y palabras) globales — ya los traen. **Cero cambios en `diarization.py`.** La diarización no se paraleliza (sigue siendo un pase whole-file); el troceo acelera solo la transcripción.

## Reensamblaje y timestamps globales

Único punto de corrección no trivial: `write_srt`/`write_vtt`/`write_json` deben recibir tiempos **globales**, no relativos al trozo. Se garantiza aplicando el offset al construir cada `TimedSegment`/`TimedWord` (no en los writers, que quedan intactos).

## Criterios de aceptación (del issue #7)

- [ ] Un archivo de 2h+ se puede interrumpir en cualquier momento y reanudar, perdiendo ≤ 1 trozo de cómputo.
- [ ] En CPU multinúcleo ociosa, el wall-clock de un archivo largo baja de forma material vs el pase único (casi lineal en `--jobs` hasta saturar núcleos).
- [ ] La salida txt/srt/json es estructuralmente idéntica a la del pase único, con timestamps globales correctos. Test: chunked vs single sobre un archivo corto producen segmentos equivalentes tras aplicar el offset.
- [ ] Archivos cortos: comportamiento y rendimiento sin cambios.

## Reutilización (no partir de cero)

- Recorte con ffmpeg: patrón de `_extract_region` (`cli/app.py:298`), `clip_window` (`finder.py:70`).
- Caché con clave de contenido + carga-o-computa: `finder.index_path` / `load_or_build_index` (`finder.py:82-127`).
- Home de estado en disco: `finder._home` (`finder.py:77`).
- Writers con timestamps: `core/formats.py` (sin cambios).

## Testing (TDD, sin correr modelos)

Lógica pura, datos sintéticos, ffmpeg/pool mockeados:

- `pick_cuts`: elección de cortes dado un set de silencios; fallback a fijo cuando no hay silencio cerca.
- **Offset de segmentos y palabras** al construir `TimedSegment` desde un resultado de trozo.
- Reensamblaje: orden y continuidad de timestamps globales.
- Clave de checkpoint: determinismo, e invalidación al cambiar un parámetro (modelo, hotwords, word_timestamps, rango).
- Resume: con checkpoint presente no se invoca la transcripción; ausente sí (vía mock).
- Equivalencia chunked-vs-single: dado el mismo resultado de trozo mockeado, los segmentos reensamblados igualan al pase único tras offset.

Nada de transcripción real ni de ffmpeg real en los tests unitarios.

## Riesgos / notas

- **Oversubscription:** con `num_workers>1`, `cpu_threads` **debe** fijarse explícito (`cores//jobs`); el default de CTranslate2 (todos los núcleos por worker) saturaría. Cubierto en `run_chunked`.
- **Costo de `silencedetect`:** una pasada de análisis sobre todo el audio (decodifica internamente, ~decenas de s en 2h). Costo único, aceptable.
- **Temp wavs:** extracción on-demand por worker (~38 MB por trozo de 20 min a 16 kHz mono); se borran tras el checkpoint. Con resume no se extraen.
