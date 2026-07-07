# `find` — buscador de segmento

- **Fecha:** 2026-07-07
- **Estado:** diseño aprobado, listo para plan de implementación
- **Alcance:** subcomando `find` del CLI — ubicar contenido en un audio largo por palabra clave, y opcionalmente recortar + transcribir en calidad ese tramo.
- **Fuera de alcance:** el servidor MCP que expondrá el toolkit (sub-proyecto siguiente, spec propio).

## 1. Contexto y objetivo

Transcribir un audio largo (p.ej. un programa de radio de 81 min) en calidad tarda horas.
Muchas veces solo interesa **un tramo** (una entrevista, una ponencia). `find` resuelve
"¿dónde está X?" sin transcribir todo en calidad: hace un pase rápido con `tiny`, busca la
consulta, y devuelve las **regiones** donde aparece. Con `--extract`, además recorta y
transcribe en calidad la región elegida.

Automatiza el flujo que hoy se hace a mano: escanear con `tiny` → buscar palabra clave →
recortar → transcribir en calidad.

## 2. Decisiones tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Disparo de la parte cara | **Ubicar por defecto; `--extract` transcribe** | Transcribir 20 min tarda ~1 h; no debe lanzarse por accidente. |
| Índice | **Cacheado por archivo** | Primer `find` construye el índice `tiny` una vez; búsquedas siguientes instantáneas. |
| Salida de la búsqueda | **Regiones agrupadas y rankeadas** | Un bloque denso ("25–45 min = la entrevista") es más útil que 20 timestamps sueltos. |
| Coincidencia | **Sin acentos, sin mayúsculas, cualquier término** | Tolerante a tildes y a los deslices de `tiny`; recall amable. |

## 3. Comando y UX

```
speechtotext find AUDIO "consulta" [opciones]
```

Por defecto (ubicar) imprime regiones rankeadas:
```
Índice: usando caché (tiny, 81 min)
3 regiones para "vulnerabilidad sísmica":
  1.  25:19 – 45:40   (18 coincidencias)  "…profesor Simón Ballesteros… patología…"
  2.  02:55 – 03:03   (2)                 "…no existe la cultura sísmica…"
  3.  …
```

Con `--extract` recorta + transcribe la región elegida:
```
find audio.mp3 "vulnerabilidad sísmica" --extract             → región #1 (top)
find audio.mp3 "vulnerabilidad sísmica" --extract --region 2  → otra
find audio.mp3 "vulnerabilidad sísmica" -e -D --speakers 4    → con diarización
```

### Opciones

| Flag | Default | Descripción |
|---|---|---|
| `--extract`, `-e` | off | Recortar + transcribir en calidad la región. |
| `--region N` | `1` | Qué región extraer (1 = la más densa). |
| `--model`, `-m` | `small` | Modelo para la transcripción en calidad. |
| `--scan-model` | `tiny` | Modelo del índice. |
| `--diarize`, `-D` | off | Diarizar la transcripción del tramo. |
| `--speakers` | auto | Nº de hablantes (pista) para la diarización. |
| `--identify / --no-identify` | `--identify` | Nombrar voces registradas. |
| `--threshold` | `0.5` | Umbral de coincidencia de voz. |
| `--context` | `10` | Segundos de margen alrededor de la región al recortar. |
| `--output`, `-o` | junto al audio | Dónde caen el clip y el transcript. |
| `--rebuild` | off | Forzar reconstrucción del índice. |
| `--top` | `5` | Cuántas regiones listar. |

## 4. Índice y caché

- **Construir:** transcodear a wav 16 kHz mono → transcribir con `--scan-model` (`tiny`) →
  guardar los segmentos `{start, end, text}` en JSON.
- **Ubicación:** `${SPEECHTOTEXT_HOME:-~/.speechtotext}/index/<hash>.json`, donde
  `hash = sha1(ruta_absoluta + tamaño + mtime)`. Si el archivo cambia, el hash cambia →
  se reconstruye solo.
- **Contenido del JSON:** `{audio, size, mtime, scan_model, segments: [{start, end, text}]}`.
- Si existe un índice válido para ese audio y `scan_model`, se reutiliza. `--rebuild` lo fuerza.

## 5. Coincidencia y agrupación en regiones (lógica pura)

- `normalize(text)`: minúsculas + quitar acentos (unicodedata NFKD, descartar diacríticos).
- La consulta se parte en términos por espacios; cada término se normaliza.
- Un segmento es "hit" si su texto normalizado contiene **alguno** de los términos.
- `match_count(segmento)`: cuántos términos distintos aparecen (para el ranking fino).
- `cluster_regions(hits, gap=60.0)`: fusiona hits consecutivos —o separados por un hueco
  menor a `gap` segundos— en una región `Region(start, end, hits, matches, snippet)`.
  `snippet` = texto del primer segmento hit de la región (recortado a ~80 chars).
- **Ranking:** por número de segmentos hit (densidad), desempate por `matches` totales.

## 6. Flujo de `--extract`

1. Elegir la región (`--region N`, default la #1 rankeada).
2. Recortar del audio original con ffmpeg: `-ss (start - context) -t (dur + 2*context)`,
   a wav 16 kHz mono.
3. Transcribir el recorte con `transcribe_file(...)` (ver §7), con los flags de calidad y
   diarización recibidos.
4. Escribir el clip y el/los transcript(s) en `--output` (o junto al audio), nombrados por
   el tramo: `<stem>_<mmMss>-<mmMss>.wav` / `.txt` / `.srt`.

## 7. Arquitectura y archivos

```
src/speechtotext/
├── core/
│   └── finder.py     NUEVO · índice (build/load/cache) + normalize + search + cluster_regions
├── cli/
│   └── app.py        + comando `find`; REFACTOR: extraer el cuerpo de `transcribe` a
│                       `transcribe_file(...)` reutilizable
```

### `core/finder.py`
- `index_path(audio: Path, scan_model: str) -> Path`
- `build_index(audio: Path, scan_model: str) -> list[dict]` — usa `WhisperModel` (faster-whisper).
- `load_or_build_index(audio: Path, scan_model: str, rebuild: bool) -> tuple[list[dict], bool]`
  (el bool indica si se usó caché).
- `normalize(text: str) -> str`
- `search(segments: list[dict], query: str, gap: float, top: int) -> list[Region]`
  (llama internamente a `cluster_regions`).
- `Region` = dataclass `{start: float, end: float, hits: int, matches: int, snippet: str}`.

`normalize`, `search`, `cluster_regions` son **puras** (sin whisper ni fs) → unit-testeables.

### Refactor de `transcribe`
Hoy el cuerpo de `transcribe()` (WhisperModel → segments → diarize → writers) está inline.
Se extrae a:
```
transcribe_file(audio, output_base, model, requested_formats, language, device,
                compute_type, vad, beam_size, diarize, speakers, identify, threshold) -> None
```
Tanto el comando `transcribe` como `find --extract` la llaman. El comando `transcribe`
mantiene su firma y comportamiento idénticos; solo delega el cuerpo.

## 8. Manejo de errores y bordes

- Sin coincidencias → `No se encontró "<consulta>" en el audio.` (exit 0).
- `--region N` fuera de rango → error claro (exit 1).
- `--extract` con `--diarize` sin el extra `[diarize]`/`HF_TOKEN` → mismos mensajes que `transcribe`.
- ffmpeg ausente → error tipado existente (`FfmpegMissingError`).
- Índice corrupto/ilegible en caché → se ignora y se reconstruye.

## 9. Testing

**Puras (sin whisper ni red):**
- `normalize` — acentos y mayúsculas (`"Sísmica"` → `"sismica"`).
- `search`/`cluster_regions` — índice sintético + consulta:
  - hits contiguos → una región;
  - hits separados por hueco < gap → misma región; > gap → dos regiones;
  - ranking por densidad;
  - snippet correcto;
  - sin coincidencias → lista vacía.

**Integración (más lenta):**
- `build_index` sobre un audio corto generado con ffmpeg (tiny) → devuelve segmentos con
  `start/end/text`; y la caché se reutiliza en la segunda llamada. Smoke, puede marcarse lento.

## 10. Continuidad

El sub-proyecto siguiente (spec propio) es el **servidor MCP** que expone el toolkit
—`transcribe`, `find`, diarización, `enroll`, conversión de media— como herramientas que un
agente pueda invocar al trabajar con audio/video.
