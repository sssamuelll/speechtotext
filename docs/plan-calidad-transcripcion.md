# Plan de calidad de transcripción — speechtotext

Estado: definitivo, ejecutable.
Árbol que ejecuta: `D:\Desktop\projects\.worktrees\audio-hibrido\speechtotext\src\speechtotext`
(rama `feat/audio-hibrido-seguro`).
Base: junta de cinco consejeros (Halberg, Serrano, Voronov, Vale, Stride), cada propuesta refutada
adversarialmente. En este plan **solo entra lo que sobrevivió la refutación, o la corrección del
refutador cuando la propuesta original se cayó**.

---

## 1. El problema en una página

Se transcribió una conversación real de 36:46 (español, dos personas, micrófono ambiente, con dos
regímenes muy distintos: ~16 min de habla normal y ~18 min de habla susurrada). Cinco corridas.
Esto es lo que pasó, sin jerga:

1. **La herramienta perdió media conversación y dijo que todo salió bien.** La corrida 1 descartó
   18 de los 37 minutos, escribió los archivos e imprimió `OK`. El hueco (16:01 a 31:38) se
   descubrió a mano, mirando los tiempos del SRT. Un usuario normal se lleva media conversación
   creyendo que la tiene entera.
2. **La salida no calla: afirma.** La línea de resumen (`cli/app.py:174-178`) imprimió
   `duración 2206.0s · 88 segmentos`, que es una afirmación de haber procesado 37 minutos, y
   `Idioma detectado: es` cuando `es` es el flag por defecto del propio usuario
   (`cli/app.py:141` → `cli/app.py:175`), no una medición.
3. **Sin VAD, el modelo rellena el silencio con frases de subtítulos de YouTube.** `"Gracias."`
   ×12, `"¿Qué pasa?"` ×9, `"Gracias por ver el video"` ×3. Salen en el archivo con exactamente la
   misma pinta que la voz real: mismo formato, misma autoridad, y encima el post-proceso de horas
   (`core/postprocess.py` vía `cli/app.py:186`) les corrige la ortotipografía igual que al resto.
4. **El proceso murió teniendo el trabajo hecho en disco.** `core/chunked.py:176` construye el
   modelo Whisper *antes* de mirar el caché, y el caché sólo se consulta dentro de
   `transcribe_chunk` (`core/chunked.py:123-129`). Con trozos cacheados el proceso igual pide el
   pico de carga (~3.5 GB en `large-v3` int8) y reventó con `mkl_malloc: failed to allocate memory`.
5. **La diarización sobre-clusterizó** (3 hablantes en una conversación de 2, con cambio de
   etiqueta a mitad de frase) y en otra corrida murió con exit 5 y cero mensaje.
6. **Lo único que separó voz real de invención fue correr dos modelos y quedarse con lo que
   ambos vieron.** Costó el doble de tiempo y se hizo a mano.

El denominador común: **el sistema no tiene ninguna medida de sí mismo**. No sabe (ni dice) cuánto
audio produjo texto, ni con qué confianza emitió cada frase, y su caché no cubre los parámetros con
los que se calculó. Por eso los fallos son indistinguibles entre sí desde la salida, y por eso
cualquier intento de "afinar el modelo" hoy sería no evaluable.

---

## 2. Diagnóstico de fondo

### Dónde convergieron los cinco

- **Es un problema de contrato, no de modelo ni de pipeline.** El modelo hizo su trabajo: con el
  tramo amplificado y sin VAD recuperó voz susurrada real (`"Yo te perdono por todo"`). El
  troceado, el checkpoint y el paralelismo funcionan. Lo que está roto es lo que la herramienta
  afirma: `OK` significa hoy "`write_text` no lanzó excepción" (`cli/app.py:196-200`).
- **La señal que falta ya está calculada y se tira.** `Segment` de faster-whisper trae
  `avg_logprob`, `no_speech_prob` y `compression_ratio`
  (`.venv/.../faster_whisper/transcribe.py:48-59`); el CLI los aplana a cuatro campos en
  `cli/app.py:185-188` y el checkpoint los vuelve a tirar en `core/chunked.py:107-111` — esta
  segunda pérdida es irreversible sin recomputar.
- **NO construir**: calibrador de confianza + corpus etiquetado, unificar `cli/` con
  `asr/faster_whisper.py`, trocear pyannote, `hallucination_silence_threshold` como default,
  y tocar `log_prob_threshold`/`no_speech_threshold`. Ver sección 4 con motivos.

### Contradicciones reales, resueltas

**C-1. ¿Cómo se mide la cobertura: `duration_after_vad` o suma de segmentos?**
Cuatro consejeros propusieron plomear `info.duration_after_vad` por el checkpoint. Los refutadores
lo tumbaron por tres motivos verificados: (a) mide "lo que el VAD dejó pasar", no "lo que produjo
texto", y con `--no-vad` vale la duración completa (`transcribe.py:879`), o sea es ciego en las
corridas 2, 3 y 5; (b) los trozos ya cacheados no lo traen, así que la suma daría 0% en silencio;
(c) plomearlo rompe la tupla de 3 de `transcribe_chunk` (`core/chunked.py:120,152`) y 5 sitios de
`tests/test_chunked.py`.
**Gana**: `sum(s.end - s.start for s in segments) / info.duration`. Una línea, misma fórmula en las
dos rutas, funciona con y sin VAD y con el caché completo. Sobre los checkpoints reales da 25% para
la corrida 1 y 33% para la corrida 2 — habría gritado en la corrida que reportó `OK`.
`duration_after_vad` no se plomea: el dato equivalente sale gratis del logger de la propia
librería (`transcribe.py:895-897`) con una línea de configuración de logging.

**C-2. F4: ¿barrido de checkpoints o modelo perezoso?**
Tres consejeros propusieron barrer `chunk_path(...)` antes de `core/chunked.py:176`. Verificado:
`chunk_path` hace `audio.stat()` (`core/chunked.py:95`) y los tres tests de `run_chunked` pasan
`Path("x.mp3")`, que no existe → `FileNotFoundError`. Además `core/chunked.py:128` traga el
checkpoint corrupto con `pass` y cae a `model.transcribe` con `model=None` → `AttributeError`.
**Gana** el modelo perezoso (thunk con lock): mismo tamaño de diff, no llama a `chunk_path` dos
veces por trozo, el checkpoint corrupto construye el modelo justo ahí y desaparece el agujero.
**Corrección importante al relato**: este arreglo NO es el fix de la corrida 3. Reconstruido desde
`~/.speechtotext/chunks`, la corrida 3 quitó `--diarize`, eso apagó `word_timestamps`, que está en
la llave del caché (`core/chunked.py:99`), y por lo tanto no tenía ni un trozo reutilizable: murió
cargando un modelo que sí necesitaba. El arreglo se vende por lo que es — elimina el pico de carga
cuando el caché sí sirve — y el desperdicio que el disco documenta de verdad (togglear `--diarize`
invalida el 100% del trabajo) queda como cuestión abierta.

**C-3. ¿`-j 4` multiplica la memoria del modelo?**
El brief lo afirma (F6). **Falso**, medido dos veces de forma independiente: `num_workers` mapea a
`inter_threads` de CTranslate2 (`transcribe.py:694-695`) y no replica pesos (`small`: 502 MB con
`nw=1`, 503 MB con `nw=4`). Lo que mata es el transitorio de carga (~1.77x el estable). **No se
toca `-j`.**

**C-4. F3: ¿la ganancia agresiva ciega al VAD?**
El brief lo afirma. **No está establecido y la medición dice lo contrario**: silero mantiene 999.6 s
de 1001.3 s con -46 dB de atenuación uniforme, y `speechnorm=e=20` no cambió la detección. Lo que sí
fragmenta es el *clipping* (+20 dB: de 1 a 34 segmentos, 69 s perdidos). **Gana la medición**: no se
toca la cadena de audio (ni ganancia, ni VAD propio, ni `clip_timestamps`) hasta reproducir F3 con
instrumentación. Ver cuestión abierta Q1.

**C-5. ¿`language_probability=1.0` es una mentira?**
Tres consejeros dijeron que sí. Verificado: faster-whisper devuelve `language_probability = 1`
cuando se le fuerza el idioma (`transcribe.py:961`), y el default del CLI es `-l es`. O sea, el
`SimpleNamespace` de `core/chunked.py:197` es **fiel** al upstream en el caso por defecto; ponerlo a
`None` haría que la ruta troceada divergiera de la directa.
**Resolución**: `1.0` se queda cuando el idioma se forzó; `None` sólo bajo `--language auto` en ruta
troceada, que es el único caso donde la probabilidad real existió y se tiró
(`core/chunked.py:140-142` sólo guarda `info.language`). Y la fabricación que sí ocurre en el 100%
de las corridas por defecto es otra: el rótulo `Idioma detectado:` sobre el flag del usuario
(`cli/app.py:175`).

**C-6. ¿Dónde se marca un segmento sospechoso?**
Un consejero lo quería dentro del texto en `cli/app.py:185-188` (un solo sitio); otro en los
writers. **Gana el writer**: el campo `text` del JSON queda limpio y el consumidor recibe un
booleano en vez de un string contaminado; los sitios extra son un token cada uno. Con la corrección
del refutador: `write_txt` tiene **tres** sitios de texto (`core/formats.py:44`, `:46` y `:50`), y
el `:50` es la rama sin diarización, o sea el default del CLI y exactamente la que produjo la
evidencia de F2.

**C-7. ¿Las señales nativas por segmento entran ya?**
Dos propuestas sobrevivieron y dos se cayeron. Las que se cayeron tienen razón en el coste: es el
único cambio que invalida el caché (una noche de `large-v3` en CPU), y su consumidor de hoy — la
marca `[?]` — funciona sin ellas por el eje de densidad de caracteres, que atrapa el 100% de la
evidencia de F2 del brief (`"Gracias."` = 8 caracteres en 30 s = 0.27 char/s).
**Resolución**: entran, pero en Fase 2 y **en el mismo commit que el cambio de la llave de caché**,
para pagar una sola invalidación en vez de dos. Forma: tres floats planos con default `None` en
`TimedSegment` y `LabeledSegment`. **No** se importa `SegmentNativeSignals` de `asr/types.py:135`:
es `frozen` con validadores de rango que reventarían al releer un checkpoint, no tiene `temperature`,
y crearía el primer import de `asr/` dentro de `core/` — la frontera entre las dos mitades del repo
se mantiene hasta que haya un motivo mayor.

**C-8. ¿Guarda de memoria con `ctypes`/`psutil` o `try/except`?**
Las dos propuestas de guarda se cayeron: `GlobalMemoryStatusEx` es Windows-only y el proyecto
declara Linux/macOS; y meter `typer.Exit` y una tabla impresa dentro de `core/chunked.py` acopla
core a la capa CLI, que hoy recibe `log=print` inyectado precisamente para no hacerlo.
**Gana** un solo `try/except RuntimeError` alrededor del bloque `if should_chunk(...) / else` de
`cli/app.py:157-172`, que es por donde pasan los dos constructores del camino de transcripción y
que además cubre un OOM durante la decodificación, cosa que una guarda en el constructor no puede.

**C-9. ¿El consenso entre modelos es gratis sobre el caché?**
No. Sólo existen checkpoints si hubo troceo, y `should_chunk` (`core/chunked.py:205-208`) trocea por
encima de 20 min; el tramo del hallazgo dura 18 min, o sea no hay nada que intersectar. Y "dos
modelos no alucinan la misma frase" es n=1: `large-v3` y `medium` comparten corpus de subtítulos, por
eso los dos conocen `"Gracias por ver el video"`. **Gana**: receta documentada, no código. Se decide
con datos (Q3).

---

## 3. Fases

Ordenadas por impacto/coste. Los arreglos de **una línea** van marcados `[1 LÍNEA]`.

### FASE 1 — Que la salida deje de mentir (esta semana, ~6 h)

Objetivo: que ninguna corrida pueda terminar sin declarar cuánto audio produjo texto, que el texto
inventado se distinga del oído, y que el proceso no se juegue la RAM cuando el trabajo ya está en
disco. Cero cambios de parámetros de decodificación, cero invalidación de caché.

---

**1.1 · Cobertura en la línea de resumen** `[1 LÍNEA]`
Archivo: `cli/app.py:174-178`.
Antes del `console.print`, `cov = sum(s.end - s.start for s in segments)`; añadir al mensaje
`· con voz {cov/60:.1f} de {info.duration/60:.1f} min ({100*cov/info.duration:.0f}%)`. Si
`cov/info.duration < 0.7`, imprimir esa parte en amarillo con la sugerencia de probar `--no-vad`.
Se calcula donde ya está la línea, antes de `_run_diarization` (`cli/app.py:180-181`), así que la
partición por palabras no lo altera. Los segmentos de Whisper no se solapan dentro de una pasada y
los trozos son contiguos por construcción (`core/chunked.py:77-78`), así que la suma cruda vale.
Aceptación: re-correr la corrida 1 (`-m medium --diarize`) sobre los checkpoints existentes imprime
un porcentaje entre 24% y 30%, no `OK` a secas.
Coste: 15 min.

---

**1.2 · Segundos con voz y huecos en el JSON**
Archivo: `core/formats.py:84-98` (`write_json`, ya recibe `segments` e `info`: sin parámetros nuevos).
Añadir al payload `"speech_s": round(sum(s.end - s.start for s in seg_list), 2)` y
`"gaps": [[a, b], ...]`, el complemento de los segmentos sobre `[0, info.duration]` filtrando huecos
`< 5.0 s` (`# ponytail: umbral arbitrario, súbelo si el JSON se llena de huecos de respiración`).
No se toca el significado de `"duration"` (`core/formats.py:87`): ahí sí es la duración del archivo
y está bien.
Aceptación: test en `tests/test_formats_speaker.py` — segmentos `(0,1)` y `(40,41)` con
`info.duration=60` producen `speech_s == 2.0` y `gaps == [[1,40],[41,60]]`.
Coste: 30 min.

---

**1.3 · F4: el modelo se construye sólo cuando hace falta**
Archivos: `core/chunked.py:172-186` y `core/chunked.py:140`; `tests/test_chunked.py:170`.
En `run_chunked`, sustituir la construcción incondicional de `core/chunked.py:176` por un thunk con
lock:

```python
import threading
_lk, _box = threading.Lock(), []
def get_model():
    with _lk:                       # ponytail: lock global, basta para jobs<=8
        if not _box:
            _box.append(WhisperModel(
                model_name, device=device, compute_type=compute_type,
                cpu_threads=max(1, (os.cpu_count() or 1) // jobs), num_workers=jobs))
        return _box[0]
```

y pasar `get_model` donde hoy va `model` (`core/chunked.py:184`); en `transcribe_chunk`, la línea
`core/chunked.py:140` pasa a `segments_iter, info = get_model().transcribe(tmp, **opts)`.
Efecto: con el 100% de los trozos cacheados no se construye ningún modelo y el pico de carga
(~3.5 GB en `large-v3` int8) desaparece; con un checkpoint corrupto el modelo se construye en ese
momento y el agujero de `core/chunked.py:128` se cierra solo.
Los tres tests de `run_chunked` (`tests/test_chunked.py:184,204,215`) monkeypatchean
`transcribe_chunk` entero, así que no se tocan. Sí cambia una línea:
`tests/test_chunked.py:170`, donde `model = SimpleNamespace(transcribe=...)` pasa a ser
`model = lambda: SimpleNamespace(transcribe=...)`.
Aceptación: test nuevo en `tests/test_chunked.py` — sembrar el checkpoint de un trozo con
`chunk_path` sobre un archivo real de `tmp_path`, monkeypatchear `chunked.WhisperModel` con algo que
lance `AssertionError`, y verificar que `run_chunked` devuelve el segmento cacheado sin explotar.
Coste: 45 min.

---

**1.4 · Batch de pyannote a 8** `[2 LÍNEAS]`
Archivo: `speakers/diarization.py:119-122`.
Después de `Pipeline.from_pretrained`, `_PIPELINE.embedding_batch_size = _BATCH` y
`_PIPELINE.segmentation_batch_size = _BATCH` con `_BATCH = 8` como constante de módulo. El `32` que
corre hoy viene del `config.yaml` del checkpoint community-1, no del default de pyannote, que es 1
(`pyannote/audio/pipelines/speaker_diarization.py:211-212`). Ambos atributos son asignables y se
leen en tiempo de llamada (`:227`, `:282-287`, `:431`, `:435`): no hace falta `hasattr` ni
`try/except`.
Medido: pico 2620 MB → 1369 MB (-48%) por +7% de reloj sobre 180 s de audio, salida equivalente.
Aceptación: `--diarize` sobre un audio de 3 min produce los mismos turnos que antes y el pico de RSS
del proceso baja por debajo de 1.5 GB.
Coste: 10 min.

---

**1.5 · Marca `[?]` en segmentos sospechosos**
Archivo: `core/formats.py`.
Función pura `is_suspect(seg)` junto a `_speaker` (`core/formats.py:29`):

```python
def is_suspect(seg) -> bool:
    # ponytail: heurística sin calibrar, techo conocido; sube a calibrador si algún día hay corpus
    dur = seg.end - seg.start
    ns = getattr(seg, "no_speech", None)
    if ns is not None and ns > 0.6:          # se enciende sola cuando llegue la Fase 2
        return True
    return dur >= 10.0 and len(seg.text.strip()) / dur < 1.0
```

Prefijar `"[?] "` en los **tres** sitios de `write_txt` (`core/formats.py:44`, `:46`, `:50`), en
`write_srt` (`:63`) y en `write_vtt` (`:76`). En `write_json` (`:89-95`), añadir
`**({"suspect": True} if is_suspect(s) else {})`, con el mismo patrón condicional que ya usa
`speaker`.
Por qué el eje de densidad basta hoy: el español hablado va a 12-15 char/s y el susurro esparcido a
3-5; `"Gracias."` en 30 s son 0.27 char/s. El umbral de 1.0 char/s no toca voz real. Y por qué el
eje de `avg_logprob` no existe aquí: la alucinación de F2 es *confiada* — sobrevive precisamente
porque su `avg_logprob` es alto (`transcribe.py:1215-1224`), así que filtrar por ahí no atraparía
nada.
Marcar, no borrar: un falso positivo cuesta un `[?]` de más; un falso negativo cuesta leer una
invención como si fuera tu conversación.
Aceptación: test en `tests/test_formats_speaker.py` — un segmento de 30 s con `"Gracias."` sale con
`[?]` en txt, srt y vtt y con `"suspect": true` en json; uno de 3 s con una frase normal, no. Los
tests existentes no se mueven: sus segmentos duran 1 s, por debajo del gate de 10 s.
Coste: 1 h.

---

**1.6 · Idioma: decir si se midió o se forzó**
Archivos: `core/chunked.py:197-198`, `cli/app.py:174-176`, `core/formats.py:86`.
- `core/chunked.py:197-198`: `language_probability=1.0 if opts.get("language") else None`. Se
  mantiene el `or "es"` (sólo dispara leyendo checkpoints viejos sin la clave `language`).
- `cli/app.py:175`: si el idioma se forzó (`cli/app.py:141` dejó `lang` no nulo), imprimir
  `Idioma: es (forzado)` sin probabilidad; si fue `auto`, `Idioma detectado: X` y la probabilidad
  sólo si no es `None`.
- `core/formats.py:86`: omitir la clave con el patrón condicional que ya usa `speaker` en `:94`
  (`**({"language_probability": round(p, 4)} if p is not None else {})`), en vez de `round(None, 4)`
  que reventaría con `TypeError`.
Aceptación: `-l es` imprime `(forzado)` y el JSON trae `language_probability: 1.0`; `-l auto` con
troceado omite la clave; los tests existentes (`tests/test_formats_speaker.py:35,45`, que pasan
`1.0`) siguen verdes.
Coste: 30 min.

---

**1.7 · Cobertura por trozo en el log** `[1 LÍNEA]`
Archivo: `core/chunked.py:192`.
Sustituir el `OK` por el porcentaje del trozo:
`log(f"[{done}/{len(chunks)}] {_mmss(s)}-{_mmss(e)} {100*sum(x.end-x.start for x in results[i])/(e-s):.0f}% ({tag})")`.
`results[i]` ya está asignado en `core/chunked.py:189`, y trae timestamps globales, así que el
denominador tiene que ser `e - s`.
Aceptación: el trozo 1200-1800 de la corrida 1 (checkpoint `96f...`, `segments: []`) imprime `0%`.
Coste: 10 min.

---

**1.8 · Logger de faster-whisper visible** `[1 LÍNEA]`
Archivo: `cli/app.py`, junto al setup de consola (`cli/app.py:44-51`).
`logging.basicConfig(level=logging.WARNING)` + `logging.getLogger("faster_whisper").setLevel(logging.INFO)`.
La librería ya emite `"VAD filter removed %s of audio"` (`transcribe.py:895-897`), una vez por
trozo en ruta troceada, y hoy nadie configura logging en `cli/` ni en `core/`.
Aceptación: una corrida con `--vad` imprime cuánto audio quitó el VAD por trozo. Es el instrumento
que cierra Q1.
Coste: 10 min.

---

**1.9 · Guarda de memoria en el único punto que las cubre**
Archivo: `cli/app.py:157-172`.
`try/except RuntimeError` alrededor del bloque `if should_chunk(...) / else` completo: si
`'alloc' in str(e).lower()`, imprimir el mensaje accionable (`large-v3` pide ~3.5 GB al cargar sobre
un `model.bin` de 3.087.284.237 B; cierra procesos o usa `-m medium`) y `typer.Exit(1)`; re-lanzar
cualquier otro `RuntimeError`. Sin `psutil`, sin `ctypes`, sin sondear RAM del sistema: reaccionar
al fallo real es más barato y más exacto que estimar el pico.
No cubre la muerte nativa de la corrida 2 (exit 5, que es pyannote/torch) — y el mensaje no debe
prometerlo. Ver Q2.
Aceptación: test que monkeypatchea `run_chunked` para lanzar
`RuntimeError("mkl_malloc: failed to allocate memory")` y verifica salida 1 con mensaje, y otro
`RuntimeError` cualquiera que se propaga sin tragar.
Coste: 45 min.

**Total Fase 1: ~5 h de trabajo.** No invalida ningún checkpoint. No cambia ningún parámetro de
decodificación.

---

### FASE 2 — Evidencia por segmento y caché honesto (~6 h + una noche de máquina)

Objetivo: que cada segmento viaje con los números con que se emitió, y que el caché deje de poder
devolver resultados calculados con otros parámetros. Todo esto va **en un solo commit**: es el único
cambio que invalida los checkpoints y no se paga dos veces.

---

**2.1 · Tres señales nativas hasta el disco y hasta el JSON**
Archivos: `core/chunked.py:28-33`, `:36-48`, `:107-117`; `core/segments.py:7-12`;
`cli/app.py:185-188`; `speakers/diarization.py:37,51,56,77`; `core/formats.py:88-97`.
Añadir `no_speech: float | None = None`, `avg_logprob: float | None = None`,
`compression_ratio: float | None = None` **al final y con default** en `TimedSegment`
(`core/chunked.py:28`) y `LabeledSegment` (`core/segments.py:7`) — todas las construcciones
existentes son posicionales, así que los defaults al final no rompen nada.
Propagar en **seis** sitios de construcción, no cuatro: `shift_segments` (`core/chunked.py:47`),
`seg_to_dict`/`seg_from_dict` (`core/chunked.py:107-117`, con `d.get(...)`, nunca `d[...]`, o
revienta `tests/test_chunked.py:143`), la comprensión de `cli/app.py:185-188`, y
`speakers/diarization.py:37`, `:51`, `:56` más `apply_names` (`:77`) — sin estos últimos, con
`--diarize` (que es justo la corrida 1) las señales se destruyen otra vez.
En la rama de `word_timestamps` de `speakers/diarization.py:42-56` un segmento se parte en varios
runs: los N runs heredan el mismo `no_speech` del padre. Es una aproximación correcta (vienen de la
misma ventana de decodificación) y **va documentada en el comentario**, o se miente de nuevo.
`write_json` (`core/formats.py:88-97`) emite los tres por segmento, o el dato muere en el caché
igual que hoy.
No se importa `SegmentNativeSignals` (`asr/types.py:135-147`): ver C-7.
Aceptación: (a) test de roundtrip `seg_to_dict`/`seg_from_dict` con y sin señales; (b) un checkpoint
viejo sin las claves se lee sin excepción y devuelve `None`; (c) el JSON de una corrida nueva trae
los tres números por segmento.

---

**2.2 · La llave del caché deja de ser una lista escrita a mano** `[1 LÍNEA]`
Archivo: `core/chunked.py:96-100`.
Sustituir la enumeración de `opts["language"], opts["beam_size"], ...` por
`json.dumps(opts, sort_keys=True)` dentro del `key`.
**Sin `default=str`**: un objeto sin `__repr__` estable metería la dirección de memoria en la llave
y el caché dejaría de acertar para siempre en silencio, que es exactamente el fallo que esto viene a
matar. Que `json.dumps` reviente ruidoso es la falla correcta.
**Sin constante `CACHE_VERSION`**: cambiar la composición de la llave ya invalida el 100% de los
checkpoints de una vez; la constante se agrega el día que haga falta invalidar *sin* cambiar la
llave.
Hoy el único campo de `opts` fuera de la llave es `condition_on_previous_text`, que es constante
(`cli/app.py:109`), así que no hay bug vivo: es el prerrequisito para que el primer cambio de
parámetros sea evaluable.
Aceptación: `test_chunk_path_determinista` y `test_chunk_path_cambia_con_cada_parametro`
(`tests/test_chunked.py:104-123`) siguen verdes; una corrida sobre el audio del caso imprime cuatro
`(nuevo)` donde antes decía `(cache)`.

---

**2.3 · `device` y `compute_type` entran en la identidad del caché** `[1 LÍNEA]`
Archivo: `core/chunked.py:184`.
`int8` y `float32` no decodifican igual, y hoy ninguno de los dos está en la llave porque viajan
aparte de `opts` (`core/chunked.py:172`, `cli/app.py:159-162`). El sitio de menor diff: en el
`pool.submit` de `core/chunked.py:184`, pasar `f"{model_name}|{device}|{compute_type}"` en lugar de
`model_name`. `transcribe_chunk` sólo usa ese argumento para `chunk_path`
(`core/chunked.py:123`), así que no cambia ninguna firma ni ningún test.
Aceptación: dos corridas idénticas salvo `--compute-type` producen rutas de checkpoint distintas.

---

**2.4 · Reportar la calidad de la diarización**
Archivo: `cli/app.py:304-310` (`_run_diarization`).
Con los datos que ya están en memoria en esas líneas: imprimir número de hablantes detectados y
porcentaje de segmentos sin atribuir, y sugerir `--speakers N` cuando el conteo automático supera lo
razonable. El brief dice literal que `--speakers` existe y nada lo sugiere cuando el resultado es
absurdo.
Aceptación: sobre la corrida 1, la salida dice "3 hablantes detectados, X% sin atribuir — prueba
`--speakers 2`".

**Total Fase 2: ~6 h + una noche de CPU** para recomputar los 36:46 con `large-v3` una sola vez.

---

### FASE 3 — Gatillada por datos, no agendada

Nada de esto se planifica hasta que las cuestiones abiertas se cierren con una medición. Se listan
para que quede claro cuál es el disparador de cada una:

- **`vad_parameters` como dial del CLI** (`threshold`, `min_silence_duration_ms`, `speech_pad_ms`) en
  `_transcribe_opts` (`cli/app.py:104-111`). Disparador: que la corrida instrumentada de Q1 muestre
  VAD ciego (no saturado). Hoy el usuario sólo puede elegir entre tirar 18 minutos (`--vad`) o
  alucinar sobre ellos (`--no-vad`); medido, `threshold` 0.50 → 0.20 cambia la fragmentación de 25 a
  4 segmentos. Prerrequisito ya cumplido en Fase 2: la llave del caché cubre `opts` completo.
- **Consenso entre modelos como subcomando** (nunca como default, nunca dos modelos vivos en el
  mismo proceso). Disparador: Q3.

---

## 4. Lo que NO se construye

| Descarte | Motivo |
|---|---|
| Calibrador de confianza + corpus etiquetado (`confidence/calibration.py`, `evaluation/`) | Exige ≥9 fechas de grabación (`evaluation/splits.py:122-123`), ≥30 min por partición y transcripción humana de referencia por clip, sobre un corpus en `D:\AudioBench` que no existe. Y el calibrador se auto-invalida con `ValueError` ante cualquier cambio de `--model`/`--beam-size`/`--hotwords`/`--language` (`confidence/calibration.py:231-258`): un calibrador por combinación de flags, en una herramienta personal. La marca `[?]` de 1.5 se lleva el 80% del valor por el 2% del coste. |
| Unificar `cli/app.py` con `asr/faster_whisper.py` | El backend exige `VerifiedModelArtifact` con SHA-256 por archivo desde `D:\Models` (no existe), fuerza `vad_filter=False` (`asr/faster_whisper.py:150`), no trocea, no cachea y no diariza. Es un rediseño del único camino que hoy entrega algo, no un cableado. |
| `hallucination_silence_threshold` | Es el único parámetro que ataca F2 de frente y es exactamente el que no se puede usar aquí: **borra sin dejar rastro** (`transcribe.py:1337`, `current_segments[si:] = []`) y marca como anómala toda palabra con `probability < 0.15` — el perfil exacto del susurro a -46 dB que el consenso rescató. Cambia una pérdida ruidosa por una silenciosa. Ver Q4 para medirlo sin encenderlo. |
| Tocar `log_prob_threshold` / `no_speech_threshold` | Es un canje directo F2-por-F1, no una mejora: endurecerlos bota más susurro, y el usuario tiene 18 minutos de susurro. La respuesta a "no distingo invención de voz" es marcar, no botar. |
| VAD propio, AGC, gain staging o `clip_timestamps` en el camino de `transcribe` | El mecanismo de F3 que enuncia el brief está contradicho por medición (C-4). Construir la solución de un fallo cuyo mecanismo no se conoce garantiza dos bugs en vez de uno. |
| Trocear la diarización | La memoria de pyannote está dominada por el batch, no por la duración (2579 MB a 60 s vs 2732 MB a 540 s), y ya satura los hilos vía torch. Trocear sólo compra checkpoint/resume, y a cambio mete identidad entre trozos más hablantes fantasma al forzar `--speakers` en tramos de monólogo (`pyannote/.../clustering.py:626-642`). El 1.4 baja el pico un 48% con dos líneas. |
| Filtro de turnos cortos (`< 0.4 s`) en `assign_segments` | Mecanismo falso: `_best_speaker` (`speakers/diarization.py:11-20`) agrega por **solape**, no por conteo, así que un turno de 17 ms aporta 0.017 s y sólo puede voltear una palabra que caiga en un hueco de todos los turnos largos. Y el síntoma observado (3 clusters en una conversación de 2) no lo arregla filtrar turnos. |
| `psutil` como dependencia | Dependencia nativa nueva en un stack que ya convive con dos `libiomp5md.dll` distintos, para predecir lo que 1.9 detecta reaccionando. |
| `pool.shutdown(wait=False, cancel_futures=True)` | En el caso observado es un no-op exacto: `core/chunked.py:175` hace `jobs = max(1, min(jobs, len(chunks)))`, así que con 4 trozos y `-j 4` los cuatro futures están corriendo y `cancel()` devuelve `False` para todos. Se agrega el día que haya más trozos que jobs. |
| Taxonomía de exit codes / estado "degradado" | Infraestructura para un consumidor que no existe: la herramienta se usa interactivamente y no hay automatización aguas arriba. Con la cobertura de 1.1 el usuario ve el fallo. Exit code nuevo cuando aparezca el primer script que lo lea. |
| Protección contra sobrescritura de artefactos | El fallo es imaginado, no observado: el brief no reporta un solo artefacto perdido, y `--output/-o` ya resuelve el caso (`cli/app.py:208-210`, `_resolve_output_base` en `:54-63`). Sobrescribir es la convención de este tipo de CLI. |
| `Gap(start, end, reason)` como dataclass nueva | El vocabulario de razones que se citaba (`audio/gate.py:9-18`) no contiene las dos razones que harían falta, y hoy no hay de dónde sacar un `reason` honesto. La laguna es el complemento de los segmentos sobre `[0, duration]`: eso es 1.2, seis líneas. |
| Un `opts` por régimen acústico dentro de `plan_chunks` | Maquinaria sin carga: nadie pudo decir qué parámetro debería cambiar por régimen, y el único candidato obvio es el canje F2-por-F1 que ya se descartó. Además rompe ~9 tests de `tests/test_chunked.py`. Se reevalúa cuando exista una medición por tramo. |
| Unificar `"Hablante ?"` entre formatos | La asimetría (`core/formats.py:40` en txt vs `:61-63` y `:74-76` en srt/vtt) no le costó nada a nadie, y meter el prefijo en subtítulos degrada el formato con el que el usuario **sí** detectó el fallo. |
| `BatchedInferencePipeline` | Mantiene estado mutable entre llamadas (`transcribe.py:117`, `:162`), o sea no es reentrante, y chocaría con el `ThreadPoolExecutor` de `core/chunked.py:182`. Cero relación con los fallos observados. |
| Guarda de memoria en `find` (`core/finder.py:94`) | El índice corre con `tiny`/`int8`, que es el caso que nunca revienta. |

---

## 5. Cuestiones abiertas

**Q1 · ¿El VAD estuvo ciego o saturado en la corrida 4?**
"1 segmento en 18 minutos" es ambiguo con el código actual: `cli/app.py:177` cuenta segmentos de
Whisper, no chunks del VAD, y un VAD que no detectó nada y uno que detectó todo como una sola región
producen la misma línea. **Dato que falta**: una corrida sobre el tramo amplificado real con el
logger de 1.8 activo, leyendo `"VAD filter removed X of audio"`. Bloquea toda decisión sobre gain
staging y sobre exponer `vad_parameters`.

**Q2 · ¿Qué mató la corrida 2 (exit 5, sin mensaje)?**
Ninguna ruta Python del árbol produce exit 5; `0xC0000005` (access violation) truncado a 8 bits da
exactamente 5, o sea fue muerte en código nativo. Nadie probó que fuera OOM. **Dato que falta**:
reproducir `--diarize` con el batch de 1.4 ya aplicado y ver si sobrevive. Si sobrevive, era presión
de memoria y está resuelto; si no, hace falta correr la diarización en un subproceso para convertir
la muerte nativa en un mensaje, y eso es una decisión de arquitectura que hoy nadie tiene datos para
justificar. Además: el `try/except` de `_run_diarization` (`cli/app.py:290-300`) cubre sólo
`diarization.diarize`; `assign_segments` (`:304`), `registry.get_embeddings` (`:307`) y
`assign_names` (`:309`) revientan con traceback crudo.

**Q3 · ¿Cuánto añade el consenso entre modelos sobre la marca `[?]`?**
Es el hallazgo más fuerte del brief y a la vez n=1 sobre un clip. **Dato que falta**: después de la
Fase 1, correr el consenso una vez más sobre el mismo tramo y medir cuánta alucinación atrapa que la
marca `[?]` no atrapó, y cuánta voz real rescata que la marca no ensució. Con `evaluation/metrics.py`
(`word_error`, que sólo depende de `unicodedata`, `numpy` y `statistics`) eso se mide sin construir
corpus. Si añade poco, se ahorró un mes. Mientras tanto: dos invocaciones y un diff, documentado en
el README.

**Q4 · ¿`hallucination_silence_threshold` sirve para algo aquí?**
Está prohibido encenderlo (sección 4), pero nadie lo midió. **Dato que falta**: una corrida
diagnóstica sobre el tramo amplificado con `word_timestamps=True` y el umbral puesto, comparando qué
segmentos habría borrado contra las 8 islas de voz que el consenso rescató. Ojo: `word_timestamps`
está atado a `--diarize` (`cli/app.py:156`) y entra en la llave del caché
(`core/chunked.py:99`), así que la medición cuesta una recomputación completa. Hacerla junto con la
noche de máquina de la Fase 2.

**Q5 · ¿El checkpoint debe depender de `word_timestamps`?**
Hallazgo del disco, no de una propuesta: togglear `--diarize` cambia `word_timestamps`, que está en
la llave (`core/chunked.py:99`), e invalida el 100% del trabajo — el usuario recomputó 36 minutos de
`large-v3` en CPU sólo por quitar una bandera que no cambia el texto, sólo si se guardan timestamps
de palabra. Un checkpoint **con** palabras es un superconjunto de uno sin ellas, así que podría
servir a ambas corridas. **Dato/decisión que falta**: confirmar que `word_timestamps=True` no altera
el texto emitido (la ruta de `add_word_timestamps` ajusta fronteras de segmento); si no lo altera,
sacar `word_timestamps` de la llave y descartar las palabras al leer cuando no se piden. Es el
desperdicio más grande que documenta el disco y no lo cubre ninguna fase.

**Q6 · Huérfanos en `~/.speechtotext/chunks`**
No hay recolección: cada cambio de llave deja checkpoints huérfanos para siempre, y ya hay tres de
34 bytes con `{"language": "es", "segments": []}` — el resultado del bug de F1 congelado en disco.
La Fase 2 va a dejar cuatro más. **Decisión que falta**: si hace falta un `--prune` o basta con
borrar la carpeta a mano. No bloquea nada.
