# Plan definitivo — speechtotext multimotor (v1)

Fecha: 2026-07-27. Decisión de Samuel (no se relitiga): speechtotext pasa a ser multimotor para
exprimir el 5900X (faster-whisper) y la GTX 980 (whisper.cpp CUDA). La junta deliberó el CÓMO.
Este documento es la síntesis ejecutable: solo material sobreviviente o refutado-con-corrección
(se usa la corrección). Cada cifra lleva su clase: [medido] / [citado] / [supuesto].

---

## 1. La decisión y la evidencia

**Se decidió**: añadir whisper.cpp CUDA (binario prebuilt v1.9.1, subprocess) como segundo motor
del CLI batch, detrás de un flag `--engine` explícito con default intacto, con large-v3 q5_0 en
la 980.

**Los números que la sostienen** (todos de la máquina real, 2026-07-27):

- Calidad, ventana limpia de 60 s (ref curada, 31 palabras): GPU large-v3 q5_0 empata EXACTO con
  el techo CPU (large-v3 int8): WER 0.355 = 0.355 (11/31 ambos). El motor actual del repo
  (small int8) da 0.419 (13/31). [medido]
- Calidad, tramo susurrado de 18 min (ref de islas, 61 palabras): q5_0 WER 2.164, 5/8 islas
  recuperadas; small int8 WER 3.508, 3/8 islas. Las corridas CPU lv3/medium que aparentan ganar
  están descartadas como comparación: la referencia se curó desde esas mismas corridas (sesgo
  declarado). Ninguna medición limpia muestra a q5_0 por debajo del techo CPU. [medido]
- Velocidad, tramo de 18 min: GPU q5_0 con `-mc 0` = 56 s (19.3x tiempo real); sin `-mc 0` = 151 s
  con loops de repetición; CPU small secuencial = 385 s. En el clip60 denso: GPU large-v3 4.4x vs
  CPU batcheado small 15.6x — el argumento del multimotor es calidad + CPU liberado, no velocidad
  bruta en audio denso. [medido]
- VRAM: el proceso q5_0 usa ~1.28 GB; pico de sistema 3800/4096 MiB con el escritorio encendido
  (baseline 1.4–2.7 GB). fp16 (3.09 GB) carga pero pagina bajo WDDM: 0.53x tiempo real, 25x más
  lento por batch — inusable. [medido]
- `--prompt` de whisper-cli es INERTE bajo `-mc 0` (salida bit-idéntica en 7 corridas con control
  positivo); con mc default el prompt sí trabaja pero contamina la ortografía de toda la corrida.
  Esto decide la política de hotwords: rechazo, no degradación. [medido]
- large-v3 q8_0 NO existe publicado en ggerganov/whisper.cpp (el ~1.7 GB del brief era estimado de
  un archivo inexistente); habría que cuantizarlo localmente desde fp16. [medido]
- El caché de chunks NO está vacío: 2 checkpoints del diagnóstico de hoy. El coste de invalidarlos
  sigue siendo despreciable (huérfanos que se recomputan), pero el argumento del orden de PRs es
  la costura compartida, no un caché vacío. [medido]

---

## 2. La arquitectura elegida

### 2.1 La costura: factory `make_engine`, opción (a)

`make_engine(engine, model, device, compute_type, **kwargs)` inyectada en los DOS únicos puntos de
construcción del modelo:

- el thunk de `run_chunked` (core/chunked.py:184-190) — `**kwargs` conserva `cpu_threads`/
  `num_workers` que el thunk hoy calcula desde `jobs` (corrección a V1: la firma original los
  omitía);
- la ruta directa (cli/app.py:173).

`transcribe_chunk` y todo el troceado+resume no se tocan. El default `faster-whisper` sigue
construyendo por el nombre `chunked.WhisperModel`, así los **7** monkeypatch existentes
(test_chunked.py:186,206,219,242,254,266,277 — son siete, no seis) y todo test_cli.py sobreviven
sin cambios.

**El contrato mínimo es la superficie de WhisperModel, duck-typed** (medido en todos los
consumidores): `transcribe(path, **opts) -> (iterable de segmentos {start, end, text, words|None},
info {language, language_probability|None, duration solo en ruta directa})`. No se crea tipo
intermedio: tiparlo obligaría a empobrecer (intersección) o fabricar campos (unión que miente).
El contrato se fija con el test de contrato dual (2.7), no con un Protocol.

**Ley de orquestación** (ya vigente, ahora enunciada): los segmentos pertenecen al motor; `info`
pertenece al orquestador. `run_chunked` ya fabrica info con `probe_duration` (chunked.py:216-218);
el adaptador whisper.cpp hace lo mismo en ruta directa (`duration=probe_duration(audio)` —
obligatoria: app.py:206-208 y :231 la consumen; omitirla es AttributeError en la primera corrida).

**Ubicación del adaptador — contradicción resuelta**: el dossier de costura proponía
`asr/whispercpp.py`; **gana Stride**: el módulo vive del lado de `core/` (`core/engines.py` +
`core/enginepin.py`). Motivo: el plan de calidad prohíbe explícitamente el primer import de `asr/`
dentro de `core/` (frontera C-7), y hoy nadie en core/ ni cli/ importa `speechtotext.asr` (grep
verificado). Las fronteras se pisan una vez y quedan pisadas.

### 2.2 El contrato de capacidades: dict CAPS, obligatorio, consultado antes de construir

Dict `CAPS` por engine **junto a `make_engine` en core/engines.py** (corrección a V2: "en cada
adaptador" no funciona porque faster-whisper no tiene adaptador). Declara por capacidad:
`honrado | mapeado | rechazado | degradado-con-aviso` — generalizado a TODO el opts dict
(corrección de Serrano: no solo hotwords; `vad_filter` y `word_timestamps` también viajan en opts
y en la llave).

`transcribe_file` lo consulta en la validación de flags (cli/app.py:141-161, antes del import de
chunked en :162), **antes de construir modelo alguno**. Regla única: se degrada con aviso cuando
el resultado sigue siendo lo pedido con menos precisión; se rechaza cuando el knob sería inerte o
semánticamente distinto; **jamás silencio, jamás sustitución callada**. El CAPS no es opcional
(gana Voronov sobre el dossier de costura): `--hotwords` y `--diarize` ya existen hoy — sin CAPS,
la primera corrida whispercpp miente por omisión.

**Contradicción resuelta — hotwords**: V2/H5 proponían "degradación con aviso"; **gana G1
corregido (rechazo explícito)**, porque la medición del 27/07 cerró el debate: `--prompt` es
inerte bajo `-mc 0` (obligatorio por calidad-q5), y sin `-mc 0` el sesgo contamina la ortografía
global [medido]. Avisar "degradado" sobre un knob inerte fabrica un efecto que no ocurrió — la
variante más cara de mentir. Consecuencia: hotwords jamás entra al join de `chunk_path` bajo
whispercpp (por construcción: la corrida rechazada no llega al caché).

### 2.3 La identidad: llave de caché completa, cambiada UNA vez

`chunk_path` (core/chunked.py:95-104) suma al join: `engine`, `compute_type` **efectivo**
(la cuantización real, ver 2.4) y `device`. Sin esto, faster-whisper int8 y whisper.cpp q5_0 con
`model='large-v3'` producen el MISMO digest: hit silencioso que sirve texto de un motor bajo la
firma del otro, y en resume parcial un transcript híbrido sin marcar — la violación más directa
de "la salida no miente".

Decisión de implementación (corrección a V3, se decide aquí y no en el PR): los campos nuevos
entran como **kwargs con default** que reproducen el comportamiento actual, así los 5+ call sites
de tests (test_chunked.py:109-123,148,157,178,236) sobreviven sin tocar.

A la llave entran los **opts EFECTIVOS post-mapeo, no los pedidos** (omisión de Serrano,
adoptada): bajo whispercpp, `vad_filter` y `word_timestamps` efectivos son False — `--vad` vs
`--no-vad` producen el mismo digest, porque la salida del motor es idéntica. Sin esto, recompute
fantasma que degrada el resume por la puerta de atrás.

La llave cambia UNA sola vez, en PR-1. El payload del checkpoint no entra al digest, así que la
Fase 2 de calidad (PR-2) no invalida nada.

### 2.4 Los knobs bajo whispercpp (omisión de Voronov, cerrada)

- `compute_type='auto'` con device≠cpu resuelve hoy a `float16` (app.py:150-151) — exactamente la
  configuración medida como inusable. El adaptador mapea explícito: tabla pinneada
  `{auto: q5_0, q5_0: q5_0}` en v1; cualquier otro `compute_type` con whispercpp → **rechazo** con
  mensaje que cita la medición (fp16 = 0.53x tiempo real por paging WDDM [medido]). Lo efectivo
  (q5_0) es lo que entra a llave y JSON.
- `jobs`: clamp a 1 con aviso cuando engine=whispercpp y device=cuda (H2). Motivo: el default
  jobs=4 (app.py:322-324) aplica automático a todo audio >20 min (chunked.py:222-228); 4
  subprocesses × 1.28 GB = 5.1 GB contra 4096 MiB con 1.4–2.7 GB de escritorio [medido]; WDDM no
  revienta, pagina 25x en silencio (proxy fp16 [medido]). El 19.3x medido hace innecesario el
  paralelismo; el clamp conserva checkpoint/resume. Coste aceptado: recarga del modelo por trozo,
  acotada ≤5 s por trozo de 600 s (<4%) [medido].
- `--engine whispercpp --device cpu`: **derogado en la implementación (2026-07-27)**. El
  smoke midió que el binario pinneado es build CUDA y corre en la GPU siempre — el
  adaptador no pasa device, así que "whispercpp en CPU" no existe con este binario
  (requeriría `-ng`, post-ship). El device efectivo es `cuda` en toda corrida whispercpp:
  se declara en header/llave/JSON y el remapeo se avisa en consola cuando no pidieron cuda.

### 2.5 Selección de motor: explícita, invariantes de auditoría como ley

`--engine {faster-whisper, whispercpp}`, default `faster-whisper` byte a byte (V6). Cero
auto-selección en v1: la señal que una heurística necesitaría no existe (VRAM libre fluctúa
1.4–2.7 GB con el escritorio [medido]); con un dueño y dos motores, el flag ES la política.

**G3 partido en dos (corrección adoptada)**: los invariantes se ratifican como ley desde v1 —
el motor RESUELTO (jamás un literal 'auto') entra a llave, header y JSON; cero fallback
silencioso: binario ausente o sha que no cuadra = **error con causa**, nunca cambio callado de
motor; transcript multi-motor mezclado prohibido por construcción. El flag `auto` mismo queda
como spec futura con su sonda definida (exe verificado presente + driver responde); no añade
código hoy.

### 2.6 La salida declara el motor — siempre (G2)

- Header: `Motor whisper.cpp v1.9.1 · large-v3 q5_0 · cuda` (paralelo exacto de app.py:157-161).
- Línea de resumen final (app.py:230-233): se añade el motor.
- JSON (formats.py:115-142): bloque `engine` = `{name, version, model, quant, device,
  selection: 'explicit'}` — `selection` es constante en v1 (corrección G2: mantener la clave evita
  otro cambio de payload cuando exista auto). Bajo `--diarize` se añade
  `diarization: 'segment'|'word'` DENTRO del bloque engine (corrección G4: un solo hogar para la
  metadata de motor). Cambio de payload aditivo, en el mismo PR que la llave.
- Cuando exista `--engine-binary` (post-ship), su sha256 entra a JSON **y a la llave** (corrección
  G2); en v1 no hay binario ajeno.

### 2.7 Señales ausentes: omitir, jamás rellenar (G5) — fijado con test de contrato dual (V5)

Ley del multimotor, ratificando la convención 1.6 ya escrita: `language_probability`, `no_speech`
y toda señal de Fase 2 que el motor no emita se OMITE del JSON y del checkpoint (formats.py:121-123
omite; chunked.py:213-217 se niega a fabricar). Prohibido rellenar con 0.0/1.0/defaults; prohibido
que el adaptador "complete" el SimpleNamespace para parecerse a faster-whisper.

Se fija con UN test de contrato que pasa la salida de AMBOS motores por el pipeline completo:
shift_segments → roundtrip seg_to_dict/from_dict → cobertura → writers → assign_segments en modo
grueso; aserta presencia/ausencia exacta de claves por motor. **La fixture JSON de `-ojf` se graba
ANTES de escribir el parser**, de una corrida real en la 980 (hoy posible con scratchpad/wcpp/) —
corrección a V5: fixture nacida del mismo entendimiento que el parser sería un test circular.

### 2.8 La frontera del subprocess (H1) y sus fallos (V4/G6)

Adaptador whisper.cpp:

- Input SIEMPRE wav 16k mono temporal (la ruta troceada ya lo garantiza, chunked.py:137-142; la
  directa reusa `transcode_to_wav`, jamás pasa el path del usuario por argv — whisper-cli es
  `main(char**)`, argv en ANSI).
- `%TEMP%` no-ASCII: se detecta y se falla con mensaje claro (corrección H1: mkstemp no lo
  garantiza, lo hereda).
- Comando: `-m <ggml> -f <wav> -l <lang> -bs <beam> -mc 0 -np -ojf -of <base-temp>`. El `-mc 0` es
  HARDCODEADO (paridad con condition_on_previous_text=False, app.py:117; sin él: loops y 3x el
  tiempo [medido]; `-nc` no existe en v1.9.1 [medido]).
- Lectura del JSON con `encoding='utf-8'` explícito (Python en Windows abre cp1252 por default).
- Parser: `offsets.from/.to` en ms enteros → segundos; `words=None`; info =
  `SimpleNamespace(language=result.language, language_probability=None)` + `duration` solo en
  directa vía probe_duration. Pérdida de precisión en salida: ninguna neta (format_timestamp emite
  exactamente ms) [medido].
- Timeout: `subprocess.run(cmd, timeout=max(120, 4*duracion))`; en ruta directa la duración sale
  de `probe_duration` (corrección H3). El piso 120 s se congela DESPUÉS de la medición H6 del JIT
  frío de large-v3.
- Errores: exit≠0 → `RuntimeError(f'whisper-cli rc={rc}: ...')` con cola de stderr decodificada
  `errors='replace'`; exe ausente, sha que no cuadra y JSON malformado → error con causa. El
  handler OOM (app.py:182-198) NO se extiende — pero como el stderr de CUDA OOM contiene 'alloc'
  y dispararía el consejo de RAM de CPU (falso amigo, corrección V4), el consejo del handler se
  vuelve consciente del engine: una rama que, bajo whispercpp, aconseja VRAM (cerrar apps GPU /
  modelo menor / `--engine faster-whisper`). En ruta troceada, el mensaje nombra motor y trozo:
  lo arma `run_chunked` al capturar (futs ya mapea future→índice, chunked.py:194-201) — el dueño
  del mensaje es el troceador, no el adaptador (corrección G6).
- `finally`: unlink del wav y del .json con el patrón de chunked.py:146-150 (el .json puede no
  existir si el exe murió antes).
- `cancel_futures=True` al primer fallo en el loop de as_completed (chunked.py:199-208, ~3
  líneas): sin esto, con motor roto cada trozo encolado paga ffmpeg+spawn+fallo — con fallos
  LENTOS (paging, timeout) el drenaje de 17 trozos serían horas (H4).

### 2.9 Binario y modelos: pin duro, sha propio, fuera de la fortaleza

- `ENGINE_PIN` = constante {version 'v1.9.1', URL del asset, sha256 del zip
  `106a2030...b601b` [medido, servido por la API de GitHub], sha del exe tras extraer}.
  Descarga única (678 MB), verificación sha256 ANTES de extraer, extracción a
  `%LOCALAPPDATA%\speechtotext\whisper-cpp\v1.9.1\` **conservando el prefijo `Release\`** tal
  como extrae el zip (decisión de layout, corrección HOY: cero código de aplanado; el pin apunta
  a `...\v1.9.1\Release\whisper-cli.exe`). El zip es autocontenido (DLLs CUDA propias, solo exige
  driver) [medido]. Fallo cerrado si el sha no cuadra. Subir el pin = commit consciente con
  re-benchmark en la 980 (riesgo PTX 500 sin garantía futura).
- Modelos ggml: `hf_hub_download('ggerganov/whisper.cpp', ...)` (huggingface_hub 1.22.0 ya en el
  venv; con symlinks/xet off el snapshot es archivo real). Como sin xet solo se valida tamaño, el
  sha256 lo verificamos nosotros tras descargar, contra tabla pinneada en código:
  `large-v3-q5_0 = 1.081.140.203 bytes / d75795ec...ad1` [medido]. Se reutiliza el patrón de
  `_sha256_stream` (manifest.py:47-55), NO la cadena VerifiedModelArtifact (DACL+lease
  incompatibles con el cache HF; sha pinneado es ascenso de seguridad neto sobre la ruta viva
  actual, que no verifica nada).

---

## 3. Orden de PRs

**Motivo del orden**: los tres frentes comparten UNA costura física — `chunk_path` y los dos
puntos de construcción del modelo. Solo los campos del join de `chunk_path` invalidan el caché;
el payload del checkpoint no entra al digest. Por tanto: la única invalidación viaja con PR-1
(multimotor, que necesita la llave sí o sí), PR-2 queda libre de invalidación por construcción,
y GPU F2 no toca la costura y se parquea por dato. Si calidad F2 fuera primero, la llave
cambiaría dos veces — exactamente lo que docs/plan-calidad-transcripcion.md:129 prohíbe.

**Contradicción resuelta — señales nativas en PR-1 o PR-2**: Stride las metía en PR-1 "para pagar
una sola invalidación"; **gana su propia corrección**: la captura NO toca la llave (el payload no
entra al digest) y sin el puente LabeledSegment las señales mueren en app.py:240-243 de todos
modos. Captura + puente + writers van JUNTOS en PR-2, coherente y sin invalidación. PR-1 queda
más chico y llega el miércoles.

### PR-1 — Multimotor v1 + llave única + contrato con el usuario (coste M, ~300-350 líneas prod + ~250 tests, ~13 h; merge objetivo miércoles 29-jul)

Cambios concretos:

1. `src/speechtotext/core/engines.py` (nuevo): `make_engine(...)`, dicts `CAPS` por engine,
   adaptador whisper.cpp completo (comando, parser, mapeo compute_type→quant, timeout, traducción
   de errores, temp ASCII, cleanup) — todo lo de 2.8 y 2.4.
2. `src/speechtotext/core/enginepin.py` (nuevo): `ENGINE_PIN`, tabla ggml pinneada,
   descarga+verificación+extracción, adopción de instalación local existente vía el zip
   verificado (2.9).
3. `core/chunked.py`: rama por engine en el thunk (:184-190, default por nombre
   `chunked.WhisperModel`); `chunk_path` (:95-104) suma engine/quant-efectiva/device como kwargs
   con default y consume opts efectivos; `cancel_futures` al primer fallo (:199-208); mensaje de
   fallo con motor+trozo (:194-201).
4. `cli/app.py`: flag `--engine`; validación CAPS antes de construir (:141-161, antes del import
   en :162): hotwords→rechazo, vad→aviso, diarize→aviso; clamp jobs=1 whispercpp+cuda
   (:322-324); ruta directa vía make_engine (:173) con `duration=probe_duration`; header/resumen
   con motor (:157-161, :230-233); consejo OOM por engine (:182-198); suprimir "prueba --no-vad"
   bajo whispercpp (:212).
5. `core/formats.py`: bloque `engine` en write_json (:115-142).
6. Tests: fixture `-ojf` real grabada en la 980 ANTES del parser; test de contrato dual (2.7);
   stub de binario para subprocess; sha256 con archivos chicos; 2 asserts nuevos en
   test_chunk_path_cambia_con_cada_parametro (test_chunked.py:115-123); test de que con
   whispercpp+cuda jamás corren dos subprocesos a la vez; smoke GPU solo-local con skip por
   capacidad medida no-circular (patrón test_private_artifacts_windows.py:930-948: la guarda mide
   el entorno por su cuenta, jamás llama al código bajo test).
7. Checklist pre-merge: las tres mediciones H6 (una tarde en la 980): (a) JIT frío de large-v3
   q5_0 con ComputeCache vaciado — congela el piso del timeout y verifica el cacheo por módulo
   PTX + la evicción (cache default 1 GB vs fatbin ggml-cuda de 564 MB); (b) 2×whisper-cli
   concurrentes — documenta el modo de fallo real (paging vs abort) y fija el texto del mensaje;
   (c) pyannote ‖ whisper-cli — ¿el busy-spin de ggml [citado] roba CPU a torch? Informa GPU F2,
   no bloquea v1.

Criterios de aceptación medibles:

- Suite completa verde SIN tocar los 7 monkeypatch de `chunked.WhisperModel` ni test_cli.py
  (default intacto byte a byte).
- Mismo audio y flags con motores distintos → digests de `chunk_path` distintos; un checkpoint
  previo de faster-whisper jamás es hit para whispercpp (test).
- `--vad` vs `--no-vad` bajo whispercpp → MISMO digest (opts efectivos, test).
- `--hotwords --engine whispercpp` → exit≠0 con el mensaje, sin construir modelo ni tocar caché.
- JSON de ambos motores pasa el test de contrato: bloque `engine` presente en ambos;
  `language_probability` presente en faster-whisper y AUSENTE (no null, no 0.0) en whispercpp.
- Con jobs=4 y whispercpp+cuda: exactamente 1 subprocess vivo a la vez (test con stub).
- Smoke local: el tramo de 18 min con `--engine whispercpp` termina en el orden de lo medido
  (56 s caliente + JIT) [medido como referencia] y el comando registrado contiene `-mc 0`.
- H6 ejecutado y anotado antes del merge.

### PR-2 — Calidad Fase 2 residual: consumidores de payload (coste S, ~4-6 h; viernes 31-jul)

Cambios: captura de señales nativas como 3 floats opcionales en TimedSegment +
seg_to_dict/from_dict con `.get` (tolera checkpoints viejos); el puente que hoy las mata —
LabeledSegment (core/segments.py), app.py:240-243, diarization.py:37,51,56,77 — según el
inventario de seis sitios ya escrito en el plan de calidad; señales en write_json; refinamiento
de is_suspect (formats.py:33-39, "se enciende sola cuando llegue la Fase 2"); sugerencia
`--speakers N` ante conteo absurdo.

Criterios de aceptación: cero cambios a `chunk_path` (el digest de un checkpoint existente no
cambia — test); resume con checkpoints pre-PR-2 sigue verde; JSON de faster-whisper muestra las
señales, JSON de whispercpp las omite (extensión del test de contrato dual). Ventana de riesgo
miércoles-viernes (checkpoints con señales que nadie muestra): cosmética, las claves son
opcionales.

### GPU F2 — PARQUEADO con gatillo escrito (coste XS, hoy)

No se escribe el batcheado CPU por-instancia ni el solape whisper‖pyannote. El argumento
corregido (no usar velocidad bruta: 19.3x es del tramo 95% silencio y 15.6x del clip denso —
audios distintos): large-v3 GPU empata al techo CPU en calidad y aplasta al motor actual
[medido], y con whisper en la GPU pyannote tiene las 24 lógicas para él solo. Ojo: el solape
whisper‖pyannote NO existe hoy en el código — la diarización es secuencial (app.py:235-236); el
beneficio v1 real es el CPU libre DURANTE la transcripción, no venderlo como solape logrado.

La nota del parking va a `docs/` del repo o a la memoria del proyecto — NO a scratchpad, que se
purga (corrección a GPU-F2-PARK). Gatillo: un caso real en dos semanas de uso donde
`--engine faster-whisper` sea la elección para audio >30 min (VRAM ocupada, máquina sin 980,
driver/PTX muerto).

### Puente HOY (sin PR, ~10 min)

1. Rescatar del Temp (se purga): copiar `scratchpad\wcpp\cublas\Release\` completo, el ZIP
   `whisper-cublas-12.4.0-bin-x64.zip` (677.887.125 bytes = el asset pinneado [medido] — con el
   zip local, PR-1 adopta la instalación por su propia ruta fail-closed sin re-descargar ni
   debilitar la verificación) y `ggml-large-v3-q5_0.bin` a
   `%LOCALAPPDATA%\speechtotext\whisper-cpp\v1.9.1\` (layout `v1.9.1\Release\`).
2. Transcribir ya: `whisper-cli.exe -m ggml-large-v3-q5_0.bin -f audio.wav -l es -bs 5 -mc 0
   -osrt -otxt -of salida` (audio no-wav: `ffmpeg -i entrada.m4a -ar 16000 -ac 1 audio.wav`
   antes). Primera corrida paga JIT PTX (~+15 s [medido en small; large-v3 lo mide H6]).
   Es un puente de 48 h: sin troceado, sin caché, sin diarización — el miércoles PR-1 lo
   reemplaza.

---

## 4. El contrato de degradación

Regla madre ("la salida no miente", extendida: la entrada no promete lo que el motor no da):
degradar con aviso cuando el resultado sigue siendo lo pedido con menos precisión; rechazar
cuando el knob sería inerte o semánticamente distinto; omitir la clave JSON cuando la señal no
existe; jamás silencio, jamás rellenar, jamás sustituir.

| Flag / knob | faster-whisper (default) | whisper.cpp |
|---|---|---|
| `--language`, `--beam-size` | honrado | honrado (`-l`, `-bs`) |
| condition_on_previous_text=False | honrado (opts) | honrado (`-mc 0` hardcodeado, no configurable) |
| `--hotwords` | honrado (sesgo por ventana) | **RECHAZO**: exit≠0 antes de construir nada — "--hotwords no tiene efecto con whisper.cpp; usa --engine faster-whisper". Motivo [medido]: --prompt inerte bajo -mc 0; sin -mc 0 contamina ortografía. Jamás entra a la llave (por construcción) |
| `--vad` (default ON) | honrado (silero) | **DEGRADACIÓN con aviso**: "whisper.cpp no trae VAD; se transcribe sin filtro". No se puede rechazar: es default. Efectivo=False entra a la llave; el consejo "prueba --no-vad" (app.py:212) se suprime |
| `--no-vad` | honrado | honrado trivialmente; MISMO digest que `--vad` (opts efectivos) |
| `--diarize` (word_timestamps) | honrado (atribución por palabra) | **DEGRADACIÓN con aviso**: "Motor sin timestamps de palabra: atribución por segmento (gruesa), sin cortes intra-segmento; la marca [?] queda activa". JSON: `engine.diarization='segment'`. Efectivo=False en la llave |
| `compute_type` | honrado (int8/auto) | `auto`→q5_0 (mapeo pinneado, entra como efectivo a llave y JSON); valor fuera de la tabla → **RECHAZO** con la medición (fp16 = 0.53x [medido]) |
| `--jobs N` (troceado, cuda) | honrado (pool N) | **CLAMP a 1 con aviso** ("la GPU no paraleliza; jobs=1") |
| `language_probability` | clave presente en JSON | **OMISIÓN de la clave** (None → formats.py:121-123 omite; jamás 0.0) |
| señales nativas (no_speech, etc., Fase 2) | claves presentes (desde PR-2) | **OMISIÓN de claves**; el checkpoint simplemente no las escribe |
| `info.duration` | del motor (ruta directa) | fabricada por el orquestador vía probe_duration — ley de orquestación, no mentira: es una medida del audio |
| Identidad del motor | bloque `engine` en JSON + header + resumen | ídem, siempre — dos JSON del mismo audio con texto distinto son distinguibles |
| Fallo del motor | RuntimeError del backend (handler 'alloc' con consejo CPU) | RuntimeError con rc + cola de stderr + motor y trozo; consejo OOM consciente del engine (VRAM, no RAM). Jamás CalledProcessError crudo |
| Binario/modelo ausente o sha que no cuadra | n/a (descarga HF de siempre) | **ERROR con causa**; jamás fallback silencioso a otro motor |

---

## 5. Lo que NO se construye en v1

- **Protocol AsrBackend / tipos de asr/ en el CLI batch**: TranscriptionRequest no expresa
  vad_filter (extenderlo rompe fingerprints de calibración), AudioClip exige factory-token +
  provenance, TranscriptionResult obliga a fabricar campos que whisper-cli no da — ~400-600
  líneas de acople entre la ontología de dictado y la de batch. Mentir con tipos es peor que
  mentir con dicts.
- **EngineSpec / registro de motores**: un registro de dos entradas es una lista disfrazada de
  arquitectura; el dict CAPS junto al factory cubre lo mismo.
- **Auto-selección de motor**: sin datos de uso toda heurística es inventada; la VRAM libre
  fluctúa 1.4–2.7 GB [medido]. Los invariantes de auditoría (2.5) sí son ley desde ya.
- **Re-ensamblado de words desde t_dtw**: BPE parte palabras por dentro, alineación
  históricamente peor [citado]; el fallback grueso ya está previsto (diarization.py:36-37) y la
  marca [?] revive como compensación — trade declarado en el aviso de G4.
- **Tipo Segmento/Resultado universal**: el contrato duck-typed medido YA es el contrato;
  tiparlo = empobrecer o fabricar.
- **Mapear --hotwords a --prompt**: inerte bajo -mc 0 [medido]; sin -mc 0 contamina [medido].
- **fp16 en la 980**: 0.53x tiempo real por paging WDDM, 25x por batch [medido].
- **q8_0 de large-v3**: no existe publicado [medido]; cuantizar local solo si word_error lo
  justifica algún día (pregunta abierta 5).
- **VerifiedModelArtifact / manifest JSON para ggml o binario**: la cadena fortress exige
  DACL+lease incompatibles con el cache HF; la tabla pinneada en código es el manifest.
- **Auto-update del binario / port de update-fingerprints**: subir el pin es un commit consciente
  con re-benchmark, no un comando.
- **Binding pywhispercpp**: exige compilar con CUDA en Windows; el subprocess al prebuilt hace lo
  mismo con cero toolchain.
- **Semáforo/cola inter-proceso para la GPU**: jobs=1 lo logra con cero estado compartido.
- **Retry/backoff del subprocess**: reintentar contra una GPU paginando reproduce el fallo más
  lento; fallo limpio con stderr es la respuesta.
- **Watchdog/pre-check de VRAM con nvidia-smi**: el patrón del repo es reaccionar al fallo con
  mensaje; el baseline oscilante daría falsos negativos.
- **Modo daemon de whisper.cpp**: la recarga por trozo cuesta <4% [medido]; un daemon suma
  gestión de vida, puerto y limpieza por un overhead que no existe.
- **Timeout configurable / pytest-timeout en CI**: la constante proporcional cubre 17x el caso
  caliente [medido]; el CI no tiene GPU, el JIT jamás corre ahí.
- **--strict-capabilities / política de degradación configurable**: una política por capacidad,
  decidida por medición; dos políticas configurables es superficie duplicada.
- **Audit-log aparte para selección**: consola + bloque engine del JSON son el registro.
- **--engine-binary de usuario**: post-ship; cuando exista, su sha256 entra a llave y JSON (spec
  ya escrita en 2.6).
- **VAD de whisper.cpp** (modelo aparte): post-ship; en v1 la degradación con aviso cubre.
- **Batcheado CPU y solape pyannote**: parqueados con gatillo escrito en docs/ (sección 3).
- **Multimotor en `find`**: build_index (core/finder.py:91-99) es un índice de barrido con tiny
  CPU hardcodeado; ni motor nuevo ni llave nueva. No tocar.
- **Migración de caché**: los 2 checkpoints huérfanos se recomputan solos; no hay nada que migrar.

---

## 6. Cuestiones abiertas (solo las que un dato cierra)

1. **JIT frío de large-v3 q5_0 en la 980** — dato: cronometrar con el ComputeCache de CUDA
   vaciado (H6-a). Cierra: el piso del timeout (¿120 s alcanza?) y si el cacheo es por módulo
   PTX (calentar con small calienta large-v3) sin evicción del cache default de 1 GB frente al
   fatbin de 564 MB. Antes de congelar la constante de H3.
2. **Modo de fallo real de 2×whisper-cli concurrentes** — dato: dos procesos sobre la ventana de
   60 s (H6-b). Cierra: el texto exacto del mensaje de error (paging silencioso vs abort CUDA)
   del clamp y del consejo OOM.
3. **¿El busy-spin CPU de ggml (~4 hilos [citado]) le roba a pyannote/torch?** — dato: pyannote ‖
   whisper-cli en dos consolas (H6-c). Cierra: si GPU F2 necesitará afinidad/set_num_threads
   (hoy nadie fija ninguno, grep verificado). No bloquea v1.
4. **¿Revive el batcheado CPU?** — dato: un caso real en dos semanas de uso donde
   `--engine faster-whisper` sea la elección para audio >30 min. Cierra: el parking de GPU F2.
5. **¿q8_0 local paga?** — dato: word_error de un q8_0 cuantizado con whisper-quantize.exe (viene
   en el zip [medido]) vs q5_0 sobre la ventana curada. Solo se corre si el uso muestra errores
   atribuibles a cuantización; produce un artefacto sin sha upstream que pinnear (se pinnearía el
   nuestro).
