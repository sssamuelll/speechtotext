# speechtotext

Librería y CLI de voz a texto **100% local**: transcripción con
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), calidad de audio,
diarización e identificación de hablantes, y evaluación. Sin APIs externas, sin
coste por uso.

## Alcance y gobernanza

- Lo **genérico** (audio, ASR, hablantes, evaluación) entra aquí; lo **específico
  de una app** se queda en la app consumidora.
- Los consumidores fijan la dependencia a un **tag o SHA**
  (`speechtotext @ git+https://github.com/sssamuelll/speechtotext@v0.5.0`),
  nunca a `@main` flotante. Cambio que un consumidor necesite → PR + merge + tag
  aquí primero, luego bump del pin allá.
- El contrato que ven los consumidores está en **[`docs/api.md`](docs/api.md)**:
  esquema del JSON, tipos públicos y qué garantiza cada módulo. Los cambios de
  ese contrato se anotan en [`CHANGELOG.md`](CHANGELOG.md).
- El servicio HTTP de evaluación de pronunciación (FastAPI + Azure) vivió aquí
  hasta la `0.3.x`; hoy vive adaptado dentro de su único consumidor (klara).

---

## Requisitos

- Python ≥ 3.11
- [`ffmpeg`](https://ffmpeg.org/) en el `PATH`
  - Linux/macOS: `apt install ffmpeg` / `brew install ffmpeg`
  - Windows: descarga desde el sitio oficial y añade `ffmpeg.exe` al PATH

> Transcripción, diarización y búsqueda corren en Linux, macOS y Windows. Los
> subsistemas de integridad (`models/`, `security/`) y el arnés de evaluación con
> corpus privado son **solo Windows**: dependen de handles, ACLs y cifrado NTFS.

## Instalación

```bash
# Solo CLI offline (faster-whisper + typer)
pip install -e .

# CLI + diarización e identificación de hablantes (pyannote + torch, ~2 GB)
pip install -e ".[diarize]"

# Arnés de evaluación y calibración (scikit-learn + scipy)
pip install -e ".[evaluation]"

# Suite de tests
pip install -e ".[dev]"
```

---

## Los subcomandos

| Comando | Para qué |
|---|---|
| `transcribe` | Transcribir un audio a `txt`/`srt`/`vtt`/`json`, con hablantes si lo pides. |
| `find` | Ubicar un tema dentro de un audio largo sin transcribirlo entero. |
| `enroll` | Registrar la voz de una persona para que se le ponga nombre. |
| `voices` | Listar las voces registradas. |
| `forget` | Borrar una voz del registro. |
| `bench` | Medir en tu máquina qué configuración conviene. |

---

## Transcripción offline

```bash
speechtotext transcribe audio.wav
speechtotext transcribe charla.mp3 --model medium --language auto --formats txt,srt
speechtotext transcribe entrevista.m4a -o transcripciones/ --device cuda
```

### Opciones

| Flag | Default | Descripción |
|---|---|---|
| `--language`, `-l` | `es` | Código ISO-639-1 (`es`, `en`, `de`, `fr`, …) o `auto` para detectar. |
| `--model`, `-m` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`. |
| `--formats`, `-f` | `txt,srt,json` | Cualquier combinación de `txt`, `srt`, `vtt`, `json`. |
| `--device`, `-d` | `cpu` | `cpu`, `cuda`, `auto`. |
| `--compute-type` | `auto` | `auto` elige `int8` en CPU y `float16` en GPU. |
| `--vad / --no-vad` | `--vad` | Filtro de silencios largos. |
| `--beam-size` | `5` | Tamaño del beam search (mínimo 1). |
| `--output`, `-o` | junto al audio | Carpeta o ruta base de salida. |
| `--engine` | `faster-whisper` | `faster-whisper` o `whispercpp` (ver [Motores](#motores)). |
| `--hotwords` | — | Términos que el modelo debe preferir, separados por coma. |
| `--hotwords-file` | — | Archivo con esos términos, uno por línea. |
| `--chunk / --no-chunk` | auto | Trocear el audio; automático por encima de 20 minutos. |
| `--jobs` | `4` | Trozos transcritos en paralelo. |
| `--diarize`, `-D` | off | Marcar quién habla (requiere el extra `[diarize]`). |
| `--speakers` | auto | Número de hablantes como pista (p.ej. `2`); auto si se omite. |
| `--identify / --no-identify` | `--identify` | Poner nombre a las voces registradas con `enroll`. |
| `--threshold` | `0.5` | Umbral de coincidencia de voz (coseno, 0–1). |

### Guía rápida de modelos

| Modelo | RAM/VRAM | Velocidad CPU | Calidad |
|---|---|---|---|
| `tiny` | ~1 GB | muy rápida | baja, solo pruebas |
| `small` | ~2 GB | buena | sweet spot CPU |
| `medium` | ~5 GB | lenta en CPU | muy buena |
| `large-v3` | ~10 GB | muy lenta en CPU | máxima |

### Hotwords

```bash
speechtotext transcribe reunion.m4a --hotwords "Aurelius,pyannote,diarización"
speechtotext transcribe clase.mp3 --hotwords-file terminos.txt
```

Sesgan el decodificador hacia términos que el modelo no conoce bien: nombres
propios, jerga, siglas. El efecto se diluye con la cantidad — a partir de 10
términos o 300 caracteres el CLI avisa, y `faster-whisper` trunca en silencio
alrededor de los 223 tokens. Una lista corta y específica funciona mejor que un
glosario entero.

### Audio largo

Por encima de 20 minutos, `transcribe` trocea el audio solo. Corta en silencios
(nunca a mitad de palabra), transcribe los trozos en paralelo según `--jobs` y
guarda cada uno en `~/.speechtotext/chunks`. Si la corrida se interrumpe, la
siguiente reanuda desde el último trozo terminado.

El checkpoint es por contenido: la llave incluye el archivo, el modelo, el motor,
la cuantización, el device y los flags de decodificación. Cambiar cualquiera de
esos invalida el caché en vez de reusar un resultado que no corresponde.

```bash
speechtotext transcribe podcast_3h.mp3 --jobs 6      # más paralelismo
speechtotext transcribe entrevista.wav --no-chunk    # forzar un solo pase
```

---

## Motores

`--engine faster-whisper` (el default) es el camino normal: CPU o CUDA, todos los
flags honrados. `--engine whispercpp` existe para GPUs viejas donde CTranslate2 ya
no rinde — usa un binario de whisper.cpp pinneado por SHA-256, que se descarga y
verifica la primera vez y se cachea.

La selección es siempre explícita: **ningún motor se elige solo**. Lo que sí
cambia solo son los flags que el motor no puede honrar:

| Flag | Bajo `whispercpp` |
|---|---|
| `--device` | Forzado a `cuda` (el binario pinneado es build CUDA), con aviso. |
| `--compute-type` | `auto` resuelve a `q5_0`; pedir `float16` explícito es un error. |
| `--vad` | Se apaga, con aviso. |
| `--jobs` | Se fuerza a 1. |
| `--hotwords` | **Rechazado**: la corrida no arranca. |

`--hotwords` corta en vez de degradar a propósito: el knob se midió inerte bajo
ese motor, y fingir que se aplicó sería peor que decirlo.

---

## Diarización e identificación de hablantes

Con el extra `[diarize]`, `speechtotext` marca **quién dijo qué** en una grabación de
conversación y, si registras las voces, les pone **nombre**. Todo local.

### Requisitos (una sola vez)

Usa modelos de [pyannote](https://github.com/pyannote/pyannote-audio) que se descargan
de Hugging Face y están _gated_:

1. Crea un token **Read** en https://huggingface.co/settings/tokens y expórtalo:
   ```bash
   export HF_TOKEN=hf_tu_token        # Windows: setx HF_TOKEN "hf_tu_token"
   ```
2. Logueado en HF, acepta el acceso al modelo en
   https://huggingface.co/pyannote/speaker-diarization-community-1
   (si en el primer uso pyannote pide aceptar algún modelo dependiente, acéptalo también).

La primera corrida descarga los modelos a `~/.cache/huggingface`; luego quedan en caché.

### Registrar voces (enrollment)

```bash
speechtotext enroll "Samuel" muestra_samuel.wav   # >=10 s de una sola voz, limpia
speechtotext voices                               # lista las voces registradas
speechtotext forget "Samuel"                      # borra una voz
```

Las voces se guardan en `~/.speechtotext/` (override con `SPEECHTOTEXT_HOME`).

Cada voz queda archivada bajo el modelo que produjo su embedding, y solo se compara
contra vectores del mismo modelo: el coseno entre dos espacios vectoriales distintos no
significa nada. Si consumes el registro como librería, `registry.get_embeddings(model)`
exige ese modelo justamente por eso — y devuelve un diccionario vacío, no un error, si
ese modelo no tiene voces enroladas. El formato en disco está en
[`docs/api.md`](docs/api.md#registro-de-voces).

### Transcribir con hablantes

```bash
# anónimo: Hablante 1 / Hablante 2
speechtotext transcribe conversacion.mp3 --diarize

# pista de 2 hablantes (mejora precisión) + nombres de las voces registradas
speechtotext transcribe conversacion.mp3 --diarize --speakers 2

# más estricto al poner nombres
speechtotext transcribe llamada.m4a -D --threshold 0.6
```

Salida `txt` de ejemplo:

```
Samuel: Hola, ¿cómo estás?
Hablante 2: Bien, ¿y tú?
```

En `json` cada segmento gana un campo `"speaker"` y hay un top-level `"speakers"`; en
`srt`/`vtt` el hablante prefija cada línea. Sin `--diarize`, la salida es idéntica a la
de siempre.

### Límites

- Etiqueta a **nivel de segmento**, no de palabra: un cambio de turno a mitad de
  segmento se lo lleva un solo hablante.
- La identificación depende de la calidad del enrollment y del `--threshold`; voces muy
  parecidas pueden confundirse.
- En CPU funciona, pero la diarización suma tiempo sobre la transcripción.
- Hoy solo pyannote produce embeddings, y corre el pipeline completo de
  diarización para hacerlo: son segundos por muestra, no milisegundos. Sirve para
  lote, no para un camino en vivo.

---

## Buscar un segmento

Transcribir un audio largo en calidad tarda mucho. Si solo te interesa un tramo (una
entrevista, una ponencia), `find` lo ubica sin transcribir todo: hace un pase rápido con
`tiny`, busca tu consulta y devuelve las **regiones** donde aparece. Con `--extract`, además
recorta y transcribe en calidad el tramo elegido.

```bash
# ubicar: imprime las regiones (minutos) donde aparece la consulta
speechtotext find programa.mp3 "vulnerabilidad sísmica"

# extraer: recorta + transcribe en calidad la región más densa
speechtotext find programa.mp3 "vulnerabilidad sísmica" --extract

# elegir otra región, y con diarización + nombres
speechtotext find programa.mp3 "entrevista" --extract --region 2 -D --speakers 4
```

| Flag | Default | Descripción |
|---|---|---|
| `--extract` | off | Recortar y transcribir en calidad la región elegida. |
| `--region` | `1` | Cuál de las regiones encontradas extraer. |
| `--top` | `5` | Cuántas regiones listar. |
| `--context` | `10.0` | Segundos de margen alrededor del recorte. |
| `--scan-model` | `tiny` | Modelo del pase rápido de indexado. |
| `--rebuild` | off | Rehacer el índice aunque exista. |

El primer `find` sobre un archivo construye el índice (lento, una vez); las búsquedas
siguientes sobre ese mismo archivo son instantáneas. El índice se guarda en
`~/.speechtotext/index/`; `--rebuild` lo fuerza. La coincidencia ignora acentos y mayúsculas.

> `find --extract` transcribe con `cpu`, `compute-type auto`, VAD encendido y
> `beam-size 5` fijos. Para otra configuración, extrae primero y luego corre
> `transcribe` sobre el recorte.

---

## Elegir configuración: `bench`

```bash
speechtotext bench muestra.wav              # mide las configuraciones viables
speechtotext bench muestra.wav --quick      # menos configuraciones
speechtotext bench muestra.wav --seconds 90 # tramo más largo
speechtotext bench --show                   # repinta la última medición
```

Mide en **tu** máquina las combinaciones de motor, modelo y cuantización que tu
hardware aguanta, cada una en un subproceso aislado, y recomienda una por caso de
uso (rápido, equilibrado, calidad). El resultado queda en `bench.json` y `--show`
lo repinta sin volver a medir.

---

## Salida

`txt` es la transcripción plana, `srt`/`vtt` son subtítulos con tiempos, y `json`
es el formato completo: segmentos con tiempos, idioma detectado, huecos sin voz,
y las señales nativas del motor (`no_speech`, `avg_logprob`, `compression_ratio`)
que permiten decidir si un segmento es de fiar.

El esquema completo, con qué claves aparecen y cuándo, está en
**[`docs/api.md`](docs/api.md#esquema-del-json)**. Un segmento marcado
`"suspect": true` es una sugerencia de revisión, no un veredicto.

---

## Estructura del paquete

```
src/speechtotext/
├── core/                 el camino de la transcripción
│   ├── engines.py        multimotor: faster-whisper / whisper.cpp + degradaciones
│   ├── enginepin.py      descarga y verificación por SHA-256 del binario y ggml
│   ├── chunked.py        troceo por silencios, checkpoint y paralelismo
│   ├── finder.py         índice rápido y búsqueda de regiones (subcomando find)
│   ├── benchmark.py      medición de configuraciones (subcomando bench)
│   ├── formats.py        writers txt/srt/vtt/json + is_suspect + huecos
│   ├── segments.py       LabeledSegment y lectura de señales nativas
│   ├── audio.py          transcode_to_wav() + errores tipados
│   └── postprocess.py    normalización de horas en el texto
├── speakers/             diarización e identificación (extra [diarize])
│   ├── diarization.py    pyannote + asignación por solape + embed_voice
│   ├── identify.py       coseno + assign_names (nombre por voz)
│   └── registry.py       registro de voces, archivado por modelo
├── audio/                medidas sobre la señal
│   ├── evidence.py       evidencia de voz por DSP determinista
│   ├── quality.py        RMS, SNR, clipping, piso de ruido
│   ├── gate.py           elegibilidad pre-inferencia contra umbrales
│   ├── io.py             decodificación a mono float32
│   ├── level.py          ganancia fija con limitador
│   ├── fingerprint.py    huella criptográfica de un pipeline de audio
│   └── types.py          modelos de dominio inmutables
├── asr/                  contratos provider-neutral de transcripción
├── models/               manifiestos y verificación de modelos (Windows)
├── confidence/           features y calibración de confianza
├── evaluation/           corpus, splits, métricas y runner de evaluación
├── security/             artefactos privados de runtime (Windows)
└── cli/app.py            typer: transcribe / find / enroll / voices / forget / bench
```

---

## Documentación

| Documento | Qué contiene |
|---|---|
| [`docs/api.md`](docs/api.md) | Contrato para consumidores: JSON, tipos públicos, garantías. |
| [`docs/audio-evaluation.md`](docs/audio-evaluation.md) | Runbook del corpus privado, evaluación y calibración. |
| [`docs/README.md`](docs/README.md) | Índice de `docs/`, con qué está vigente y qué es histórico. |
| [`CHANGELOG.md`](CHANGELOG.md) | Qué cambió entre tags, y qué rompe. |

---

## Desarrollo

```bash
pip install -e ".[dev,evaluation]"
pytest -q
```

El CI de GitHub Actions está deshabilitado: los gates son locales.

### Añadir un formato de salida nuevo al CLI

1. Añadir `write_xxx(segments, path)` en `src/speechtotext/core/formats.py`.
2. Añadir `"xxx"` a `VALID_FORMATS` y a la tabla `writers` en `src/speechtotext/cli/app.py`.

---

## Licencia

MIT.
