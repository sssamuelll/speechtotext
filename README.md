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

### Qué modelo

`large-v3` en int8 corre en CPU a ~1,3× tiempo real con unos 3,5 GB de RAM y fue
el único que no perdió nada en [lo medido](#lo-medido); `small` va cinco veces
más rápido y cambia lo que se dijo. `tiny` y `base` son para probar el
pipeline, no para leer el resultado. En tu máquina, [`bench`](#elegir-configuración-bench).

### Hotwords

```bash
speechtotext transcribe clase.mp3 --hotwords "pyannote,diarización"
speechtotext transcribe clase.mp3 --hotwords-file terminos.txt
```

Sesgan el decodificador hacia términos que el modelo no conoce bien: nombres
propios, jerga, siglas. **Medidos, hicieron daño**: con listas de 4–5 términos,
tres apagones de 28–30 s en dos grabaciones distintas —una ventana entera
sustituida por una palabra— y ninguna mejora en el término que se quería
arreglar (ver [lo medido](#lo-medido)). Si los usas, compara contra una corrida
sin ellos. El CLI avisa a partir de 10 términos o 300 caracteres, y
`faster-whisper` trunca en silencio alrededor de los 223 tokens.

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

Trocear tiene un precio [medido](#lo-medido): en la costura no se pierde nada,
pero cada trozo después del primero decodifica con las ventanas de 30 s corridas
y deriva un 2–3 % respecto al pase único. Con VAD, para que el trozo no termine
en silencio y Whisper no invente una despedida sobre el relleno.

---

## Lo medido

Catorce minutos de una reunión real: dos voces, español de España y de
Venezuela, un micrófono, jerga técnica. Nueve configuraciones sobre el mismo
audio. La métrica no es WER: son **25 puntos concretos** que había que poder
redactar sin volver a la grabación, cada uno con su ventana de tiempo y las
palabras sin las cuales no se entiende.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/benchmark-dark.svg">
  <img alt="Velocidad contra errores por mil palabras de seis configuraciones: los large-v3 abajo, los small arriba" src="docs/img/benchmark-light.svg" width="760">
</picture>

| Configuración | Puntos | Errores / 1000 pal. | Reloj |
|---|---:|---:|---:|
| faster-whisper `large-v3`, sin VAD, sin trocear | **25 / 25** | **1,6** | 683 s |
| faster-whisper `large-v3` + VAD | 25 / 25 | 7,7 | 662 s |
| whisper.cpp `large-v3` CUDA | **25 / 25** | 6,0 | **109 s** |
| faster-whisper `large-v3` + hotwords | 24 / 25 | 3,8 | 667 s |
| faster-whisper `large-v3` troceado, con o sin VAD | 24 / 25 | — | — |
| faster-whisper `large-v3` troceado + VAD + hotwords | 23 / 25 | — | ≈480 s |
| faster-whisper `small` | 22 / 25 | 29,1 | 135 s |
| whisper.cpp `small` CUDA | 21 / 25 | 27,5 | 55 s |

Los errores se cuentan sobre los 66 sitios en que las transcripciones discrepan
y la respuesta es objetiva —un disparate que no es español, un término, un
número, una omisión confirmada—; los otros 100 sitios en desacuerdo (*esta/esto*,
muletillas, comas) quedan fuera a propósito. Las configuraciones troceadas se
compararon por pares contra su gemela sin trocear, no en esa alineación. Reloj
de un Ryzen 9 5900X con una GTX 980, corridas en serie, carga del modelo incluida.

Lo que se aprendió:

- **`small` cambia lo que se dijo.** Diecisiete veces más errores que `large-v3`,
  y no son erratas: *"todo se desordena"* salió como *"entonces ordenas"*. Ahorra
  nueve minutos y cuesta tres o cuatro de los 25 puntos.
- **Los hotwords fallan en bloque.** Con 4–5 términos, tres apagones de 28–30 s
  en dos grabaciones —una ventana entera sustituida por *"listo"*— y ninguna
  mejora en el término que se quería arreglar. n = 3, sin contraejemplo.
- **El VAD borra frases cortas sin avisar.** Cuatro omisiones confirmadas y cero
  huecos declarados: lo que descarta antes de que el modelo lo vea no deja
  agujero en la línea de tiempo.
- **Trocear no pierde nada en la costura.** Cuesta un 2–3 % de deriva en cada
  trozo después del primero, porque las ventanas de 30 s quedan corridas; ahí
  cayó un punto de 25.
- **whisper.cpp empata en puntos y pierde en jerga.** Seis veces más rápido en la
  GTX 980 con 2 GB de VRAM, las mismas 25/25, y *Bézier* mal escrito las diez
  veces — medido en un solo audio.

Lo que no prueba: una grabación, un dominio, una máquina. La referencia la
adjudicó quien hizo el benchmark, no un transcriptor humano, y las omisiones se
confirmaron con otra configuración de Whisper — un error que todo Whisper
comparta (*"clico la fecha"* por *flecha*, en las nueve) este método no lo ve.
La grabación es privada y no se publica; el gráfico se regenera con
`python scripts/benchmark_chart.py`.

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
