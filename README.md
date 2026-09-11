# speechtotext

Librería y CLI de voz a texto **100% local**: transcripción con
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), calidad de audio,
diarización e identificación de hablantes, y evaluación. Sin APIs externas, sin
coste por uso.

## Alcance y gobernanza

- Lo **genérico** (audio, ASR, hablantes, evaluación) entra aquí; lo **específico
  de una app** se queda en la app consumidora.
- Los consumidores fijan la dependencia a un **tag o SHA**
  (`speechtotext @ git+https://github.com/sssamuelll/speechtotext@v0.4.0`),
  nunca a `@main` flotante. Cambio que un consumidor necesite → PR + merge + tag
  aquí primero, luego bump del pin allá.
- El servicio HTTP de evaluación de pronunciación (FastAPI + Azure) vivió aquí
  hasta la `0.3.x`; hoy vive adaptado dentro de su único consumidor (klara).

---

## Requisitos

- Python ≥ 3.10
- [`ffmpeg`](https://ffmpeg.org/) en el `PATH`
  - Linux/macOS: `apt install ffmpeg` / `brew install ffmpeg`
  - Windows: descarga desde el sitio oficial y añade `ffmpeg.exe` al PATH

## Instalación

```bash
# Solo CLI offline (faster-whisper + typer)
pip install -e .

# CLI + diarización e identificación de hablantes (pyannote + torch, ~2 GB)
pip install -e ".[diarize]"
```

---

## CLI: transcripción offline

```bash
speechtotext transcribe audio.wav
speechtotext transcribe charla.mp3 --model medium --language auto --formats txt,srt
speechtotext transcribe entrevista.m4a -o transcripciones/ --device cuda
```

> Nota: ahora es un CLI de subcomandos (`transcribe`, `enroll`, `voices`, `forget`).
> La transcripción va bajo `speechtotext transcribe`.

### Opciones principales

| Flag | Default | Descripción |
|---|---|---|
| `--language`, `-l` | `es` | Código ISO-639-1 (`es`, `en`, `de`, `fr`, …) o `auto` para detectar. |
| `--model`, `-m` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`. |
| `--formats`, `-f` | `txt,srt,json` | Cualquier combinación de `txt`, `srt`, `vtt`, `json`. |
| `--device`, `-d` | `cpu` | `cpu`, `cuda`, `auto`. |
| `--compute-type` | `auto` | `auto` elige `int8` en CPU y `float16` en GPU. |
| `--vad / --no-vad` | `--vad` | Filtro de silencios largos. |
| `--beam-size` | `5` | Tamaño del beam search. |
| `--output`, `-o` | junto al audio | Carpeta o ruta base de salida. |
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
exige ese modelo justamente por eso.

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

El primer `find` sobre un archivo construye el índice (lento, una vez); las búsquedas
siguientes sobre ese mismo archivo son instantáneas. El índice se guarda en
`~/.speechtotext/index/`; `--rebuild` lo fuerza. La coincidencia ignora acentos y mayúsculas.

---

## Estructura del paquete

```
src/speechtotext/
├── core/                 lógica compartida
│   ├── audio.py          transcode_to_wav() + errores tipados
│   ├── formats.py        format_timestamp + writers (txt/srt/vtt/json)
│   └── segments.py       LabeledSegment (segmento con hablante)
├── speakers/             diarización e identificación (extra [diarize])
│   ├── diarization.py    pyannote + asignación por solape + embed_voice
│   ├── identify.py       coseno + assign_names (nombre por voz)
│   └── registry.py       registro de voces (enroll/list/get/remove)
├── cli/
│   └── app.py            typer: transcribe / find / enroll / voices / forget
├── audio/                captura, calidad y gate pre-inferencia
├── asr/                  backends de transcripción (faster-whisper)
├── models/               manifiestos y verificación de modelos
├── confidence/           features y calibración de confianza
├── evaluation/           corpus, splits, métricas y runner de evaluación
└── security/             almacenamiento de artefactos privados
```

## Desarrollo

### Añadir un formato de salida nuevo al CLI

1. Añadir `write_xxx(segments, path)` en `src/speechtotext/core/formats.py`.
2. Añadir `"xxx"` a `VALID_FORMATS` y a la tabla `writers` en `src/speechtotext/cli/app.py`.

---

## Licencia

MIT.
