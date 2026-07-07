# speechtotext

Toolkit de voz a texto con dos superficies independientes:

- **CLI offline** (`speechtotext`) — transcripción local con [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper). Sin APIs externas, sin coste por uso.
- **Servicio HTTP** (`speechtotext.api`) — endpoint FastAPI de evaluación de pronunciación contra un texto de referencia, respaldado por [Azure AI Speech Pronunciation Assessment](https://learn.microsoft.com/azure/ai-services/speech-service/how-to-pronunciation-assessment).

Ambas comparten utilidades en `speechtotext.core` (transcoding ffmpeg, serializadores de subtítulos).

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

# CLI + servicio HTTP de pronunciación (FastAPI + Azure SDK)
pip install -e ".[api]"
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

## API: evaluación de pronunciación

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `AZURE_SPEECH_KEY` | _(requerida)_ | Clave del recurso Azure Speech. |
| `AZURE_SPEECH_REGION` | `westeurope` | Región del recurso (debe coincidir con la del portal). |
| `CORS_ORIGINS` | `*` | Orígenes permitidos separados por coma. **Restringir en producción.** |

### Arrancar en local

```bash
export AZURE_SPEECH_KEY=tu_clave_de_azure
export AZURE_SPEECH_REGION=westeurope
uvicorn speechtotext.api.app:app --reload --port 8000
```

Documentación interactiva en `http://localhost:8000/docs` (OpenAPI).

### `POST /score`

`multipart/form-data`:

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `audio` | file | — | Audio del usuario en cualquier formato que ffmpeg sepa decodificar (webm/ogg/wav/mp3/m4a). |
| `reference_text` | string | — | Texto que el usuario debía pronunciar. |
| `language` | string | `de-DE` | Código BCP-47 (p.ej. `de-DE`, `en-US`, `es-ES`). |

Ejemplo con `curl`:

```bash
curl -X POST http://localhost:8000/score \
  -F audio=@user.webm \
  -F reference_text="Ich hätte gern einen Kaffee, bitte." \
  -F language=de-DE
```

Ejemplo desde el frontend (grabando con `MediaRecorder`):

```js
const fd = new FormData();
fd.append("audio", audioBlob, "user.webm");
fd.append("reference_text", "Ich hätte gern einen Kaffee, bitte.");
fd.append("language", "de-DE");

const res = await fetch("http://localhost:8000/score", {
  method: "POST",
  body: fd,
});
const { scores, words } = await res.json();
// scores.accuracy / fluency / completeness / pronunciation (0–100)
// words[i].phonemes[j].accuracy_score → para resaltar fonemas mal pronunciados
```

### Forma de la respuesta

```json
{
  "recognized_text": "Ich hätte gern einen Kaffee bitte",
  "reference_text": "Ich hätte gern einen Kaffee, bitte.",
  "language": "de-DE",
  "scores": {
    "accuracy": 87.0,
    "fluency": 92.0,
    "completeness": 100.0,
    "pronunciation": 89.5
  },
  "words": [
    {
      "word": "hätte",
      "accuracy_score": 78.0,
      "error_type": "None",
      "phonemes": [
        {"phoneme": "h", "accuracy_score": 95.0},
        {"phoneme": "ɛ", "accuracy_score": 62.0},
        {"phoneme": "t", "accuracy_score": 88.0},
        {"phoneme": "ə", "accuracy_score": 80.0}
      ]
    }
  ]
}
```

### Códigos de error

| Status | Significado |
|---|---|
| `400` | Audio vacío, sin `reference_text`, o ffmpeg no pudo decodificar el archivo. |
| `422` | Azure no detectó voz en el audio (recoverable: pide al usuario que repita). |
| `500` | `AZURE_SPEECH_KEY` no configurada o `ffmpeg` no instalado en el servidor. |
| `502` | Azure devolvió un error no recuperable. |

### `GET /health`

```json
{ "status": "ok", "azure_configured": true, "azure_region": "westeurope" }
```

### Sobre Azure Pronunciation Assessment

- **Idiomas:** `de-DE`, `en-US`, `en-GB`, `es-ES`, `fr-FR`, `it-IT`, `ja-JP`, `zh-CN`, … ([lista completa](https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=stt#pronunciation-assessment)).
- **Free tier (F0):** 5 horas de audio al mes — suficiente para un MVP con docenas de usuarios activos.
- **Paid tier (S0):** ~$1 USD por hora de audio. Sin compromiso mínimo.
- La función `prosody` solo está disponible en `en-US`; el resto de idiomas devuelven `accuracy`, `fluency`, `completeness` y `pronunciation`.

---

## Estructura del paquete

```
src/speechtotext/
├── core/                 lógica compartida CLI ↔ API
│   ├── audio.py          transcode_to_wav() + errores tipados
│   ├── formats.py        format_timestamp + writers (txt/srt/vtt/json)
│   └── segments.py       LabeledSegment (segmento con hablante)
├── speakers/             diarización e identificación (extra [diarize])
│   ├── diarization.py    pyannote + asignación por solape + embed_voice
│   ├── identify.py       coseno + assign_names (nombre por voz)
│   └── registry.py       registro de voces (enroll/list/get/remove)
├── cli/
│   └── app.py            typer: transcribe / enroll / voices / forget
└── api/
    ├── app.py            create_app() — FastAPI + CORS + router
    ├── config.py         Settings (env vars)
    ├── schemas.py        modelos pydantic de respuesta
    ├── azure_client.py   cliente Azure + AzureSpeechError
    └── routes/
        ├── health.py     GET /health
        └── pronunciation.py  POST /score
```

## Desarrollo

### Añadir un endpoint nuevo

1. Crear `src/speechtotext/api/routes/mi_ruta.py` con `router = APIRouter(...)` y los handlers.
2. Incluirlo en `src/speechtotext/api/routes/__init__.py`:
   ```python
   from speechtotext.api.routes.mi_ruta import router as mi_router
   api_router.include_router(mi_router)
   ```

### Añadir un formato de salida nuevo al CLI

1. Añadir `write_xxx(segments, path)` en `src/speechtotext/core/formats.py`.
2. Añadir `"xxx"` a `VALID_FORMATS` y a la tabla `writers` en `src/speechtotext/cli/app.py`.

---

## Licencia

MIT.
