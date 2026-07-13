# Esqueleto andante del asistente de voz local ("Jarvis") — diseño

Fecha: 2026-07-13
Estado: aprobado por Samuel (brainstorming + panel de arquitectura de 6 agentes + auditorías de runtime y sobre-ingeniería)

## 1. Contexto y objetivo

Samuel construye su asistente de voz personal 100% local como hito de carrera **open-source público**. speechtotext (este repo) aporta los activos de voz ya construidos; el asistente vive en un **repo nuevo público: `aurelius`** (decidido 2026-07-13, ver §14).

Estrategia: **esqueleto andante**. Primer entregable (3-3.5 semanas a tiempo parcial): un loop end-to-end vivo — oye → piensa → habla — feo pero completo y medido. Después se profundiza órgano por órgano, empezando por la razón de ser del proyecto: un **pipeline DSP profesional en vivo** (supresión de ruido, EQ de frecuencias de voz, compresor, AEC). El esqueleto no incluye el DSP pro, pero instala su fontanería completa (§6).

Descomposición del Jarvis completo: **oídos** (este spec) → **voz** (incluida mínima en este spec) → **cerebro** (mínimo aquí; profundo en fase 4) → **manos** (fase 3). Cada fase futura tendrá su propio spec.

## 2. Decisiones cerradas

| Decisión | Valor | Fuente |
|---|---|---|
| Orden de construcción | Esqueleto andante end-to-end primero | Samuel, 2026-07-13 |
| Ejecución | 100% local, sin APIs cloud, incluido el cerebro | Samuel, 2026-07-13 |
| Interacción día 1 | Wake word + gating por identificación de voz enrolada | Samuel, 2026-07-13 |
| Vara del hito | Open-source público: repo pulido, demo, licencias limpias | Samuel, 2026-07-13 |
| Enfoque arquitectónico | **B — Bus fullband** (48 kHz / frames de 10 ms desde el día 1) | Samuel, 2026-07-13, sobre recomendación del panel |
| Hardware objetivo | Ryzen 9 5900X, 32 GB RAM, **CPU-only** (GTX 980 Maxwell: sin soporte moderno) | verificado en la máquina |
| Idioma primario | Español (voz y respuestas) | contexto |

## 3. Alcance del esqueleto

**Incluye:** captura de micrófono always-on; bus de audio 48 kHz/10 ms con cadena DSP enchufable (passthrough + highpass biquad); wake word; VAD; ASR por utterance en español; gate de identidad de voz (solo la voz de Samuel activa el cerebro); LLM local con respuestas breves; TTS por oración con playback solapado; barge-in por wake word; instrumentación JSONL; README público con demo y latencias medidas.

**No incluye (explícito):** DSP pro (supresión/EQ/compresor/AEC — fase 2); acciones sobre la máquina (fase 3); memoria de largo plazo del cerebro (fase 4); GUI/overlay; daemon multi-cliente, WebSocket o API de red; soporte multi-usuario; modo always-on sin wake word; empaquetado instalable (PyInstaller etc.).

## 4. Arquitectura (Enfoque B — bus fullband)

Un solo proceso Python (threads + colas de stdlib) y dos subprocesos externos: `llama-server` (HTTP en localhost) y Piper (un subprocess por oración).

```
mic ─ sounddevice/WASAPI callback (48 kHz mono float32, blocksize 20-30 ms del device)
        │  (el callback SOLO copia al ring; jamás bloquea ni aloca)
        ▼
   ring SPSC de captura
        ▼
   thread pipeline:
     re-partición a frames de 10 ms (480 muestras)
     → cadena DSP (§6): hoy passthrough + highpass biquad
     → decimador CON ESTADO 48→16 kHz (sosfilt con zi persistente + 1 de cada 3)
     → fan-out por colas acotadas (con contador de drops)
             ├──▶ openWakeWord (vista 16 kHz)
             └──▶ Silero VAD → segmentador de utterances (vista 16 kHz)
        ▼
   thread FSM (§7): orquesta ASR → gate de identidad → LLM → splitter de oraciones
        ▼
   thread playback: cola de PCM por oración → salida de audio
     └──▶ copia cada frame reproducido al ring de referencia far-end
          (timestamps time.monotonic(); los timestamps de device precisos son trabajo de la fase AEC)
```

Reglas de higiene: `SimpleQueue.put_nowait` desde el callback (sin locks); colas acotadas en todo el fan-out, drop de frames viejos con contador visible; blocksize del device desacoplado del frame DSP (pelear por callbacks de 10 ms contra WASAPI no aporta); threads de inferencia con afinidad limitada (CTranslate2 `cpu_threads=8`, llama-server `-t 8-10`) para no matar el deadline del pipeline de audio.

## 5. Órganos (stack, todo CPU)

| Órgano | Elección | Licencia | Nota |
|---|---|---|---|
| Captura/playback | sounddevice (PortAudio, WASAPI) | MIT | probar con el hardware real el día 1 |
| VAD | Silero VAD v5 ONNX | MIT | sin torch; cierre de utterance a 300-500 ms de silencio |
| Wake word | openWakeWord, modelo `hey_jarvis` preentrenado | Apache-2.0 | prueba de humo con el acento de Samuel el día 1; plan B: entrenamiento sintético custom (~1.5 h Colab) |
| ASR | faster-whisper `small` int8, por utterance | MIT (código y pesos) | pseudo-streaming por VAD; streaming real en español no existe con calidad (verificado 2026-07-13); upgrade path a `medium` int8 (+~1 s) |
| Identidad de voz | sherpa-onnx speaker embeddings (wespeaker/3dspeaker) + `speechtotext.speakers.registry/identify` | Apache-2.0 | decenas de ms por utterance; sin token HF. **No** pyannote en el camino vivo: `embed_voice()` corre el pipeline completo (segundos). Re-enrolar una vez (el manifest ya guarda `model`). Gate de UNA etapa sobre la utterance completa, antes del LLM |
| Cerebro | Qwen2.5-7B-Instruct Q4_K_M vía `llama-server` | Apache-2.0 | `-t 8-10` explícito; `cache_prompt=true` obligatorio; system prompt fuerza respuestas de 1-3 oraciones |
| TTS | Piper, subprocess por oración (EOF = fin del audio) | GPL-3.0 (binario externo, jamás importado) | voz es_MX/es_ES/es_AR a decidir a oído (§14); RTF ~0.03-0.1 |

pyannote queda confinado al batch de speechtotext: la instalación pública del asistente funciona sin cuenta de HuggingFace.

## 6. La costura DSP (contrato de fase 2, instalado hoy)

Contrato del bus: **mono float32, 48 000 Hz, frames de 480 muestras (10 ms)** — el formato nativo de WebRTC APM/AEC3, con banda completa para EQ y compresor de voz (a 16 kHz no hay banda que trabajar: Nyquist en 8 kHz).

```python
class Stage(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray:  # in/out: float32, len == 480
        ...
    def reset(self) -> None:                              # p. ej. tras barge-in o cambio de device
        ...
```

- La cadena se instancia con `(sample_rate=48_000, frame_size=480)` y corre dentro del thread pipeline, antes de la decimación: los órganos ML siempre oyen audio ya procesado.
- Presupuesto: la cadena completa debe cerrar en < 10 ms por frame. En el esqueleto: un timer de cadena completa + contador de xruns. EMA por etapa y bypass fail-open se difieren a cuando haya > 2 etapas.
- El esqueleto incluye passthrough + un highpass biquad real (80 Hz) como prueba de que el estado entre frames sobrevive (zi persistente).
- El ring de referencia far-end (§4) queda instalado desde el día 1: la fase AEC arranca escribiendo la etapa, no la fontanería.
- Cláusula de degradación: si al cierre de la semana 1 el camino de 48 kHz no está estable (xruns fuera de criterio), se congela el bus a 16 kHz (Enfoque A) sin tocar wake/VAD/ASR/gate/LLM/TTS — el fork está contenido en el front-end de audio.

## 7. Máquina de estados

```
IDLE ──wake word (emite earcon <150 ms)──▶ AWAKE ──primera voz──▶ LISTENING
LISTENING ──silencio 300-500 ms──▶ THINKING   (ASR ∥ embedding → gate → LLM streaming)
THINKING ──primera oración del splitter──▶ SPEAKING   (Piper por oración, playback solapado)
SPEAKING ──cola de oraciones vacía──▶ hangover 300-500 ms + drenaje del buffer del device ──▶ IDLE
SPEAKING ──wake word──▶ barge-in: flush de cola de oraciones + kill del Piper en curso ──▶ AWAKE
AWAKE ──timeout sin voz (p. ej. 6 s)──▶ IDLE
```

- **Gate de identidad:** coseno del embedding de la utterance contra la voz enrolada de Samuel; bajo el threshold → se descarta la utterance y se loguea el score (modo debug explica cada rechazo). Calibración en S3: 10 utterances propias + 10 ajenas.
- **Half-duplex:** en SPEAKING se gatean VAD/ASR; solo el wake word sigue escuchando (botón de interrupción por voz). Limitación documentada del v0.1: con altavoz fuerte y sin AEC, el wake pierde SNR — best effort.
- **Hangover obligatorio** al salir de SPEAKING: sin él, el sistema transcribe su propia cola reverberante y conversa consigo mismo.
- ASR devuelve texto vacío → directo a IDLE sin tocar el LLM.

## 8. Manejo de errores

- `llama-server` caído o timeout → reinicio con backoff exponencial; mientras, respuesta enlatada vía Piper ("dame un segundo, se me cayó el cerebro"). El loop de audio nunca muere porque el cerebro murió.
- Piper falla en una oración → log + saltar a la siguiente; la FSM no se bloquea.
- Ring/colas llenas → drop de lo más viejo + contador; el callback jamás bloquea ni espera.
- Cambio/desconexión de device de audio → error visible + reintento de apertura; no crash.
- Ctrl+C → drenaje de colas, kill de subprocesos, cierre limpio del JSONL.
- Configuración: un TOML mínimo (device, thresholds, voz, ruta del GGUF, threads). Sin sistema de config elaborado.

## 9. Instrumentación y testing

**Instrumentación (desde la semana 1):** eventos JSONL a archivo con 5 fronteras por turno — fin de habla, ASR listo, primer token del LLM, primera oración al TTS, primer sample audible — más xruns, drops por cola y coseno de cada gate. El README publica latencias medidas por este log, no prometidas.

**Tests unitarios (solo lógica pura):**
- Decimador con estado: sweep sintético 48→16 comparado contra resample offline; discontinuidades en fronteras de frame = fallo.
- Re-partición blocksize→frames de 10 ms (incluye blocksizes que no dividen exacto).
- Splitter de oraciones (abreviaturas, números, signos de apertura del español).
- FSM: tabla de transiciones ejercitada con eventos sintéticos, incluidos barge-in y timeout.

**Integración sin micrófono:** la fuente de audio es intercambiable por WAVs fixture — el camino completo (bus → wake → VAD → ASR → gate) corre en CI sin hardware.

**Criterios de salida por semana:**
- S1: xruns instrumentados < N/hora con N definido tras las primeras 48 h de medición (no "cero xruns"); decimador y cadena DSP pasando sus tests.
- S2: pregunta hablada → respuesta streameada en consola; la voz de otra persona no lo despierta.
- S3: conversación multivuelta sin manos con latencias dentro del presupuesto (§10); benchmark script que genera la tabla del README.

## 10. Presupuesto de latencia (5900X, CPU, cache caliente)

| Tramo | Presupuesto |
|---|---|
| Fin de habla → cierre VAD | 0.3-0.5 s |
| Earcon "te oí" | < 150 ms tras el cierre |
| ASR (utterance 3-5 s, small int8) | 0.4-1.6 s (medir con la voz de Samuel en S2) |
| Gate de identidad | < 100 ms |
| LLM primer token (cache_prompt) | 0.3-0.8 s |
| Primera oración + Piper | 0.5-1 s |
| **Fin de habla → primera sílaba (p50)** | **3.5-5 s** |

Turno con cache frío: 10-15 s (se documenta como limitación). La física la fijan Whisper + el 7B memoria-bound en DDR4; el enmascaramiento (earcon + TTS solapado) es parte del diseño, no un parche.

## 11. Repo, licencias y distribución

- Repo nuevo público, licencia **MIT**, en GitHub de Samuel. Depende de speechtotext vía `pip install git+...` únicamente por `speakers.registry/identify` (comparte `~/.speechtotext`). Cero refactor de speechtotext, cero monorepo.
- Piper: binario externo descargado por script de setup (GPL-3.0 aislado por frontera de proceso). Modelos (GGUF, ONNX, voces) descargados por script — jamás vendorizados. Cada voz de Piper tiene licencia propia por dataset: verificar el MODEL_CARD de la elegida antes de citarla en el README.
- Minas de licencia conocidas: Qwen2.5-**3B** tiene licencia de investigación (no usar como fallback); Llama 3.x es licencia Meta no-OSI. Fallbacks Apache-2.0: Qwen3-4B, Mistral-7B-v0.3.
- Estructura de paquete: `audio/` (captura, rings, re-partición, decimador, playback) · `dsp/` (chain, stages — contrato público de fase 2) · `ears/` (wake, vad, segmentador, asr, gate) · `brain/` (cliente llama-server, prompt) · `voice/` (piper, splitter, cola de playback) · `fsm.py` · `events.py` · `config.py` · `cli.py`.
- README del hito: demo en GIF/video, diagrama del bus, tabla de latencias medidas, tabla de licencias, promesa honesta ("~4 s y habla fluido").

## 12. Plan de 3 semanas (S1 la más cargada)

- **S1 — el camino del audio:** repo + captura 48k (probar WASAPI el día 1) → ring → re-partición → cadena DSP → decimador → fan-out; playback + ring far-end; VAD segmentando utterances a WAV; JSONL + contadores. *Hito: audio limpio segmentado y medido, xruns dentro de criterio.*
- **S2 — oye, entiende, responde en texto:** wake word + prueba de acento; ASR; re-enrolamiento sherpa-onnx + gate; llama-server + cache_prompt + splitter. *Hito: "hey jarvis, ¿qué hora es?" → respuesta streameada en consola; una voz ajena no lo despierta.*
- **S3 — habla y vitrina:** Piper por oración + playback + barge-in + earcon; calibración del threshold; TOML; README + demo. *Hito: conversación multivuelta sin manos, grabada para el README.* (Colchón fino: la calibración puede resbalarse a una semana 4 corta.)

## 13. Roadmap de fases (post-esqueleto, cada una con su spec)

- **F2 — DSP pro en vivo** (la razón de ser): AEC (WebRTC APM/AEC3 sobre el ring far-end ya instalado), supresión de ruido, EQ paramétrico de voz, compresor. Se enchufan como `Stage`s sin tocar la topología.
- **F3 — Manos:** acciones sobre la máquina/servicios.
- **F4 — Cerebro profundo:** memoria, herramientas, personalidad.

## 14. Decisiones diferidas (con dueño y momento)

1. **Nombre del repo público** — DECIDIDO 2026-07-13: **`aurelius`** (elección de Samuel tras sesión de naming; sin colisiones en el campo de voz/audio). "Hey Jarvis" queda como wake word privado. Pendientes menores: dominio (aurelius.com pertenece a Aurelius Group; usar aurelius.dev u otro al montar el README) y verificar disponibilidad del nombre en PyPI antes del primer publish.
2. **Voz de Piper** — Samuel, a oído, en S3 (candidatas verificadas: es_MX-claude-high, es_MX-ale-medium, es_ES-davefx-medium, es_ES-sharvard-medium, es_AR-daniela-high).
3. **Política si el 7B se siente lento** — Samuel, con el loop andando en S2: aceptar 3.5-5 s con brevedad forzada, o fallback a Qwen3-4B (~-1.5 s, menos calidad de español).

## 15. Riesgos principales

| Riesgo | Mitigación |
|---|---|
| `hey_jarvis` preentrenado (TTS inglés) falla con el acento de Samuel | prueba de humo día 1; plan B acotado: entrenamiento sintético custom (~1.5 h) |
| Camino 48k inestable en WASAPI a tiempo parcial | cláusula de degradación a 16 kHz (§6) sin tocar el ML |
| Jitter de audio durante generación del LLM (contención de cores) | `-t 8-10`, blocksize 20-30 ms, colas acotadas, medición de xruns desde S1 |
| Eco del propio TTS re-despierta el sistema | hangover + drenaje del buffer; half-duplex; AEC en F2 |
| Latencia percibida decepciona | earcon < 150 ms, TTS solapado, respuestas breves forzadas, README honesto |
