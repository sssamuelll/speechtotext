# Índice de `docs/`

Qué de aquí es guía viva y qué es arqueología. Un plan con casillas sin marcar no
significa trabajo pendiente: las casillas nunca se marcaron. El estado se
determinó cruzando cada documento contra `git log`.

## Vigente

| Documento | Qué es |
|---|---|
| [`api.md`](api.md) | Contrato para consumidores: esquema del JSON, evidencia de voz, registro de voces, verificación de modelos. Lo que se rompe si cambia sale en el `CHANGELOG.md`. |
| [`superpowers/specs/2026-07-13-jarvis-voice-core-design.md`](superpowers/specs/2026-07-13-jarvis-voice-core-design.md) | Diseño del asistente de voz local que consume esta librería. Sigue informando trabajo: la evidencia de voz de la `0.5.0` nació de aquí. |

## Ejecutado

Se conservan porque explican **por qué** el código quedó como quedó. No son
trabajo pendiente.

| Documento | Dónde aterrizó |
|---|---|
| [`plan-multimotor.md`](plan-multimotor.md) | `--engine faster-whisper\|whispercpp`, en `core/engines.py` y `core/enginepin.py`. |
| [`plan-calidad-transcripcion.md`](plan-calidad-transcripcion.md) | Línea de huecos en consola, mínimos y máximos de los flags, prosa de `--hotwords`. |
| [`plan-calidad-transcripcion-2.md`](plan-calidad-transcripcion-2.md) | `is_suspect` sobreviviendo a `--diarize`, reporte de calidad de diarización, y las señales nativas en el JSON (su pregunta Q9). |
| [`benchmark-turboscribe.md`](benchmark-turboscribe.md) | Diagnóstico contra TurboScribe de julio de 2026; su propio backlog quedó resuelto (puntuación, hotwords, alineación palabra→hablante). |
| [`superpowers/plans/2026-07-07-diarizacion-identificacion-batch.md`](superpowers/plans/2026-07-07-diarizacion-identificacion-batch.md) | `speakers/` y los subcomandos `enroll` / `voices` / `forget`. |
| [`superpowers/plans/2026-07-07-find-buscador-segmento.md`](superpowers/plans/2026-07-07-find-buscador-segmento.md) | `core/finder.py` y el subcomando `find`. |
| [`superpowers/plans/2026-07-08-transcripcion-chunked.md`](superpowers/plans/2026-07-08-transcripcion-chunked.md) | `core/chunked.py`: troceo por silencios, checkpoint y `--jobs`. |

Sus specs correspondientes están en [`superpowers/specs/`](superpowers/specs/).

### Lo que quedó abierto

Una pregunta del `plan-calidad-transcripcion-2.md` sigue sin cerrarse:

- **Q6** — los trozos huérfanos en `~/.speechtotext/chunks` no se limpian nunca.

**Q8** (cuánta transcripción se pierde en las costuras de `plan_chunks`) se midió el
2026-09-11 sobre 14 minutos de reunión real en español: **cero palabras en la
costura** — el corte cae en un silencio y las frases de los dos lados llegan
enteras. El coste del troceado está en otra parte: cada trozo después del primero
decodifica con las ventanas de 30 s corridas y deriva un 2–3 % respecto al pase
único. Y sin VAD, el trozo termina en silencio y Whisper alucina sobre el relleno
de la última ventana — una despedida de YouTube con 30 s de marca falsa que caía
encima del trozo siguiente; `clip_to_end` en `core/chunked.py` la recorta desde
entonces.

## Ajeno a este repo

[`superpowers/plans/2026-07-13-aurelius-esqueleto.md`](superpowers/plans/2026-07-13-aurelius-esqueleto.md)
(2149 líneas) es el plan de implementación de **aurelius**, el asistente de voz
que consume esta librería, no de `speechtotext`. Se quedó aquí por accidente de
dónde se escribió, y aquí sigue porque es la única copia que existe: el repo de
aurelius no lo tiene. Si algún día se mueve allá, que sea copiándolo primero.
