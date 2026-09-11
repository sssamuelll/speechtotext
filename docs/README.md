# Índice de `docs/`

Qué de aquí es guía viva y qué es arqueología. Un plan con casillas sin marcar no
significa trabajo pendiente: las casillas nunca se marcaron. El estado se
determinó cruzando cada documento contra `git log`.

## Vigente

| Documento | Qué es |
|---|---|
| [`api.md`](api.md) | Contrato para consumidores: esquema del JSON, evidencia de voz, registro de voces, verificación de modelos. Lo que se rompe si cambia sale en el `CHANGELOG.md`. |
| [`audio-evaluation.md`](audio-evaluation.md) | Runbook del corpus privado: cómo se mide WER, se entrena el calibrador y se administra la retención. Solo Windows. |
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

Dos preguntas del `plan-calidad-transcripcion-2.md` no se cerraron:

- **Q6** — los trozos huérfanos en `~/.speechtotext/chunks` no se limpian nunca.
- **Q8** — cuánta transcripción se pierde en las costuras entre trozos de
  `plan_chunks`, sin medir.

## Ajeno a este repo

[`superpowers/plans/2026-07-13-aurelius-esqueleto.md`](superpowers/plans/2026-07-13-aurelius-esqueleto.md)
(2149 líneas) es el plan de implementación de **aurelius**, el asistente de voz
que consume esta librería, no de `speechtotext`. Se quedó aquí por accidente de
dónde se escribió, y aquí sigue porque es la única copia que existe: el repo de
aurelius no lo tiene. Si algún día se mueve allá, que sea copiándolo primero.
