# Changelog

Los consumidores pinnean un tag, así que este archivo existe para una sola
pregunta: **¿me conviene subir el pin, y qué se me rompe si lo hago?**

Lo marcado como **rompe** exige cambios en el código que consume la librería.

---

## Sin publicar

### Añadido

- Las señales nativas del motor llegan al JSON: cada segmento puede traer
  `no_speech`, `avg_logprob` y `compression_ratio` (#23). Un valor no finito se
  trata como ausencia de medida y la clave se omite, en vez de escribir `NaN`,
  que ni siquiera es JSON válido. Contrato en [`docs/api.md`](docs/api.md#esquema-del-json).
- Documentación del contrato para consumidores (`docs/api.md`) y este changelog.

### Arreglado

- `speechtotext.__version__` decía `0.3.0` desde hace dos tags, y el docstring del
  paquete seguía anunciando un servicio HTTP de pronunciación que se fue en la
  `0.4.0`.

---

## v0.5.0 — 2026-09-11

### Añadido

- **Evidencia de voz por DSP determinista** (#22): `compute_voice_evidence` mide
  energía en la banda de voz, proporción de tramos sonoros, F0 mediana y planitud
  espectral. Son medidas, no veredictos — el módulo no clasifica voz contra
  no-voz a propósito, y distingue "no pude medir" (`None`) de "medí y no hay"
  (`0.0`). Contrato en [`docs/api.md`](docs/api.md#evidencia-de-voz).

### Cambiado — rompe

- **El registro de voces archiva y filtra por modelo.** `registry.enroll` exige
  ahora `model`, y `registry.get_embeddings(model)` exige el modelo como
  argumento obligatorio: el coseno entre dos espacios vectoriales distintos no
  significa nada, así que una voz solo se compara contra vectores del mismo
  extractor.

  Migración: los manifiestos planos de la `0.4.x` se siguen leyendo — se
  reagrupan en memoria al cargar y el disco se reescribe al formato nuevo en el
  siguiente `enroll` o `remove`. Lo que sí hay que tocar es el código: cualquier
  `get_embeddings()` sin argumento levanta `TypeError`.

  Cuidado con el modo de fallo silencioso: pedir un modelo sin voces enroladas
  devuelve `{}`, no un error. Un nombre mal escrito se ve igual que un registro
  vacío.

---

## v0.4.0 — 2026-08-07

### Cambiado — rompe

- **Se extrajo el servicio HTTP de evaluación de pronunciación** (FastAPI +
  Azure). Vivía en `api/` y hoy vive adaptado dentro de su único consumidor
  (klara). Esta librería quedó como lo que es: voz a texto local, sin servidor y
  sin dependencias de nube.
- A partir de aquí el repo se consume como **librería versionada por tags**. Los
  consumidores fijan tag o SHA, nunca `@main`: cambio que un consumidor necesite
  entra por PR aquí, se taggea, y recién entonces se sube el pin allá.
