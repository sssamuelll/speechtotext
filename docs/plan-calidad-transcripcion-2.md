# Plan de calidad de transcripción 2 — speechtotext

Estado: definitivo, ejecutable. Ship de la Fase 1: **viernes 2026-08-07**.
Árbol que ejecuta: `D:\Desktop\projects\.worktrees\audio-hibrido\speechtotext\src\speechtotext`
(rama `feat/audio-hibrido-seguro`).
Base: junta del 2026-08-03, cinco consejeros (Lindgren, Halberg, Serrano, Voronov, Stride).
Todo lo que entra aquí está verificado contra el código del árbol o medido sobre datos reales.

**Este documento es la continuación de `docs/plan-calidad-transcripcion.md`, no su reemplazo.**
Hereda su formato, su tono y su numeración: las contradicciones siguen desde **C-10** y las
cuestiones abiertas desde **Q7**. La sección 3 dice, ítem por ítem, qué partes del plan anterior
quedan cerradas, corregidas o refutadas por la evidencia nueva. Lo que aquel plan cerró con
refutación y la evidencia nueva no toca, no se vuelve a litigar.

---

## 1. El problema en una página

El plan anterior se envió casi entero: la herramienta ya se mide a sí misma. Imprime cobertura
(`cli/app.py:315-328`), calcula huecos (`core/formats.py:127`), marca `[?]` los segmentos de baja
densidad (`core/formats.py:33-47`), construye el modelo perezoso (`core/chunked.py:210-226`) y
distingue idioma forzado de idioma medido (`cli/app.py:333-338`).

Se volvió a transcribir la misma conversación de 36:46. Esto es lo que pasó:

1. **La consola dijo `83%` y `OK`, y el usuario no encontró los huecos.** Los huecos estaban
   calculados y escritos: `core/formats.py:127` los emite en el JSON desde el plan anterior. Pero
   `txt`, `srt` y `vtt` —los tres archivos que un humano abre— no llevan marca de hueco, y el
   único artefacto que sí la lleva es el que nadie abre. El fallo fue de **canal**, no de señal.
2. **El `83%` está inflado, y su sesgo crece con el fallo.** Con `vad_filter=True`, faster-whisper
   remapea los timestamps sumando el silencio eliminado (`faster_whisper/transcribe.py:1867-1868`
   sobre `faster_whisper/vad.py:274-275`). Un segmento que cruza dos trozos de VAD recupera un span
   que **contiene el silencio que el VAD quitó**. Medido sobre esta corrida: **125.7 s de silencio
   viven dentro de los spans**. Consola 83%, spans reales 73%, habla real contada por palabras 65%.
   Mientras peor el fallo del VAD, más generoso el número que lo reporta.
3. **El único dial que teníamos no disparó donde importaba.** `ratio < 0.7` (`cli/app.py:319`)
   contra un 73%: no avisó. Ni con el número correcto habría avisado. Y los tests fijan los dos
   extremos —25% "avisa" (`tests/test_cli.py:85-92`) y 90% "no avisa" (`:95-101`)— dejando sin
   cubrir la banda 70-90%, que es exactamente donde se pierde contenido caro.
4. **El operador leyó mal la señal y construyó la solución equivocada.** Concluyó que la
   herramienta había perdido 214 s, montó a mano un pipeline de cuatro pases para recuperarlos, y
   el pipeline **perdió contra la herramienta pelada**: 2675 palabras contra 2707, 1215 s de span
   contra 1331. La causa real de la degradación fue una lista de 25 hotwords que él mismo pasó, por
   una vía cuyo docstring dice algo falso sobre lo que hace. La sección 4 cuenta el episodio
   completo, porque es el hallazgo más instructivo del día.
5. **`--diarize` redefine en silencio tres instrumentos del plan anterior.**
   `speakers/diarization.py:51,56` recomprime cada span a la extensión de sus palabras. `cov` se
   calcula en `cli/app.py:315` (antes de diarizar) y `speech_s` en `core/formats.py:126` (después):
   **dos cantidades distintas con el mismo nombre en la misma corrida**. Confirmado en los datos
   nuevos: al diarizar, el span bajó de 1331 s a 1268 s con **más** palabras. Y el gate `dur >= 10.0`
   de `is_suspect` (`core/formats.py:39`) deja de dispararse: el caso canónico del propio plan
   anterior ("30 s con `Gracias.`") queda en ~1 s tras diarizar.

El denominador común de este ciclo es distinto al del anterior. Antes, **el sistema no tenía
ninguna medida de sí mismo**. Ahora la tiene, y el problema es otro: **la medida no llega por el
canal que el humano lee, cambia de significado según los flags, y el knob que el usuario tocó para
arreglarla documenta lo contrario de lo que hace**. No es un problema de instrumentación. Es un
problema de fidelidad: cada cosa que la herramienta afirma tiene que valer lo mismo en todos los
artefactos y bajo todas las banderas.

---

## 2. Diagnóstico de fondo

### Dónde convergieron los cinco

- **El núcleo enviable cabe en una frase: que la consola imprima los huecos que el JSON ya
  calcula.** No hay que calcular nada nuevo. `_gaps` (`core/formats.py:98-112`) ya existe, ya está
  probado (`tests/test_formats_speaker.py:76-94`) y ya corre en cada corrida que emite JSON. Falta
  llamarlo desde `cli/app.py` e imprimir el resultado.
- **Seis de siete ítems son ediciones en archivos existentes: ~130-150 líneas y dos decisiones
  arquitectónicas abiertas en total.** No es un rediseño. El único ítem grande que se propuso
  —consumir una segunda fuente— son ~250 líneas y cuatro decisiones, y está en la sección 6.
- **Ninguna medición nueva justifica tocar un parámetro de decodificación.** Cero cambios en
  `_transcribe_opts` (`cli/app.py:112-126`), cero invalidación de checkpoints, cero dependencia
  nueva. El techo del dial de VAD está medido y es +1.8% (ver 3.1).
- **La prosa que miente cuesta más que el código que falla.** Tres textos del árbol afirman cosas
  falsas sobre lo que hacen: el docstring de `_resolve_hotwords` (`cli/app.py:99-101`), el `--help`
  de `--hotwords` (`cli/app.py:440-441`) y el comentario de `core/engines.py:108`. El primero le
  costó una tarde a un usuario real. Se arreglan juntos en el ítem 2.2.
- **NO construir**: segunda fuente (captions, `mov_text`, segundo motor), cualquier calibrador, el
  pipeline de cuatro pases, guarda de cifras, recalibración del umbral de cobertura y el dial de
  `vad_parameters`. Ver sección 6 con motivos.

### Contradicciones reales, resueltas

**C-10. ¿La cobertura que imprime la consola mide lo que su nombre dice?**
Una posición: es la medida de sí misma que instaló el plan anterior, funciona y basta. Otra: está
inflada y no sirve como evidencia. **Gana la medición**: con `vad_filter=True` el audio se concatena
sin silencios y `faster_whisper/transcribe.py:1867-1868` devuelve los timestamps al eje original
sumando `total_silence_before` (`faster_whisper/vad.py:274-275`). Un segmento que cruza dos trozos
de VAD se lleva el silencio de por medio dentro de su span. Sobre esta corrida: 125.7 s de silencio
dentro de los spans, consola 83%, spans 73%, palabras 65%.
**Resolución**: el porcentaje se queda impreso como contexto barato —cuesta una resta y orienta—
pero deja de ser un veredicto. El instrumento accionable pasa a ser la lista de huecos, que es un
hecho geométrico sobre la línea de tiempo y no depende de cómo el VAD remapee nada. Ver 1.1.
**Corolario que hay que decir en voz alta**: hasta que Q7 se cierre, ningún porcentaje que imprima
la herramienta es evidencia de nada. Es una pista, y así se presenta.

**C-11. ¿El umbral de cobertura se recalibra o se borra?**
Una posición: subirlo de 0.70 a ~0.85 y el caso habría gritado. **Se cae por dos vías verificadas**:
(a) el número que el umbral compara está sesgado hacia arriba con un sesgo que **crece con el
fallo** (C-10), o sea que cualquier constante que se elija hoy queda mal calibrada justo en las
corridas peores; (b) el umbral ya falló su único caso real y sus dos tests
(`tests/test_cli.py:85-92` y `:95-101`) fijan 25% y 90%, dejando sin cubrir la banda donde el fallo
ocurrió.
**Gana borrarlo.** Un dial que falló su único caso real no merece afinación, merece desaparecer. La
línea de huecos sale **incondicional**, sin umbral que la porteé. Ver 1.2.

**C-12. ¿`--hotwords` es sesgo léxico o texto previo?**
El docstring del árbol afirma lo primero (`cli/app.py:99-101`: "los hotwords son sesgo
probabilístico"). **Falso, verificado en la librería**: `faster_whisper/transcribe.py:1542-1548`
mete los tokens de hotwords en el prompt **después de `tokenizer.sot_prev`**, o sea por el mismo
canal que el texto de la ventana anterior. No hay ningún prior sobre el vocabulario: hay una frase
inventada colgada delante de la ventana. Y se **trunca en silencio** por encima de
`self.max_length // 2 - 1 = 223` tokens (`transcribe.py:1546-1547` sobre `max_length = 448` en
`:722`), justo por la vía que el `--help` ofrece como "léxico por proyecto"
(`cli/app.py:448-449`).
**Resolución**: el mecanismo manda sobre la prosa. Se corrigen los tres textos y se agrega el aviso
de tamaño (ítem 2.2). El flag no se toca ni se quita: con listas cortas es útil, y la ablación lo
respalda (3 términos: 508 palabras contra 495 sin nada).

**C-13. ¿Puede `--diarize` cambiar el valor de una métrica de calidad?**
Una posición: los segmentos post-diarización son "los segmentos finales" y medir sobre ellos es
correcto. **Se cae**: `speakers/diarization.py:51,56` construye cada `LabeledSegment` con
`run_start`/`run_end`, o sea la extensión de sus **palabras**, no la del segmento que el ASR emitió.
Eso no es una métrica distinta, es la misma métrica sobre un objeto distinto, con el mismo nombre y
sin aviso. Medido: el span bajó de 1331 s a 1268 s con más palabras.
**Gana medir sobre lo que el ASR emitió, antes de cualquier post-proceso.** `cov` ya se calcula en el
sitio correcto (`cli/app.py:315`, antes de `_run_diarization` en `:347-348`); lo que falta es que el
JSON reciba ese valor en vez de recalcular el suyo. Una sola cantidad, calculada una vez, pasada
como kwarg. Ver 1.3. El tercer instrumento afectado, `is_suspect`, no cabe en el mismo diff (toca
cinco sitios de construcción) y va en 2.3.

**C-14. ¿La suma cruda de `cli/app.py:315` doble-cuenta cuando hay solape?**
Se afirmó que sí y que por eso el 83% estaba inflado. **Falso, y el mecanismo real es otro**: el
solape medido sobre los datos de esta corrida es **0.0 s** — `sum(end - start)` coincide exactamente
con la unión de los intervalos. La premisa que el plan anterior asumió en 1.1 ("los segmentos de
Whisper no se solapan dentro de una pasada y los trozos son contiguos por construcción") ahora está
**medida**, no supuesta. La inflación existe, pero viene del remapeo del VAD (C-10), no de contar
dos veces. **Resolución**: `cli/app.py:315` se queda como está. El comentario de `:310-314` sigue
siendo correcto y se le agrega la medición como respaldo.

### El tercer mecanismo de pérdida: las costuras

Dos de los huecos de esta corrida caen en **600.5** y **1200.2**: fronteras de trozo de
`plan_chunks` (`core/chunked.py:86-96`, `target_len = 600.0`, cortes elegidos por `pick_cuts` en
`:66-83`). Es un mecanismo de pérdida **proporcional al número de trozos** que ningún parámetro de
VAD toca y que ninguna fase de ningún plan cubre hoy. No entra en la Fase 1 porque no está
cuantificado: entra como Q8, con el experimento escrito.

---

## 3. Lo que la evidencia nueva corrige del plan anterior

### 3.1 · Q1 queda cerrada: el VAD **no** estaba ciego, y el dial no debe dispararse

El plan anterior dejó Q1 abierta con esta redacción: *"'1 segmento en 18 minutos' es ambiguo con el
código actual (…) un VAD que no detectó nada y uno que detectó todo como una sola región producen
la misma línea"*, y la usó como **disparador** de la Fase 3 (`vad_parameters` como dial del CLI).
La medición existe desde el 2026-08-03 y dice lo contrario.

**La medición.** Silero standalone sobre el archivo completo: **87.0% de habla (1398 s), en 4.5 s de
cómputo**. La herramienta emitió 1331 s de span. O sea el VAD dejó pasar **67 s más** de los que el
decoder convirtió en texto. El VAD no estuvo ciego: estuvo por delante del decoder. Lo que se perdió,
se perdió aguas abajo del VAD.

**El techo del dial, también medido.** `threshold` de 0.5 a 0.2 compra **+28.6 s (+1.8%)**, y la
variante agresiva **empeora** el resultado. Es decir: aunque el dial existiera y el usuario lo
girara hasta el fondo, el techo de la ganancia es 1.8 puntos sobre una cantidad cuyo sesgo de
medición es de nueve puntos (C-10). El instrumento sería más ruidoso que el efecto que mide.

**Consecuencias, explícitas:**

- **Q1 se cierra por medición.** No queda como cuestión abierta.
- **El disparador de la Fase 3 del plan anterior no se cumple.** `vad_parameters` como dial del CLI
  no se construye, y no vuelve a la mesa por "todavía no lo medimos": ya se midió. Baja de "Fase 3,
  gatillada por datos" a la sección 6 con el número escrito al lado.
- **El coste del instrumento que Q1 pedía era 270x menor que un pase de ASR.** El plan anterior ató
  Q1 al ítem 1.8 (logger de faster-whisper) y a "una corrida instrumentada", que se leía como cara.
  Un barrido de silero sobre el archivo entero cuesta 4.5 s. Q1 llevaba meses abierta por un precio
  que no existía.
- **1.8 se queda igual.** El logger (`cli/app.py:58-63`) sigue siendo útil por trozo y no cuesta
  nada, pero deja de ser "el instrumento que cierra Q1": Q1 la cierra un script de 4.5 s.

### 3.2 · La Fase 2 cuesta 15-20 minutos, no una noche de máquina

El plan anterior escribió, tres veces, que el recómputo de la Fase 2 costaba *"una noche de
`large-v3` en CPU"* (C-7, encabezado de la Fase 2, total de la Fase 2) y ató a esa noche dos
cuestiones abiertas: Q4 (`hallucination_silence_threshold`, *"hacerla junto con la noche de máquina
de la Fase 2"*) y, por transitividad, Q5 (`word_timestamps` fuera de la llave).

**Medido el 2026-08-03: el recómputo completo de los 36:46 son 15-20 minutos.** El plan
sobreestimaba por un orden de magnitud. Consecuencias:

- **Q4 queda desbloqueada.** La corrida diagnóstica que pide —`word_timestamps=True` con el umbral
  puesto, comparando qué habría borrado contra las islas de voz rescatadas— cuesta 20 minutos, no
  una noche. Sigue prohibido **encender** el parámetro (sección 6); medirlo ya es barato.
- **Q5 queda desbloqueada.** Confirmar que `word_timestamps=True` no altera el texto emitido es dos
  corridas de 20 minutos y un `diff`. Es el desperdicio más grande que documenta el disco
  (togglear `--diarize` cambia `word_timestamps`, que está en la llave en `core/chunked.py:111`, e
  invalida el 100% del trabajo) y llevaba meses aparcado por un precio inexistente.
- **Y aun así, la Fase 2 del plan anterior no sube de prioridad.** El coste bajó 20x, pero el
  consumidor de las señales nativas sigue sin probar su valor: la rama `no_speech` de `is_suspect`
  (`core/formats.py:36-38`) está muerta, y el eje vivo —densidad de caracteres— atrapa el 100% de la
  evidencia que motivó la marca. Barato no es lo mismo que necesario. Va a post-ship (5.3) con su
  pregunta delante (Q7 del plan anterior no existía; aquí es la parte de Q9 que la toca).
- **Dato colateral que también abarata el diagnóstico**: el probe de pistas del contenedor cuesta
  **12 ms**. Cualquier decisión futura sobre qué trae un archivo se toma con datos, no con
  suposiciones.

### 3.3 · C-9: su premisa no aplica a una pista de captions, pero el descarte se mantiene

El plan anterior descartó el consenso entre modelos en C-9 con dos argumentos. El segundo fue:
*"'dos modelos no alucinan la misma frase' es n=1: `large-v3` y `medium` comparten corpus de
subtítulos, por eso los dos conocen `'Gracias por ver el video'`"*.

**Esa premisa no aplica a una pista de captions del contenedor.** Una pista `mov_text` no es un
modelo, no comparte corpus con nadie y no aprendió `"Gracias por ver el video"` de YouTube: es texto
que alguien —humano o plataforma— escribió aparte. El argumento del corpus compartido no la toca.

**El descarte se mantiene, por otro motivo, y es más fuerte que el original:**

- Ata la corrección de la herramienta a **una pista que solo trae este contenedor de esta
  plataforma**. El día que el usuario transcriba un `.wav` de su grabadora, la ruta no existe y la
  calidad cae sin aviso. Es una dependencia de datos disfrazada de feature.
- Importa el problema del **empalme a nivel de palabra** entre dos fuentes con timestamps
  independientes: ~250 líneas y cuatro decisiones arquitectónicas abiertas, según la calibración de
  tamaño. Es el único ítem L de toda la mesa y justificaría documento propio.
- **Lo único que entregó de valor en el caso real fueron los nombres de los hablantes**, y eso sale
  más barato escribiendo dos nombres a mano. Receta en el README, no código.

O sea: **mismo veredicto que C-9, distinto motivo**. Queda en la sección 6 con la razón corregida,
para que nadie lo reabra atacando la premisa del corpus compartido —que efectivamente se cae— y
crea que con eso desbloqueó la construcción.

### 3.4 · Estado de cada fase del plan anterior

| Ítem del plan 1 | Estado | Dónde está en el árbol |
|---|---|---|
| 1.1 Cobertura en la línea de resumen | **Enviado** | `cli/app.py:310-328` |
| 1.2 `speech_s` y `gaps` en el JSON | **Enviado** | `core/formats.py:98-112`, `:126-127` |
| 1.3 Modelo perezoso (thunk con lock) | **Enviado** | `core/chunked.py:208-226`, `:157` |
| 1.4 Batch de pyannote a 8 | **Enviado** | `speakers/diarization.py:96`, `:130-131` |
| 1.5 Marca `[?]` | **Enviado, con dos defectos** | `core/formats.py:33-47` + writers; ver 2.3 |
| 1.6 Idioma medido vs forzado | **Enviado** | `cli/app.py:333-338`, `core/chunked.py:263-265`, `core/formats.py:118-123` |
| 1.7 Cobertura por trozo en el log | **Enviado** | `core/chunked.py:251-255` |
| 1.8 Logger de faster-whisper | **Enviado** | `cli/app.py:58-63` |
| 1.9 Guarda de memoria | **Enviado** | `cli/app.py:284-308` |
| 2.1 Señales nativas por segmento | **No enviado** — baja a post-ship (5.3) | `TimedSegment` (`core/chunked.py:33-38`) y `LabeledSegment` (`core/segments.py:7-12`) siguen sin los campos |
| 2.2 `json.dumps(opts)` en la llave | **No enviado** — post-ship (5.3), sin bug vivo | La llave sigue enumerada a mano en `core/chunked.py:108-112` |
| 2.3 `device`/`compute_type` en la identidad del caché | **Enviado y superado** — cerrado | `chunk_path` recibe `engine`, `quant` y `device` y los mete en el join (`core/chunked.py:99-116`, `:109`), vía el plan multimotor |
| 2.4 Reporte de calidad de diarización | **No enviado** — sube a Fase 2 de este plan (4.4) | La línea cambió: hoy es `cli/app.py:497-503`, no `:304-310` |
| Fase 3 · `vad_parameters` como dial | **Refutada** — pasa a la sección 6 | Disparador no cumplido (3.1) + techo medido en +1.8% |
| Fase 3 · Consenso entre modelos como subcomando | **Refutada** — pasa a la sección 6 | Cae bajo "segunda fuente" (3.3) |

Los defectos de 1.5, en concreto: (a) el gate `dur >= 10.0` (`core/formats.py:39`) deja de
dispararse bajo `--diarize`, y el test que lo certifica
(`tests/test_formats_speaker.py:121-130`) construye a mano un `LabeledSegment(0, 30, "Gracias.",
"Samuel")` que la ruta real **no puede producir** con `--diarize` puesto; (b) la rama `no_speech`
(`core/formats.py:36-38`) está muerta porque ninguna de las dos dataclasses tiene el campo y
`getattr(..., None)` la apaga en silencio. Queda **un solo eje vivo**: densidad de caracteres. Ambos
se atacan en 4.3.

### 3.5 · Estado de cada cuestión abierta del plan anterior

| | Estado |
|---|---|
| **Q1** · ¿VAD ciego o saturado? | **Cerrada por medición** (3.1): ni ciego ni saturado — dejó pasar 67 s más de los que el decoder usó. 87.0% de habla en 4.5 s de cómputo. |
| **Q2** · ¿Qué mató la corrida 2 (exit 5)? | **Cerrada con la evidencia disponible**: `--diarize` corrió hasta el final sobre el mismo audio con el batch de 1.4 ya aplicado (de esa corrida sale el dato de C-13). El propio Q2 fijó el criterio: *"si sobrevive, era presión de memoria y está resuelto"*. n=1: si vuelve a morir con exit 5, se reabre con el subproceso sobre la mesa. |
| **Q3** · ¿Cuánto añade el consenso sobre `[?]`? | **Cerrada por decisión, no por medición**: el consumidor que la justificaba —un segundo motor vivo— entró en la sección 6 (3.3). La respuesta operativa sigue siendo dos invocaciones y un `diff`, documentado en el README. |
| **Q4** · ¿Sirve `hallucination_silence_threshold`? | **Sigue abierta, ahora barata** (3.2): la corrida diagnóstica cuesta 20 min, no una noche. Encenderlo sigue prohibido. Se renumera como parte de Q9. |
| **Q5** · ¿`word_timestamps` en la llave? | **Sigue abierta, ahora barata** (3.2). Va a post-ship (5.3) con el experimento escrito. |
| **Q6** · Huérfanos en `~/.speechtotext/chunks` | **Sigue abierta, no bloquea nada.** Dato nuevo: la Fase 2 no se envió, así que no dejó los cuatro huérfanos que el plan preveía; el multimotor sí cambió la composición de la llave y esa invalidación única ya ocurrió y está asumida (`core/chunked.py:106`). |

---

## 4. El fallo del operador

Esta sección existe porque el hallazgo más instructivo del 2026-08-03 no fue un bug del código. Se
escribe con nombre y números para que no se repita.

### Qué pasó

Se transcribió la conversación de 36:46. La consola dijo `83%` y `OK`. El operador leyó los
timestamps del SRT, encontró 214 s sin texto, concluyó que la herramienta los había perdido, y
construyó a mano un pipeline de cuatro pases —amplificación, reencuadre, reintentos con distintos
flags— para recuperarlos. Y le pasó a la herramienta una lista de 25 hotwords con los términos que
esperaba oír.

### Qué dicen los checkpoints

Reconstruido desde `~/.speechtotext/chunks` (marcas de tiempo 17:48-17:51):

| | palabras | span | % del archivo |
|---|---|---|---|
| `speechtotext transcribe`, **una invocación** | **2707** | **1331 s** | 83% |
| pipeline manual de **cuatro pases** | 2675 | 1215 s | 76% |

El pipeline no recuperó nada: devolvió **32 palabras menos y 116 s menos de span**. La corrida
simple ya contenía `999`, `1500` (×4) y `500 euros`, y **cero** apariciones de
`www.ingenieros.com`. Todo lo que el pipeline "recuperó" ya estaba en la salida que se descartó por
mala.

### Qué causó la degradación

Ablación sobre 240 s del mismo audio, variando **solo** la lista de hotwords:

| hotwords | cobertura | palabras |
|---|---|---|
| ninguno | **94.1%** | 495 |
| **25 términos** | **85.0%** | **430** |
| 3 términos | 84.3% | 508 |

La lista larga se llevó **9.1 puntos de cobertura y 65 palabras (-13%)** respecto de no pasar nada.
Los cuatro pases después devolvieron el resultado a **por debajo del punto de partida**. La secuencia
completa fue: degradar la base con un knob mal entendido, medir la base degradada, y construir un
pipeline para compensar la degradación que uno mismo introdujo.

### Por qué pasa, mecánicamente

**`--hotwords` no es sesgo léxico.** `faster_whisper/transcribe.py:1542-1548`:

```python
if previous_tokens or (hotwords and not prefix):
    prompt.append(tokenizer.sot_prev)
    if hotwords and not prefix:
        hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
        if len(hotwords_tokens) >= self.max_length // 2:
            hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
        prompt.extend(hotwords_tokens)
```

Los términos entran al prompt **después de `tokenizer.sot_prev`**, o sea por el mismo canal que el
texto de la ventana anterior. No hay ningún prior sobre el vocabulario: hay una frase inventada
—una lista de nombres propios separados por comas— colgada delante de cada ventana de
decodificación, y el modelo la trata como conversación previa. Eso explica mecánicamente dos
síntomas que parecían inconexos:

- **La degradación ortográfica sin tildes.** La lista se escribió sin tildes; el modelo continúa el
  registro del texto que cree que viene antes.
- **El sangrado.** `"Ingenieros"` apareció insertado en una frase ajena. No es un término
  reconocido en el audio: es **continuación de contexto**, exactamente el fallo contra el que este
  mismo CLI puso `condition_on_previous_text=False` (`cli/app.py:124`, documentado en
  `_transcribe_opts` en `:114-118`). El proyecto cerró la puerta del arrastre de contexto por una
  vía y la dejó abierta por la otra, con un nombre que no sugiere que sea la misma puerta.

Y hay un tercer efecto que nadie ve: **el truncado silencioso**. Por encima de
`self.max_length // 2 - 1 = 223` tokens (`max_length = 448` en `transcribe.py:722`) la lista se
recorta sin aviso, sin log y sin excepción — justo por la vía que el `--help` ofrece como "léxico
por proyecto" (`cli/app.py:448-449`), que es precisamente la que invita a listas largas.

### El docstring es falso

`cli/app.py:99-101`:

```python
def _resolve_hotwords(hotwords: Optional[str], hotwords_file: Optional[Path]) -> Optional[str]:
    """Combina --hotwords (inline) y --hotwords-file. Scoped por invocación, sin default global:
    los hotwords son sesgo probabilístico, un léxico global envenenaría todo otro audio."""
```

**"los hotwords son sesgo probabilístico" es falso.** La segunda mitad de la frase —que un léxico
global envenenaría otro audio— resulta ser verdadera, pero **por un motivo distinto del que se
afirma**: envenena porque es texto previo, no porque sea un prior mal transferido. Un usuario que
lee ese docstring razona sobre el knob equivocado: si crees que estás dando un prior léxico, más
términos parece mejor; si sabes que estás escribiendo el párrafo anterior a mano, la lista de 25
suena a lo que es.

El mismo error está en el texto que **sí** lee todo el mundo: el `--help` de `--hotwords`
(`cli/app.py:440-441`) dice *"para sesgar el modelo en cada ventana"*.

### La lección

**El operador cambió dos cosas a la vez —25 hotwords y un pipeline de cuatro pases— y le atribuyó
el resultado a la herramienta.** Ninguna medida de cobertura lo habría impedido. Lo que sí lo habría
impedido es que el knob documentara lo que hace.

Lo que se lleva al código: el ítem **4.2** (corregir los tres textos y avisar del tamaño de la
lista). Lo que se lleva al proceso, y va escrito en el README junto a la receta de las dos
invocaciones: **antes de construir un pipeline para compensar una pérdida, corre una vez con los
flags pelados.** Cuesta 20 minutos y en este caso habría ahorrado la tarde entera.

---

## 5. Fases

Ordenadas por impacto/coste. Los arreglos de **una línea** van marcados `[1 LÍNEA]`.

### FASE 1 — Que la consola diga dónde falta audio (2 h 30 min, ship viernes 2026-08-07)

Objetivo: que el hueco que el JSON ya calcula llegue al canal que el humano lee, que ningún umbral
decida si el usuario merece verlo, y que consola y JSON dejen de emitir dos cantidades distintas con
el mismo nombre. **Un solo diff, dos archivos, una firma.** Cero checkpoints invalidados, cero
parámetro de decodificación tocado, cero dependencia nueva.

---

**5.1.1 · Los huecos salen por consola**
Archivos: `core/formats.py:98`, `core/formats.py:127`, `cli/app.py:42-48`, `cli/app.py:315-345`.

`_gaps` (`core/formats.py:98-112`) se renombra a `find_gaps` y se agrega al bloque de imports del
CLI (`cli/app.py:42-48`), junto a `write_json`. Un renombre y un call site (`core/formats.py:127`):
importar un nombre privado entre módulos es la clase de atajo que alguien "arregla" a las 3 a.m.
metiendo una copia de la función.

En `cli/app.py`, junto al cálculo de `cov` (`:315`) y **antes** de `_run_diarization` (`:347-348`),
`gaps = find_gaps(segments, info.duration)`. Después de la línea de resumen (`:342-345`), una línea
más:

- Con huecos: `N huecos sin texto: mm:ss-mm:ss (Xs), ... ` reusando `_fmt` (`cli/app.py:81-83`),
  que ya da mm:ss. Cuando `vad` está puesto, la línea cierra con `— prueba --no-vad`; con
  `--no-vad` o bajo whispercpp, no (el motor no tiene VAD que apagar, `cli/app.py:202-206`).
- Sin huecos: `sin huecos > 5 s`. **Se imprime explícitamente**, no se calla. Callar convierte la
  ausencia en una omisión indistinguible de un fallo del instrumento; decirlo la convierte en una
  afirmación que la herramienta puede sostener. Es el caso gemelo de Q9.
- Con `info.duration` falsy (probe fallido) **no se imprime nada de huecos**: sin denominador no hay
  línea de tiempo sobre la que existan complementos, y afirmar `sin huecos > 5 s` ahí sería
  fabricar. Esa rama ya imprime `cobertura desconocida` (`cli/app.py:328`) y el test que la fija
  (`tests/test_cli.py:104-112`) queda verde sin tocarlo.

`# ponytail: se listan los primeros 5 huecos y luego "y N más (ver el JSON)"; con 40 huecos la
consola se inunda y el artefacto completo ya existe. Techo: si alguien pide los 40 en consola, sale
un --gaps-all.`

**Aceptación**: la corrida del caso (2206 s de audio, texto en 0-300 y 600-850) imprime
`2 huecos sin texto: 05:00-10:00 (300 s), 14:10-36:46 (1356 s)` además del porcentaje; una corrida
con cobertura alta y sin huecos ≥ 5 s imprime `sin huecos > 5 s`.
**Test**: `tests/test_cli.py`, tres casos con `_fake_transcribe` (`:25-44`) — (a) segmentos con
hueco: el mm:ss del hueco sale en stdout; (b) un solo segmento contiguo: sale `sin huecos`; (c)
`info.duration == 0`: no sale ni `huecos` ni `sin huecos`.
**Coste: 1 h.**

---

**5.1.2 · El umbral `ratio < 0.7` se borra**
Archivo: `cli/app.py:316-328`.

Se elimina el bloque `if ratio < 0.7:` (`:319-323`) completo: el amarillo, el texto
`— se perdió audio` y el consejo condicional `, prueba --no-vad`. El porcentaje se sigue imprimiendo
—cuesta una resta y orienta— pero deja de portear la decisión de si el usuario ve o no el problema.
El consejo `--no-vad` se muda al pie de la línea de huecos (5.1.1), que es donde hay evidencia de
**qué** se perdió y no solo de cuánto.

No se recalibra a ningún otro número. Motivo en C-11: el valor que el umbral compara tiene un sesgo
que crece con el fallo, así que cualquier constante queda peor calibrada justo en las corridas
peores.

**Aceptación**: `--vad` sobre el caso del hallazgo imprime la lista de huecos y el porcentaje; la
cadena `se perdió audio` no aparece en ninguna salida del árbol.
**Tests que cambian** (cuatro, todos en `tests/test_cli.py`): `:85-92` pasa a asertar la línea de
huecos en vez del amarillo; `:95-101` pasa a asertar `sin huecos`; `:115-123`
(`test_no_sugiere_no_vad_a_quien_ya_lo_apago`) y `:223-231` (`test_whispercpp_no_sugiere_no_vad`)
mantienen su invariante —el consejo no aparece— movida a la línea de huecos. `:104-112`
(`test_resumen_sobrevive_duracion_cero`) no se toca.
**Coste: 30 min.**

---

**5.1.3 · Una sola cantidad, calculada una vez**
Archivos: `core/formats.py:115-127`, `cli/app.py:315`, `cli/app.py:382`.

`write_json` gana dos kwargs opcionales con el **mismo patrón condicional que ya usa `engine_info`**
(`core/formats.py:115`):

```python
def write_json(segments, info, path: Path, *, engine_info=None, speech_s=None, gaps=None) -> None:
```

Si llegan, se emiten tal cual en `:126-127`; si no llegan (`None`), se calculan como hoy. Los call
sites viejos y los tests existentes (`tests/test_formats_speaker.py:76-94`) no se mueven: cero
regresión.

El CLI calcula **una vez**, en `cli/app.py:315`, antes de diarizar, y los pasa en el lambda de
`:382`:

```python
"json": (".json", lambda p: write_json(segments, info, p, engine_info=engine_info,
                                       speech_s=round(cov, 2), gaps=gaps)),
```

`segments` sí se reasigna después (`:348`, `:352-355`) y el lambda captura el nombre, así que el
JSON sigue emitiendo los segmentos finales —correcto—; lo que deja de recalcularse es la métrica.
Sin esto, `speech_s` (`core/formats.py:126`) y `gaps` (`:127`) se calculan sobre segmentos que
`speakers/diarization.py:51,56` ya recomprimió a la extensión de sus palabras, y la misma corrida
publica 83% en consola y otro número en disco bajo el mismo nombre.

**Aceptación**: una corrida con `--diarize` produce un `speech_s` en el JSON idéntico al `cov` que
la consola usó para su porcentaje, y una lista `gaps` idéntica a la impresa. Sin `--diarize`, el
JSON no cambia byte a byte respecto de hoy.
**Test**: `tests/test_formats_speaker.py` — (a) `write_json` con `speech_s=99.0` y `gaps=[[1,2]]`
los emite tal cual sobre segmentos que darían otro valor; (b) sin los kwargs, los sigue calculando
(los tests `:76-94` cubren esta rama y quedan como regresión).
**Coste: 1 h.**

**Total Fase 1: 2 h 30 min.** Cero checkpoints invalidados. Cero parámetros de decodificación
tocados. Cero dependencias nuevas. Un commit.

---

### FASE 2 — Contratos y prosa que no mienten (esta semana o la siguiente, ~3 h 05 min)

Objetivo: que ningún flag pueda desactivar una función en silencio, y que ningún texto del árbol
afirme un mecanismo que no es el suyo.

---

**5.2.1 · Validación `min`/`max` de typer en tres flags** `[3 LÍNEAS × 2 COMANDOS]`
Archivos: `cli/app.py:424`, `:428-430`, `:434-436`; y los mismos flags en `find`: `cli/app.py:559`,
`:561`.

Los tres contratos están declarados **solo en la prosa del `--help`** y no los valida nadie:

- `--threshold` (`cli/app.py:434-436`, default `0.5`, "coseno, 0-1"): con `2.0` la identificación de
  voz queda desactivada en silencio —`assign_names` (`speakers/identify.py:30-32`) rompe el bucle en
  el primer candidato y devuelve `{}`—; con `-1` nombra todo. Se le pone `min=0.0, max=1.0`.
- `--speakers` (`cli/app.py:428-430`): `0` es falsy y `speakers/diarization.py:147`
  (`kwargs = {"num_speakers": num_speakers} if num_speakers else {}`) lo reinterpreta como "auto"
  sin aviso. Eso es exactamente la **sustitución callada** que la regla madre del contrato de
  capacidades prohíbe (`core/engines.py:19-21`). Se le pone `min=1`.
- `--beam-size` (`cli/app.py:424`): `min=1`.

Cero código propio: son tres kwargs de `typer.Option`. `find` los reexpone (`:559`, `:561`) y hereda
el mismo defecto, así que se corrige en los dos comandos o queda una puerta abierta.

**Aceptación**: `--threshold 2.0` y `--speakers 0` salen con exit 2 y el mensaje de typer, en
`transcribe` y en `find`.
**Test**: `tests/test_cli.py` — cuatro invocaciones con valores fuera de rango, exit 2 en las
cuatro.
**Coste: 20 min.**

---

**5.2.2 · La prosa que miente, corregida, y el tamaño de la lista, avisado**
Archivos: `cli/app.py:99-101`, `:220-222`, `:440-441`, `:448-449`; `core/engines.py:108`.

Tres textos y un aviso:

1. **Docstring de `_resolve_hotwords`** (`cli/app.py:100-101`): fuera "los hotwords son sesgo
   probabilístico". Entra el mecanismo real, con la referencia: los términos entran al prompt
   después de `tokenizer.sot_prev` (`faster_whisper/transcribe.py:1542-1548`), o sea **como texto
   previo**; por eso un léxico global envenenaría otro audio, y por eso la lista larga degrada.
2. **`--help` de `--hotwords`** (`cli/app.py:440-441`): fuera "para sesgar el modelo en cada
   ventana". Es el texto que sí lee el usuario y es donde el error hace daño.
3. **`--help` de `--hotwords-file`** (`cli/app.py:448-449`): "para un léxico por proyecto" se queda,
   con la advertencia del techo — por encima de 223 tokens la lista se **trunca en silencio**
   (`transcribe.py:1546-1547`, `max_length = 448` en `:722`), y esta es justo la vía que invita a
   listas largas.
4. **Comentario de `core/engines.py:108`** ("segmentos vacios no aportan texto y ensucian la
   cobertura"): la lógica de `:107` es **correcta** y se queda —descartar spans sin texto **baja** la
   cobertura, porque un span sin texto la inflaría—; el comentario dice lo contrario de lo que hace
   el código. Se reescribe. Es ruta whisper.cpp solamente.

**El aviso.** La línea que ya imprime los hotwords (`cli/app.py:220-222`) suma el conteo de términos
y de caracteres, y por encima de un techo conservador agrega la medición: *"25 términos degradaron
la cobertura 9 puntos sobre 240 s de audio real (2026-08-03); entran como texto previo, no como
léxico"*.

`# ponytail: se cuentan términos y caracteres, no tokens. El conteo exacto necesita el tokenizer del
modelo, que en este punto todavía no está cargado; 223 tokens son del orden de 600-700 caracteres en
español. Techo: si alguien necesita el número exacto, se cuenta después de construir el modelo.`

**Aceptación**: `--hotwords` con 25 términos imprime el conteo y la advertencia con el número
medido; la cadena `sesgo probabilístico` no aparece en el árbol; `speechtotext transcribe --help` no
dice "sesgar".
**Test**: `tests/test_hotwords.py` — (a) `assert "sesgo" not in _resolve_hotwords.__doc__`, guarda
barata contra la regresión de prosa (el defecto de este ciclo **fue** un docstring, así que el
docstring se testea); (b) un test de CLI que asserta el conteo y la advertencia en stdout con una
lista larga, y su ausencia con una de tres términos.
**Coste: 30 min.**

---

**5.2.3 · `is_suspect` sobrevive a `--diarize`**
Archivos: `core/segments.py:7-12`; `speakers/diarization.py:37`, `:51`, `:56`, `:77`;
`core/formats.py:35`.

Hoy la marca `[?]` tiene **un solo eje vivo** (densidad de caracteres) y ese eje se apaga bajo
`--diarize`: el gate `dur >= 10.0` (`core/formats.py:39`) mide la extensión del `LabeledSegment`, y
`speakers/diarization.py:51,56` lo construye con `run_start`/`run_end`, la extensión de sus
**palabras**. El caso canónico del plan anterior —30 s con `Gracias.`— queda en ~1 s y no dispara.
La señal que importa (un span largo con casi nada de texto) vive en el **padre**, así que hay que
llevarla.

`LabeledSegment` (`core/segments.py:7-12`) gana `src_dur: float | None = None` **al final y con
default**: todas las construcciones existentes son posicionales y no se rompe ninguna.
`assign_segments` lo rellena con `s.end - s.start` del segmento del ASR en los tres sitios de
construcción (`speakers/diarization.py:37`, `:51`, `:56`) y `apply_names` lo propaga (`:77`).
`is_suspect` usa `getattr(seg, "src_dur", None) or (seg.end - seg.start)` como denominador y como
gate (`core/formats.py:35`).

Los N runs de un mismo segmento heredan el mismo `src_dur`: **va documentado en el comentario**, o se
miente de nuevo. Es la aproximación correcta —vienen de la misma ventana de decodificación— y es
exactamente el mismo criterio que el plan anterior fijó para `no_speech` en su ítem 2.1.

La rama `no_speech` (`core/formats.py:36-38`) **no se toca**: sigue inerte hasta que las señales
nativas lleguen (5.3), y su test (`tests/test_formats_speaker.py:105-110`) ya cubre que se enciende
sola.

**Aceptación**: `--diarize` sobre el caso canónico marca `[?]` en el mismo segmento que sin
`--diarize`.
**Test**: (a) `tests/test_diarization_pure.py` — `assign_segments` sobre un segmento de 30 s con una
palabra de 1 s produce un `LabeledSegment` con `src_dur == 30.0`; (b)
`tests/test_formats_speaker.py:121-130` se reescribe para usar la forma que la ruta real **sí**
produce (`LabeledSegment(0.4, 1.4, "Gracias.", "Samuel", src_dur=30.0)`), en vez del
`LabeledSegment(0, 30, ...)` construido a mano que hoy certifica un caso imposible.
**Coste: 1 h 30 min.**

---

**5.2.4 · Reportar la calidad de la diarización**
Archivo: `cli/app.py:497-503` (`_run_diarization`). *(El plan anterior lo ubicaba en `:304-310`; el
archivo creció con el multimotor.)*

Con los datos que ya están en memoria en esas seis líneas: número de hablantes detectados
(`len(clusters)`), porcentaje de segmentos sin atribuir (`speaker is None` sobre `labeled`, `:497`),
y cuántos nombres se asignaron contra cuántas voces registradas (`len(name_map)` contra
`len(enrolled)`, `:498-502`). Sugerir `--speakers N` cuando el conteo automático supera lo razonable.

**Sube de prioridad respecto del plan anterior por un motivo nuevo**: la identificación de voz no
puede diagnosticarse. `speakers/identify.py:30-32` rompe el bucle en el primer candidato bajo umbral
**sin registrar el near-miss**, y el score no se imprime en ningún sitio. `registry.py:49` guarda
`seconds` del enrollment pero `get_embeddings()` (`:60-67`) lo descarta, así que `assign_names` no
puede ver la calidad de la muestra aunque quisiera. Y el aviso de `enroll` (<10 s,
`cli/app.py:617-620`) y el umbral que decide (0.5, `cli/app.py:434-436`) son dos constantes sin
relación establecida: una muestra de 13.2 s pasó el aviso y falló la identificación, sin que nada lo
dijera.

Extensión barata sobre el plan anterior: **cuando `name_map` sale vacío, imprimir el mejor score**.
No hace falta tocar `assign_names` ni sus tests: `cosine` ya está exportado
(`speakers/identify.py:7-11`) y el CLI tiene `clusters` y `enrolled` en la mano; son dos líneas de
comprensión, solo en la rama vacía.

**Aceptación**: sobre la corrida del caso, la salida dice algo como
`2 hablantes · 0% sin atribuir · 0 de 1 voces identificadas (mejor score 0.38 < 0.50)`.
**Test**: `tests/test_cli.py` con `diarization.diarize` y `registry.get_embeddings`
monkeypatcheados — (a) dos clusters y ninguna voz registrada: sale el conteo y no sale la línea de
score; (b) dos clusters y una voz que no alcanza el umbral: sale el mejor score y el umbral.
**Coste: 45 min.**

**Total Fase 2: ~3 h 05 min.** Tampoco invalida checkpoints.

---

### FASE 3 — Post-ship, gatillada por datos

Nada de esto se agenda. Se listan con su disparador y con el coste corregido, porque la evidencia
nueva abarató tres de los cuatro.

- **Señales nativas por segmento + `json.dumps(opts)` en la llave** (Fase 2 del plan anterior,
  ítems 2.1 y 2.2). El coste bajó 20x (3.2): el recómputo son 15-20 min, no una noche. Siguen en
  post-ship porque **su consumidor no ha probado su valor**: la rama `no_speech`
  (`core/formats.py:36-38`) está muerta y el eje de densidad atrapa el 100% de la evidencia que
  motivó la marca. Sobre `json.dumps`: no hay bug vivo — el único campo de `opts` fuera de la llave
  (`core/chunked.py:108-112`) es `condition_on_previous_text`, que es constante (`cli/app.py:124`), y
  `cpu_threads`/`num_workers` producen texto byte a byte idéntico con configuraciones distintas
  (medido). Se arregla **el día que se agregue un knob**, y en el mismo commit que 2.1 para pagar una
  sola invalidación. **Disparador**: Q9.
- **Sacar `word_timestamps` de la llave del caché** (`core/chunked.py:111`, Q5 del plan anterior).
  El desperdicio más grande que documenta el disco: togglear `--diarize` invalida el 100% del
  trabajo. Un checkpoint **con** palabras es superconjunto de uno sin ellas. **Disparador**:
  confirmar que `word_timestamps=True` no altera el texto emitido — dos corridas de 20 min y un
  `diff`, no una noche.
- **Costuras de `plan_chunks`** (`core/chunked.py:86-96`, `pick_cuts` en `:66-83`). Dos de los
  huecos de esta corrida caen en 600.5 y 1200.2, fronteras de trozo. **Disparador**: Q8.
- **`hallucination_silence_threshold`, medido y no encendido** (Q4 del plan anterior). Sigue
  prohibido encenderlo (sección 6); la corrida diagnóstica ahora cuesta 20 min. **Disparador**: Q9.

---

## 6. Lo que NO se construye

| Descarte | Motivo |
|---|---|
| **Consumir una segunda fuente** (pista `mov_text`, captions del contenedor, segundo motor vivo) | Ata la corrección de la herramienta a una pista que solo trae **este contenedor de esta plataforma**: con un `.wav` de grabadora la ruta no existe y la calidad cae sin aviso. Importa el empalme a nivel de palabra entre dos fuentes con timestamps independientes: ~250 líneas y cuatro decisiones arquitectónicas abiertas, el único ítem L de la mesa. Y lo único que entregó de valor —los nombres de los hablantes— sale más barato escribiendo dos nombres. **Receta en el README, no código.** Mismo veredicto que C-9 del plan anterior, con el motivo corregido: la premisa del corpus compartido no aplica a una pista de captions (ver 3.3), pero el descarte no dependía de ella. |
| **`vad_parameters` como dial del CLI** (`threshold`, `min_silence_duration_ms`, `speech_pad_ms`) | Era la Fase 3 del plan anterior, gatillada por *"que la corrida instrumentada de Q1 muestre VAD ciego"*. **El disparador no se cumple**: el VAD dejó pasar 67 s más de los que el decoder convirtió en texto (87.0% de habla contra 1331 s emitidos). Y el techo del dial está medido: `threshold` 0.5→0.2 compra **+28.6 s (+1.8%)** y la variante agresiva **empeora**. El instrumento sería más ruidoso que el efecto que mide (el sesgo de la propia cobertura es de nueve puntos, C-10). No vuelve por "todavía no lo medimos". |
| **Cualquier calibrador de confianza** (`confidence/calibration.py`, corpus etiquetado, `evaluation/`) | Ya muerto en la sección 4 del plan anterior y la evidencia nueva no lo resucita: sigue exigiendo ≥9 fechas de grabación, ≥30 min por partición y transcripción humana de referencia por clip, sobre un corpus que no existe, y sigue auto-invalidándose ante cualquier cambio de flags. La marca `[?]` se lleva el 80% del valor por el 2% del coste. |
| **El pipeline de cuatro pases** | **Perdió contra la herramienta pelada**: 2675 palabras / 1215 s de span contra 2707 / 1331 (sección 4). Es el argumento más fuerte que existe contra construirlo y sale gratis: ya está medido. Lo que "recuperó" ya estaba en la corrida simple. |
| **Recalibrar el umbral de cobertura a cualquier otro número** | Ver C-11 y 5.1.2: el valor que compara tiene un sesgo que crece con el fallo, así que la constante queda peor calibrada justo en las corridas peores. Se borra, no se afina. |
| **Guarda de cifras / marca `[?]` sobre números** (`999`, `1500`, `500 euros`) | Depende de tener una segunda fuente contra la cual verificar, y la segunda fuente está descartada arriba. Sin ella, marcar cifras sería marcar **todas** las cifras: ruido con formato de señal. |
| **Tocar la cadena de audio** (ganancia, AGC, VAD propio, `clip_timestamps`) | C-4 del plan anterior sigue en pie —silero mantiene 999.6 s de 1001.3 con -46 dB, y lo que fragmenta es el *clipping*— y la evidencia nueva lo refuerza: el VAD no fue el culpable (3.1). Construir la solución de un fallo cuyo mecanismo no se conoce garantiza dos bugs en vez de uno. |
| **`-j` como palanca de throughput** | Medido el 2026-08-03: un trozo solo a 8 hilos va a **1.74x tiempo real**; a 24 hilos, **2.1x**. La paralelización está devolviendo **menos** que el serial. Sumado a C-3 del plan anterior (`num_workers` no replica pesos), `-j` no se toca en ninguna dirección: ni sube, ni baja, ni sale del CLI. |
| **Cambiar el descarte de segmentos vacíos** (`core/engines.py:107`) | La lógica es **correcta**: descartar spans sin texto **baja** la cobertura, porque un span sin texto la inflaría. Lo único mal es el comentario de `:108`, y eso se arregla en 5.2.2. Es ruta whisper.cpp solamente. |
| **Reabrir `"selection": "explicit"` como ítem suelto** (`cli/app.py:371`) | Afirma en el 100% de las corridas por defecto que el usuario eligió motor, y es el mismo defecto que C-5 mató en `Idioma detectado:`. Pero arreglarlo suelto repite el error de fondo: se aplicó la lección a un campo en vez de convertirla en regla. Entra cuando exista la regla, y entonces se auditan todos los campos del payload de una pasada. Ver Q10. |
| **`--prune` de checkpoints huérfanos** | Q6 del plan anterior sigue sin bloquear nada, y la invalidación única del multimotor ya ocurrió y está asumida (`core/chunked.py:106`). Borrar la carpeta a mano cuesta un comando. |

---

## 7. Cuestiones abiertas

Numeradas continuando desde Q6 del plan anterior.

**Q7 · ¿Qué cantidad física es "con voz" bajo `--vad`?**
Hoy la herramienta imprime un número y no hay definición escrita de qué mide. Sobre esta corrida
existen **tres** respuestas distintas y todas defendibles: consola **83%** (suma de spans tal como
faster-whisper los devuelve), spans reales **73%** (descontando los 125.7 s de silencio que el
remapeo del VAD metió dentro, `transcribe.py:1867-1868` sobre `vad.py:274-275`), y habla real
contada por palabras **65%**. **Decisión que falta**: elegir cuál de las tres declara la herramienta,
declararlo en el `--help` y en el JSON, y aceptar la consecuencia. Sostiene C-10 y C-11. **Bloquea**:
cualquier umbral futuro sobre el porcentaje, y cualquier afirmación de que un porcentaje es evidencia
de algo. Mientras siga abierta, el porcentaje es una pista, no un veredicto — y así se presenta desde
la Fase 1.

**Q8 · ¿Cuánto pierden las costuras de `plan_chunks`?**
Dos huecos de esta corrida caen en 600.5 y 1200.2, fronteras de trozo (`core/chunked.py:86-96`).
Es un mecanismo de pérdida **proporcional al número de trozos** que ningún parámetro de VAD toca y
que ninguna fase cubre. **Dato que falta**: correr el mismo audio con `--no-chunk` y con un
`target_len` distinto, y medir la diferencia de palabras en ventanas de ±5 s alrededor de cada
frontera. Con el recómputo completo en 15-20 min (3.2), el experimento cuesta una tarde, no una
semana. Si la pérdida por costura es material, la corrección probablemente sea solape entre trozos
con deduplicación —que es un ítem M con decisiones propias, no un parche.

**Q9 · ¿La ausencia de `[?]` es una afirmación o una omisión?**
Nunca se declaró. Hoy un archivo sin `[?]` puede significar "nada sospechoso" o "el único eje vivo
no aplicaba", y desde `--diarize` también podía significar "el gate no disparó porque el span se
recomprimió" (que 5.2.3 arregla). El caso gemelo de los huecos se resuelve en 5.1.1 imprimiendo
`sin huecos > 5 s` en vez de callar; falta la decisión equivalente para la marca. Y con ella viene
la pregunta de si el segundo eje merece existir: **¿cuánta alucinación atrapa `no_speech` que la
densidad de caracteres no?** Esa medición cuesta 20 min desde 3.2, y es el disparador de las
señales nativas (5.3) y de Q4 del plan anterior. Una sola corrida contesta las dos.

**Q10 · ¿Cuál es la regla general de "no afirmar lo que no se midió"?**
C-5 del plan anterior la aplicó a un campo (`Idioma detectado:` sobre el flag del propio usuario) y
funcionó. Pero no se convirtió en regla: `"selection": "explicit"` (`cli/app.py:371`) afirma en el
100% de las corridas por defecto que el usuario eligió motor, y `language_probability=1.0`
(`core/chunked.py:264`) se sostiene solo porque es fiel al upstream, no porque alguien lo haya
decidido como política. **Decisión que falta**: escribir la regla —qué puede afirmar un campo del
payload y qué tiene que omitirse— y auditar los campos existentes contra ella **de una sola pasada**,
en vez de arreglar uno por incidente. La línea de `selection` es trivial; la regla no lo es, y es la
que impide el tercer incidente.

**Q11 · ¿Puede `--diarize` mover cualquier número que la herramienta reporta sobre sí misma?**
Hoy mueve tres (C-13). La Fase 1 cierra dos (`speech_s` y `gaps`, ítem 5.1.3) y la Fase 2 cierra el
tercero (`is_suspect`, ítem 5.2.3). Lo que queda abierto no es un bug, es una regla: **las métricas
de calidad se calculan sobre los segmentos que el ASR emitió, antes de cualquier post-proceso**. Hoy
eso se cumple porque cada ítem se acordó de cumplirlo, no porque la estructura lo garantice. **Decisión
que falta**: si se hace estructural —por ejemplo, calculando el bloque de calidad una sola vez y
pasándolo como dato inmutable a todo lo que escribe— o si se deja como convención documentada. Lo
segundo es más barato hoy y garantiza que este mismo hallazgo reaparezca con el cuarto post-proceso
que alguien agregue.
