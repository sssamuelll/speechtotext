# Benchmark: speechtotext vs TurboScribe

**Fecha:** 2026-07-07
**Audio:** ponencia de radio de Simón Ballesteros (papá de Samuel) — patología estructural, sismos, San Cristóbal–Táchira. Tramo evaluado ~25–46 min.
**Fuentes comparadas:**

- TurboScribe (ASR comercial): `E:\ponencia-papa 25-46min.txt`
- speechtotext (esta herramienta): `C:\Users\simon\Desktop\Papa\ponencia-papa_transcripcion.txt`

Este documento captura el diagnóstico para orientar el backlog. No es un README ni una spec.

---

## Veredicto por dimensión

TurboScribe gana las **tres** dimensiones de producto: diarización, precisión léxica y legibilidad/formato.

**Pero el dato clave:** el **núcleo acústico** de speechtotext es competitivo, y de hecho **gana los nombres propios más difíciles y regionales**. Donde el modelo comercial se equivoca, el nuestro acierta:

| Término | speechtotext | TurboScribe | Nota |
|---|---|---|---|
| **Cúcuta** | ✅ Cúcuta | ❌ "Cuxtra" | Terremoto de 1875, verificado histórico |
| **Yaracuy** | ✅ Yaracuy | ❌ "Yaraquí" | Estado real de Venezuela |
| **granito de arena** | ✅ granito de arena | ❌ "garnito de harina" | "garnito" ni es palabra |
| **nudos** | ✅ nudos | ❌ nodos | "nudos" es el término válido de ingeniería |

La conclusión estructural: **la brecha real no está en el modelo acústico, está en TODO el post-proceso.**

---

## Dónde falla speechtotext (todo post-proceso)

### 1. Puntuación y truecasing — ausente por completo
Salida sin comas, sin puntos, sin signos `¿?`. Muros de texto corridos. Mayúsculas y acentos erráticos: `tachira`, `guaira`, `caracas` en minúscula. Es la falla de mayor impacto en legibilidad y **no toca el modelo acústico**.

### 2. Alineación palabra→hablante (diarización) — va con retraso
La asignación de hablante llega tarde: **arrastra la cola de un turno al siguiente** y corta a mitad de sintagma (ej: `al profesor | Simón Ballesteros`, partido entre dos hablantes).

Su única ventaja —nombrar al invitado "Simón Ballesteros"— **se le vuelve en contra**: por el desfase, le atribuye frases ajenas, incluido el corte de estación y el ID de emisora del final (`833 minutos... san sebastián 92`).

### 3. Normalización de números y horas — sin criterio
Colapsa `8:33` en `833`. Mezcla letras y dígitos sin regla única.

### 4. Léxico regional — sorprendentemente bien, pero con huecos
Ya está muy por encima de lo esperado (ver tabla de arriba). Un **hotword list** cerraría casi toda la brecha restante. Nombres que sí falló:

| Correcto | speechtotext puso |
|---|---|
| Sofitasa | "sofitaza" |
| La Guaira | "agua ira" / "guayera" |
| Táchira | "tacho era" |
| aluvionales | "alubionales" |
| columna corta | "coluna corta" |
| casco histórico | "caco histórico" |
| urbanización Los Teques | "organización los teques" |
| Eliana | "aliena" |

**Falla real de la zona: Boconó.** Ninguno de los dos la acertó; TurboScribe quedó más cerca ("Evocono").

---

## Ventajas propias de speechtotext (a conservar)

- **Sin marca de agua.** TurboScribe incrusta publicidad al inicio y al final de la transcripción; el nuestro no.
- **Dígitos para magnitudes sísmicas** (`7.2` / `7.5`), más legibles que escribirlas en letras — aunque hoy es inconsistente.

---

## Backlog semilla (orden de ROI)

1. ~~**Restauración de puntuación + truecasing** sobre el texto plano.~~ **RESUELTO en Fase 0** — era talla de modelo, no post-proceso (ver abajo). No hace falta modelo restaurador.
2. **Arreglar el desfase temporal palabra→hablante en la diarización.**
   La cola de cada turno se está asignando al hablante siguiente; corregir el alineamiento resuelve las atribuciones erróneas (incl. el ID de emisora atribuido al papá).
3. **Normalización de números y horas.**
   Detectar `8:33` como hora; criterio único para dígitos vs letras. (Con large-v3 quedó `8.13`/`8.33` — solo falta `.`→`:` con contexto de hora.)
4. **Hotword list / léxico regional del Táchira.**
   Blindar lo que large-v3 aún falla: Sofitasa, Cúcuta, Boconó, casco histórico, granito de arena.

---

## Fase 0 — resultado (2026-07-08)

Misma ponencia, mismo CLI, un solo cambio: `-m large-v3` (antes `small`). Sin diarización, sin hotwords, sin ningún cambio de código. 23 min para 20.7 min de audio en CPU int8 (~1.1× tiempo real). Salida: `C:\Users\simon\Desktop\Papa\fase0-large-v3.txt`.

**Hipótesis confirmada: la brecha de puntuación/léxico era talla de modelo.** TurboScribe corre Whisper large; compararlo contra nuestro `small` no era pelea justa.

- **Puntuación/truecasing: resuelto de raíz.** Salida completa con comas, puntos, `¿?`, mayúsculas y tildes — paridad con TurboScribe o mejor. El ítem #1 del backlog muere: no hace falta restaurador.
- **Léxico: de perder por goleada a paridad.** Corregidos vs small: Eliana, La Guaira, Táchira, aluvionales, columna corta, urbanización Los Teques. **Boconó lo acertó una vez (línea "la falla de Boconó") — ningún otro motor lo logró, ni TurboScribe.**
- **Regresiones curiosas vs small:** `Cúcuta`→"Kutla" y `granito de arena`→"garnito de harina" (idéntico error que TurboScribe — large es más literal con el audio; small "alucinó" hacia la frase común y acertó de chiripa). Persisten: "Sofitaza", "del CACO" (casco histórico), "Tachibes" (1 de 6 menciones). Todo esto es exactamente lo que cubre el hotword list (#4).
- **Horas:** `8.13` / `8.33` — mucho mejor que el `833` de small; falta solo la regla `.`→`:` (#3).

**Backlog re-priorizado tras Fase 0:** (a) defaults/flags del modelo — recomendar `large-v3` para calidad, cablear `hotwords` + `condition_on_previous_text=False`; (b) alineación palabra→hablante (#2, sin cambios — sigue siendo el fix grande); (c) regex de horas (#3, trivial); (d) hotword list regional (#4).

**Trampas de Windows descubiertas en el camino** (fix pendiente en el CLI, `os.environ.setdefault` al arranque o README):
- `HF_HUB_DISABLE_SYMLINKS=1` — sin esto, la primera descarga de modelo muere con `WinError 1314` sin Developer Mode.
- `HF_HUB_DISABLE_XET=1` — sin esto, la descarga de modelos grandes se cuelga **en silencio** (el downloader xet en Rust; los archivos chicos bajan por HTTP normal y engañan).
