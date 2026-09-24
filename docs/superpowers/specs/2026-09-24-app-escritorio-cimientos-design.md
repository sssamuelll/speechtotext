# App de escritorio — diseño de los cimientos

**Fecha:** 2026-09-24 · **Estado:** aprobado en brainstorming; pendiente de revisión del spec y de plan
**Alcance:** los cimientos de la app de escritorio para la persona C del lanzamiento público: el motor como programa, el protocolo, la entrega de modelos, el camino básico en las dos apps y la distribución. Las partes 2, 3 y 4 (§2) tendrán su propio spec.
**Dónde vive el código:** el núcleo cambia en este repo (§6.4); todo lo demás, en un repo nuevo (§4).
**Idioma de este documento:** español — documento de trabajo, como el resto de `docs/superpowers/`.

---

## 1. Objetivo

La persona C del diseño del lanzamiento público (`2026-09-12-lanzamiento-publico-design.md`, §2: la persona no técnica, con app de escritorio) instala la app con dos clics, suelta una grabación y recibe la transcripción separada por hablantes. Sin cuentas, sin tokens, sin avisos de seguridad y sin que el audio salga del equipo. Pública, gratis y de código abierto, como la librería.

"La interfaz, no una cualquiera" se definió con cuatro pilares, todos a la vez:

1. **Facilísima:** un solo camino (soltar, esperar, leer); la máquina elige motor y modelo.
2. **Se siente de primera:** cuidada, ligera, rápida y hecha para cada sistema.
3. **Hace lo que otras no:** tocar una palabra y oírla, recordar voces, buscar en todas las grabaciones.
4. **Dice en qué no confiar:** pasajes dudosos y hablantes inciertos a la vista.

**Criterio de éxito de los cimientos:** alguien que nunca abrió una terminal descarga la app en Mac o en Windows, la instala, arrastra una grabación y recibe la transcripción con hablantes, sin cuentas, tokens ni advertencias de seguridad.

## 2. Descomposición

| Parte | Contenido | Pilares |
|---|---|---|
| **1. Cimientos** (este spec) | Motor como programa, protocolo, modelos, camino básico en las dos apps, distribución | 1, 2 |
| 2. El documento | Tocar una palabra y oírla, renombrar hablantes, recordar voces, exportar a Word | 3 |
| 3. La biblioteca | Todas las grabaciones, y buscar en todas a la vez | 3 |
| 4. La confianza | Pasajes dudosos y hablantes inciertos a la vista; necesita señales nuevas del núcleo | 4 |

Cada parte sigue su ciclo de spec, plan e implementación. Los cimientos dejan sitio a las otras tres sin rehacer nada: el motor guarda cada resultado (la base de la biblioteca), el protocolo admite métodos nuevos sin romper (§6.5) y la ventana de lista y contenido crece hasta la biblioteca.

## 3. Decisiones cerradas

| Decisión | Elegido | Descartado y por qué |
|---|---|---|
| Para quién | Público general, gratis, código abierto | Una persona concreta primero; producto de pago |
| Interfaz | Nativa en cada sistema: SwiftUI en Mac, WinUI 3 (C#) en Windows | Tauri + TypeScript, la recomendación de la sesión (ventana ligera, una sola interfaz, el stack diario del autor): se eligió nativa por la sensación de cada sistema, asumiendo dos interfaces para siempre. Electron: su Chromium suma unos 100 MB y un par de cientos de MB de RAM solo para la ventana, que en un equipo de 8 GB se le quitan al modelo. Móvil: el motor no corre allí sin reescribirlo. Interfaz en Python (Qt, pywebview): no trae instaladores firmados ni actualizador, y cuesta que se sienta de primera |
| Orden | Los dos sistemas a la vez | Mac primero; Windows primero |
| Motor | La librería empaquetada como programa aparte; JSON-RPC por stdin/stdout | Servidor local: puertos, y cualquier proceso del equipo podría hablarle. Motor reescrito en nativo: se perdería el pipeline medido |
| Ventana | Lista a la izquierda, contenido a la derecha | Un solo foco: se queda corto con la cola y la biblioteca. Una ventana por grabación: poco natural en Windows |
| Dirección visual | "Swiss con calidez" (§7.3) | El tema Swiss de una librería de componentes propia (React, privada) portado tal cual —1 px, cero radio, sin sombras—: se sintió frío. Quedó como base |
| Idiomas de la interfaz | Español e inglés desde el primer día | — |
| Modelo de transcripción | `large-v3`, sin selector | `small` por defecto: cambia lo que se dijo (`docs/design.md`) |
| Diarizador | Nemotron (§5.4) | pyannote: 25 min frente a 55 s por hora de audio en CPU, y pide cuenta de Hugging Face. Llega con la parte 2, para poner nombres |
| Mac | Apple Silicon, macOS 14 o posterior | Intel: torch 2.14 solo publica `macosx_14_0_arm64` |
| Windows | x64; Windows 10 (1809 o posterior) y 11 | ARM64: ni torch ni CTranslate2 publican para `win_arm64` |
| Linux | El CLI y la librería | App gráfica |
| Distribución en Mac | `.dmg` notarizada y Sparkle | Mac App Store: pendiente de probar el motor dentro del sandbox |
| Distribución en Windows | Microsoft Store (MSIX) | Instalador propio firmado: Azure Artifact Signing solo admite particulares de EE. UU. y Canadá, y desde 2024 ni los certificados EV dan reputación inmediata en SmartScreen |
| Cuenta de publicación | Particular | Empresa: solo si se publica con una marca |
| Telemetría | Ninguna | — |

## 4. Arquitectura

```
App Mac (SwiftUI) ────┐
                      ├── JSON-RPC por stdin/stdout ──▶ motor ──▶ speechtotext @ tag
App Windows (WinUI) ──┘                               (Python empaquetado, un proceso por sesión)
```

- **Un repo nuevo, público y en inglés**, con `mac/`, `windows/`, `engine/`, `protocol/` y `design/` (tokens y textos compartidos, §7). Usa `speechtotext` fijado a un tag, como decidió el diseño del lanzamiento. Lo que las apps necesiten del núcleo entra en este repo por PR, se publica con un tag y allí se sube el pin. El repo se crea en el paso 2 (§10), con el nombre de la app ya decidido.
- **Un proceso del motor por sesión.** La app lo arranca al abrirse y lo mantiene vivo: el modelo se carga una vez y se reutiliza entre archivos (`transcribe(backend=…)`), salvo con poca memoria (§5.5). Si el motor se cae, la app lo detecta al cerrarse el canal, lo dice y lo relanza. Si la app muere, el motor ve cerrarse stdin y sale.
- **stdin/stdout y no un servidor local:** sin puertos que abrir ni proteger, solo la app puede hablarle y muere con ella. Es el canal del servidor MCP, y funciona por la regla del núcleo: el núcleo nunca imprime.
- **`protocol/` es el contrato entre Python, Swift y C#** (§6).
- **El motor va dentro del instalador; los modelos no.** Se descargan la primera vez, anunciando tamaño y progreso; después todo funciona sin conexión.

## 5. El motor

### 5.1 Qué es

Un programa corto en `engine/` que lee peticiones por stdin, llama a la librería y responde por stdout. No transcribe nada por su cuenta: todo pasa por `transcribe()`, `probe` y `models`, las mismas funciones que usa el CLI. Guarda cada resultado en su carpeta de datos, con un identificador propio.

### 5.2 El canal, protegido

Al arrancar, antes de importar nada pesado, el motor duplica el descriptor 1 para el protocolo y redirige el descriptor 1 a stderr (`os.dup` y `os.dup2`). Así ni una barra de progreso de transformers ni un `printf` de una librería en C pueden escribir en el canal. stderr va a un log rotativo que la app adjunta a "Copiar diagnóstico". Todo en UTF-8, también en Windows.

### 5.3 Empaquetado

Dos candidatos; decide la prueba de empaquetado (§10, paso 0):

- Python portátil (python-build-standalone, el que usa uv) con las dependencias instaladas dentro desde un lockfile.
- PyInstaller en modo carpeta.

Criterios, en orden: que arranque en un equipo limpio; que pase la notarización de Apple (cada binario interno firmado, runtime endurecido y los entitlements que exija el JIT de numba si librosa lo arrastra); el tamaño; el tiempo de arranque. ffmpeg va dentro, en una compilación LGPL y en el PATH del motor, porque `plan_chunks` usa `silencedetect`.

### 5.4 Qué usa por defecto

- **Modelo: `large-v3`** (3,09 GB), con el tiempo estimado a la vista antes de empezar. Sin selector en esta versión. Si el sondeo corta con `insufficient_resources`, la app ofrece el borrador `small` (0,49 GB) como lo que es: más rápido, y puede cambiar frases enteras.
- **Hablantes: Nemotron** (0,40 GB, sin cuenta, licencia OpenMDW 1.1). En la llamada privada de 64 minutos del benchmark separó hablantes en 55 s frente a los 25 min de pyannote, con menos palabras mal asignadas que pyannote cuando nadie le dice cuántos hablan (1,55 % frente a 1,75 %), que es justo el caso de la persona C. No pone nombres. Salvedad: en cuatro reuniones AMI, tal como sale del modelo, dio un 30,1 % de DER frente a un 20,1 %, porque corta el turno en cada pausa; se mide con reuniones de tres o más personas antes de publicar (§10, paso 5).
- **Idioma: `auto`.** Se muestra en cuanto se detecta; "Cambiar" cancela y vuelve a empezar con el idioma elegido.
- **Primera descarga: 3,5 GB** (large-v3 y Nemotron), una sola vez.

### 5.5 Ruta y memoria

- **La ruta se decide midiendo.** Con faster-whisper cada palabra lleva su tiempo, y la asignación de hablantes es la mejor medida: 1,2–1,8 % de palabras mal asignadas. whisper.cpp con GPU va unas diez veces más rápido, pero asigna segmentos enteros y duplica ese error: 2,3–2,5 %. La prueba de empaquetado mide las dos rutas en un MacBook Air M2 de 8 GB y en un Ryzen 9 5900X con GTX 980, y de ahí sale la tabla de rutas de la app.
- **faster-whisper con CUDA queda fuera:** sus librerías de NVIDIA añadirían más de 1 GB al instalador.
- **Memoria.** Con 8 GB, el modelo de transcripción (pico medido de 3,6 GB) y Nemotron (1,7 GB) no caben juntos con holgura. El motor libera el primero antes de separar hablantes y lo recarga con el siguiente archivo; con más memoria, lo mantiene. El umbral sale de la prueba de empaquetado.
- **Retomar.** Los audios de más de 20 minutos se procesan por trozos con puntos de control: si la app se cierra a mitad, al abrir el mismo archivo sigue donde iba.

## 6. El protocolo

### 6.1 Formato

JSON-RPC 2.0, un mensaje por línea, en UTF-8. En C#, StreamJsonRpc (de Microsoft) con `NewLineDelimitedMessageHandler`; en Swift, un cliente propio sobre `FileHandle`. Son pesadas `models.ensure` y `transcribe`, y el motor atiende una pesada a la vez; las demás peticiones se responden siempre.

### 6.2 Métodos

| Método | Parámetros | Devuelve |
|---|---|---|
| `hello` | — | Versión del protocolo, del motor y de la librería |
| `probe` | — | `Machine` y `Route` del sondeo, con su motivo |
| `models.status` | — | Por cada modelo necesario: si está y cuánto pesa |
| `models.ensure` | — | Descarga lo que falte, con progreso |
| `transcribe` | `path`, `language` (`auto`), `diarize` (`true`) | `result_id` y el mismo JSON que escribe `-f json` (`docs/api.md`) |
| `export` | `result_id`, `format` (`txt`, `srt` o `vtt`), `path` | La ruta escrita, con los escritores de la librería |
| `cancel` | — | La petición pesada en curso termina con `cancelled` |

Notificaciones del motor:

- `plan`: duración, ruta, tiempo estimado y si la estimación sale de una medición o de la tabla.
- `progress`: los campos de `Progress` del núcleo (etapa, hecho, total, detalle).
- `partial`: inicio, fin y texto de cada fragmento ya transcrito, sin hablantes. Con trozos en paralelo llegan intercalados; la app muestra el último como vista previa, que es una señal de vida y no la transcripción.

### 6.3 Errores

Un error JSON-RPC con `data.code` y `data.recoverable`:

- Los códigos de `AsrError`: `unsupported_option`, `insufficient_resources`, `out_of_memory`, `backend_failed`, `cancelled`, `diarize_unavailable`, `diarize_failed`.
- Los del motor: `audio_unreadable` (el archivo no se puede decodificar), `busy` (ya hay una petición pesada en curso) e `internal` (cualquier otro fallo, con la traza en el log).

Las apps escriben sus propios mensajes, traducidos, a partir del código; el texto en inglés del motor va a "detalles".

### 6.4 Cambios en el núcleo (este repo)

Hoy el progreso y la cancelación solo se consultan entre trozos de 10 minutos, y un audio de menos de 20 minutos es un solo trozo: con uno de 15 minutos la barra salta de 0 a 100, y cancelar puede tardar muchos minutos. Un PR aquí, con TDD, añade al backend un gancho por fragmento que sirve para tres cosas:

- el progreso dentro de cada trozo, en segundos de audio transcritos;
- la cancelación, consultada en cada fragmento;
- el texto del fragmento, para `partial`.

`Progress` es contrato público: el cambio se documenta en `docs/api.md` y en el CHANGELOG. Si aun así el motor no se detiene en unos segundos, la app lo reinicia; en audios largos, los trozos terminados se conservan.

### 6.5 Versiones

`protocol/` guarda el esquema de cada mensaje en JSON Schema, y los tipos de Swift y de C# se generan de él. Añadir campos o métodos opcionales no cambia la versión; romper algo existente la sube. App y motor viajan en el mismo instalador, así que la comprobación de `hello` solo protege del desarrollo con piezas de versiones distintas.

## 7. Las apps

### 7.1 La ventana y el camino básico

Lista de grabaciones a la izquierda (en cola, en curso, listas) y contenido a la derecha. Cuatro pantallas (maqueta de estructura, con un estilo provisional que §7.3 sustituye: [`flujo-base.png`](2026-09-24-app-escritorio-cimientos/flujo-base.png)):

1. **Primera vez.** Dice qué hace la app y cuánto pesan los modelos (3,5 GB) antes de descargarlos, y empieza con un clic. No pide cuenta, correo ni token. Mientras descarga ya acepta archivos, que esperan en la cola.
2. **En curso.** Soltar es empezar: no hay paso de confirmación. Muestra los pasos (preparar el audio, transcribir, separar hablantes), el tiempo restante calculado con los minutos ya transcritos y marcado como estimado, el idioma detectado con "Cambiar", la vista previa del texto y "Cancelar".
3. **Resultado.** Cada hablante con su color, tiempos en monoespaciada, "Copiar" (todo, con hablantes) y "Exportar" (txt, srt o vtt, con el diálogo de guardar del sistema). Volver a abrirlo es instantáneo.
4. **Cuando algo falla.** Nunca se cambia el modelo en silencio. Con poca memoria: "Este equipo tiene 4 GB de memoria, y la transcripción precisa necesita 6 GB", con el borrador como alternativa y su precio. Cada código de error tiene su mensaje para personas y "Copiar diagnóstico".

Además: la cola procesa en orden de llegada; cerrar con un trabajo en curso pide confirmación; los controles son nativos, con etiquetas para VoiceOver y el Narrador, y con los atajos de teclado de cada sistema.

### 7.2 Textos

Un solo archivo de textos en `design/`, en español e inglés, genera el String Catalog de Xcode y los `.resw` de WinUI: las dos apps nunca dicen cosas distintas. La interfaz sigue el idioma del sistema.

### 7.3 Dirección visual: "Swiss con calidez"

Del Swiss se queda la disciplina: un solo acento, jerarquía clara, cifras en monoespaciada. De la referencia (eko.lumaris.works) entra el aire de 2026: atmósfera de color, vidrio, esquinas amplias, píldoras y una serif de titular.

- **Atmósfera:** un degradado difuminado. Melocotón `#f3d0b5`, rosa `#dcb7cb` y lavanda `#a892d2` en claro; `#1d1621`, `#2b2034` y `#3b2c52` en oscuro. Se ve de lleno en la bienvenida y a través del vidrio de la lista; la transcripción va sobre una superficie casi opaca, para leer mucho rato sin cansarse.
- **Vidrio:** Liquid Glass en macOS 26 y los materiales anteriores en macOS 14 y 15; Mica y Acrylic en Windows 11, y superficies sólidas con los mismos tokens en Windows 10.
- **Formas:** esquinas de 14–16 px en paneles y de 10 px en elementos de la lista; botones en píldora.
- **Tipografía:** Instrument Serif para títulos, Inter para la interfaz, JetBrains Mono para tiempos y cifras. Las tres con licencia OFL, incluidas en la app.
- **Tokens, con el contraste medido** sobre el peor fondo que deja ver el vidrio (mínimo exigido para texto: 4,5:1):

| | Claro | Oscuro |
|---|---|---|
| Texto | `#18181b` · 11,8:1 | `#f5eff3` · 14,3:1 |
| Secundario | `#5e5058` · 5,0:1 | `#ab9da8` · 6,3:1 |
| Acento | `#8a0f1c` · 6,4:1 | `#ee8797` · 6,6:1 |
| Vidrio de lectura | blanco al 82 % | `#1e1822` al 80 % |
| Vidrio de la lista | blanco al 55 % | `#140f18` al 55 % |
| Botón principal | fondo `#18181b`, texto blanco · 17,7:1 | fondo `#f5eff3`, texto `#18181b` · 15,6:1 |

Los tokens viven en `design/`, junto a los textos, y de ahí se generan los colores y tipografías de Swift y de XAML. La app sigue la apariencia del sistema.

Maqueta aprobada: [`direccion-2026.png`](2026-09-24-app-escritorio-cimientos/direccion-2026.png). Sus cifras de contraste son cotas inferiores, medidas con el vidrio de la lista al 45 %; las de la tabla usan los valores finales.

## 8. Distribución

- **Mac:** `.dmg` notarizada, desde la web y GitHub Releases. Actualizaciones con Sparkle 2 (firma EdDSA), que en el segundo arranque pregunta si puede buscarlas sola. Exige el Apple Developer Program (99 al año). La Mac App Store queda para después, si el motor funciona dentro del sandbox.
- **Windows:** Microsoft Store, con MSIX. El registro es gratis para particulares desde septiembre de 2025; la Store firma el paquete (sin aviso de SmartScreen), lo actualiza y se instala con "Obtener". La política 10.2.2 permite descargar lo que corresponde a la función descrita: los modelos, y un acelerador de GPU si se describe.
- **Publicación automática:** al crear un tag, GitHub Actions construye el motor y la app en cada sistema. En Mac firma, notariza y sube la `.dmg` y el appcast de Sparkle a GitHub Releases; en Windows empaqueta el MSIX y lo envía a la Store. Las dos apps salen con el mismo número de versión, y "Acerca de" muestra la versión del motor, de la librería y del protocolo.
- **Web:** una página en GitHub Pages, con la misma estética y dos botones: "Descargar para Mac" y "Obtener en Microsoft Store".
- **Privacidad:** sin telemetría, sin analítica y sin envío automático de errores. La app solo se conecta para descargar modelos y buscar actualizaciones. Si una versión trae otro modelo, lo anuncia con su tamaño antes de descargarlo.
- **Coste:** 99 al año, la cuenta de Apple. La Store, GitHub y los modelos cuestan cero.
- **Cuenta:** particular. El nombre del autor aparece como editor en la Store y en el certificado de Apple; publicar con una marca exigiría cuenta de empresa en las dos tiendas.

## 9. Pruebas

- **Núcleo (este repo):** los cambios de §6.4 con TDD, en el CI de siempre: tres sistemas, sin red y sin modelos.
- **Motor:** el canal sobrevive a escrituras en stdout desde Python y desde C; cada error llega con su código; cancelar tarda segundos; con 8 GB libera el modelo antes de separar hablantes.
- **Protocolo:** escenarios contra el motor real con un backend falso (progreso en orden, cancelación, cada error), con cada mensaje validado contra el esquema. Sus respuestas grabadas son el motor falso de las dos apps: si el motor cambia, los falsos cambian con él.
- **Apps:** una lista única de escenarios de producto (primera vez, soltar, cola, cancelar, error de memoria, exportar) que las dos apps tienen que pasar, y capturas de las pantallas clave en claro y oscuro, en español e inglés.
- **Accesibilidad:** una prueba recalcula el contraste de cada par de tokens sobre el peor fondo y falla por debajo de 4,5:1. Cada control lleva su etiqueta de accesibilidad.
- **De punta a punta, antes de cada versión:** el instalador real en una máquina limpia, con los modelos reales, sobre una grabación pública; se miden el tiempo, la memoria pico y el resultado.

## 10. Orden de trabajo

0. **Prueba de empaquetado**, antes de cualquier pantalla y con código desechable: empaquetar el motor con los dos candidatos, arrancarlo en un Mac y un Windows limpios, notarizarlo y generar el MSIX; medir tamaño, arranque, las dos rutas y la memoria pico en el M2 y en el 5900X. Sale una respuesta escrita y la tabla de rutas.
1. **Núcleo:** §6.4 en este repo. PR y tag.
2. **Motor y protocolo**, en el repo nuevo: el esquema, los escenarios y el motor falso.
3. **Las dos apps a la vez**, primero sobre el motor falso y después sobre el real: primera vez, cola, progreso, resultado, errores, exportar, idiomas y los dos temas.
4. **Distribución:** publicación automática, firma, Sparkle, la Store y la web.
5. **Medir y publicar:** Nemotron con reuniones AMI de tres o más personas, y la prueba de punta a punta en máquinas limpias. Después, la v1.

**Antes de empezar, por parte del autor:**

- Abrir la cuenta de desarrollador de Apple: la verificación puede tardar un par de días, y el paso 0 la necesita para notarizar.
- Abrir la cuenta de la Microsoft Store: gratis, con documento de identidad y selfie.
- Decidir el nombre de la app, en una sesión aparte, antes del paso 2.

## 11. Fuera de alcance

- Las partes 2, 3 y 4 (§2).
- Interfaz en Linux, Windows ARM64, Mac con Intel y Mac App Store.
- faster-whisper con CUDA.
- Exportar a Word (parte 2).

## 12. Riesgos declarados

- **Empaquetar torch y notarizarlo** es la pieza con más riesgo; por eso va primero. Si librosa arrastra numba, su JIT exige entitlements en macOS que hay que probar.
- **Dos interfaces que se separan con el tiempo.** Lo contienen el protocolo con esquema, los textos y tokens compartidos, la lista única de escenarios y las capturas comparadas.
- **Nemotron en reuniones de tres o más personas** puede quedarse corto: 30,1 % de DER en AMI sin postproceso. Si la medición lo confirma, se añade al núcleo un puente de pausas cortas, validado con reuniones que no se usaron para ajustarlo.
- **Velocidad en Apple Silicon por CPU:** faster-whisper no usa la GPU del Mac. Si la tabla de rutas no convence, whisper.cpp con Metal cambia velocidad por precisión de hablantes.
- **La revisión de la Store tarda días por versión,** y eso marca el ritmo de publicación en Windows.
- **Instrument Serif tiene un solo peso** (normal y cursiva): la jerarquía de títulos se hace con el tamaño.
- **Windows 10 no tiene Mica:** superficies sólidas con los mismos tokens.
