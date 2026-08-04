"""Transcripción de audio a texto 100% local con faster-whisper.

Sin claves de API, sin subir audio a la nube. Solo necesita ffmpeg en el PATH
(en Linux/macOS: paquete `ffmpeg`; en Windows: https://ffmpeg.org/download.html).

Uso rápido:
    speechtotext transcribe src/static/audio.wav
    speechtotext transcribe charla.mp3 --model medium --language auto --formats txt,srt
"""
from __future__ import annotations

import logging
import os
import sys

# En Windows sin Developer Mode la caché de Hugging Face intenta crear symlinks y
# revienta con WinError 1314; y el downloader xet se cuelga EN SILENCIO con
# archivos grandes (los chicos bajan por HTTP y engañan). Estas dos flags fuerzan
# copia + HTTP plano. huggingface_hub congela estas env vars en constantes AL
# IMPORTARSE (constants.py:275,339), así que hay que setearlas ANTES de importar
# faster_whisper (que lo arrastra). setdefault: quien ya lo configuró manda.
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from pathlib import Path
from typing import Optional

import typer
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from speechtotext.core.engines import (
    ENGINE_FASTER,
    ENGINE_WHISPERCPP,
    ENGINES,
    make_engine,
    quant_for,
)
from speechtotext.core.formats import (
    find_gaps,
    parse_formats,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)
from speechtotext.core.postprocess import normalize_hours
from speechtotext.core.segments import LabeledSegment

# En Windows la consola suele ser cp1252 y rich escribe glifos Unicode (spinner
# Braille, etc.) que revientan al codificar. Forzamos UTF-8 en los streams.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# faster-whisper ya calcula y emite "VAD filter removed X of audio" (una vez por trozo en
# ruta troceada), pero nadie configura logging en el paquete y ese diagnóstico se pierde:
# es el único instrumento que distingue un VAD ciego de uno saturado. Root en WARNING para
# que sólo suba lo que pedimos explícitamente y rich no quede sepultado.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.INFO)

app = typer.Typer(add_completion=False, help="Transcripción de audio offline con Whisper.")
console = Console()


def _resolve_output_base(audio: Path, output: Optional[Path]) -> Path:
    if output is None:
        return audio.with_suffix("")
    if output.exists() and output.is_dir():
        return output / audio.stem
    if str(output).endswith(("/", "\\")):
        output.mkdir(parents=True, exist_ok=True)
        return output / audio.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    return output if output.suffix == "" else output.with_suffix("")


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_file(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _load_hotwords_file(path: Path) -> Optional[str]:
    """Lee un léxico de un archivo: un término por línea o separados por coma."""
    # utf-8-sig tolera el BOM que dejan algunos editores de Windows.
    text = path.read_text(encoding="utf-8-sig")
    words = [w.strip() for w in text.replace("\n", ",").split(",")]
    return ", ".join(w for w in words if w) or None


def _resolve_hotwords(hotwords: Optional[str], hotwords_file: Optional[Path]) -> Optional[str]:
    """Combina --hotwords (inline) y --hotwords-file. Scoped por invocación, sin default global:
    los términos entran al prompt después de tokenizer.sot_prev, o sea como texto previo de la
    conversación (faster_whisper/transcribe.py:1542-1548), no como prior sobre el vocabulario.
    Un léxico global envenenaría todo otro audio: el modelo lo continuaría como si fuera la
    conversación en curso, y una lista larga degrada la corrida entera (ablación en §4 del
    plan de calidad 2)."""
    parts = []
    if hotwords_file is not None:
        from_file = _load_hotwords_file(hotwords_file)
        if from_file:
            parts.append(from_file)
    if hotwords and hotwords.strip():
        parts.append(hotwords.strip())
    return ", ".join(parts) or None


def _transcribe_opts(lang: Optional[str], beam_size: int, vad: bool,
                     hotwords: Optional[str], word_timestamps: bool = False) -> dict:
    """Opciones de decodificación de Whisper. condition_on_previous_text=False corta el
    arrastre de contexto entre ventanas: sin esto el modelo alucina hacia frases comunes
    (Fase 0, docs/benchmark-turboscribe.md). word_timestamps sólo se pide al diarizar:
    la asignación palabra→hablante parte los segmentos en el cambio de voz. Aquí aterrizan
    futuros knobs de tuning."""
    return {
        "language": lang,
        "beam_size": beam_size,
        "vad_filter": vad,
        "hotwords": hotwords,
        "condition_on_previous_text": False,
        "word_timestamps": word_timestamps,
    }


def transcribe_file(
    audio: Path,
    output: Optional[Path],
    language: str,
    model: str,
    formats: str,
    device: str,
    compute_type: str,
    vad: bool,
    beam_size: int,
    diarize: bool,
    speakers: Optional[int],
    identify: bool,
    threshold: float,
    hotwords: Optional[str] = None,
    chunk: Optional[bool] = None,
    jobs: int = 4,
    engine: str = ENGINE_FASTER,
) -> None:
    """Transcribe un archivo (opcionalmente con diarización) y escribe los formatos pedidos."""
    try:
        requested = parse_formats(formats)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    base = _resolve_output_base(audio, output)
    base.parent.mkdir(parents=True, exist_ok=True)

    lang = None if language.lower() == "auto" else language
    if engine not in ENGINES:
        raise typer.BadParameter(
            f"engine {engine!r} no existe; disponibles: {', '.join(ENGINES)}"
        )
    hotwords = (hotwords or "").strip() or None

    # Contrato de capacidades (CAPS, plan 2.2): se valida ANTES de construir modelo
    # alguno y antes del import de chunked. Jamás silencio, jamás sustitución callada.
    if engine == ENGINE_WHISPERCPP:
        # El binario pinneado es build CUDA y corre en la GPU SIEMPRE (medido en el
        # smoke: con -d cpu el exe usa la GPU igual, el adaptador no pasa device).
        # Etiquetar cpu sería mentir en el header, la llave y el JSON: el device
        # efectivo es cuda, se declara, y el remapeo se AVISA — pisar un -d cpu
        # explícito en silencio sería la sustitución callada que el contrato prohíbe.
        # Un modo CPU real (-ng) es post-ship. typer no distingue default de
        # explícito, así que el aviso sale siempre que no pidieran cuda.
        if device != "cuda":
            console.print(
                "[yellow]whisper.cpp (build CUDA) corre en la GPU; device=cuda[/yellow]"
            )
        device = "cuda"
        from speechtotext.core.enginepin import _MODEL_ALIAS

        if model not in _MODEL_ALIAS:
            # Sin pre-validación, ensure_model revienta con RuntimeError crudo a
            # mitad de corrida; aquí el error llega antes de construir nada.
            raise typer.BadParameter(
                f"modelo {model!r} no está pinneado para whispercpp; disponibles: "
                f"{', '.join(sorted(_MODEL_ALIAS))}"
            )
        try:
            # 'auto' NO resuelve a float16 aquí: fp16 en la 980 pagina bajo WDDM
            # (0.53x tiempo real, medido). quant_for mapea o rechaza con la medición.
            compute_type = quant_for(engine, compute_type, device)
        except ValueError as e:
            raise typer.BadParameter(str(e))
        if hotwords:
            # Rechazo, no degradación: avisar "degradado" sobre un knob inerte
            # fabricaría un efecto que no ocurrió. Ni modelo ni caché se tocan.
            console.print(
                "--hotwords no tiene efecto con whisper.cpp (--prompt es inerte con "
                "-mc 0, medido 2026-07-27); usa --engine faster-whisper"
            )
            raise typer.Exit(2)
        if vad:
            console.print("[yellow]whisper.cpp no trae VAD; se transcribe sin filtro[/yellow]")
            # vad efectivo = False: entra así a opts (misma llave que --no-vad) y
            # apaga el consejo "prueba --no-vad" del resumen (el motor no tiene VAD).
            vad = False
        if diarize:
            console.print(
                "[yellow]atribución por segmento (gruesa), sin cortes intra-segmento; "
                r"la marca \[?] queda activa[/yellow]"
            )
        if jobs != 1:
            # 4 subprocesos × 1.28 GB contra 4096 MiB: WDDM no revienta, pagina 25x
            # en silencio (medido). El 19.3x de un solo proceso hace innecesario más.
            # Sin condición de device: bajo whispercpp el device efectivo es cuda siempre.
            console.print("[yellow]la GPU no paraleliza; jobs=1[/yellow]")
            jobs = 1
    elif compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    if hotwords:
        n_terms = len([t for t in hotwords.split(",") if t.strip()])
        # ponytail: se cuentan términos y caracteres, no tokens. El conteo exacto necesita
        # el tokenizer del modelo, que en este punto todavía no está cargado; 223 tokens
        # son del orden de 600-700 caracteres en español. Techo: si alguien necesita el
        # número exacto, se cuenta después de construir el modelo.
        console.print(
            f"Hotwords ({n_terms} términos, {len(hotwords)} caracteres): {hotwords}",
            # markup=False: los términos pueden traer corchetes/acentos que rich malinterpretaría.
            markup=False,
        )
        if n_terms >= 10 or len(hotwords) >= 300:
            # Techos conservadores, uno por daño verificado: la ablación midió degradación
            # con 25 términos (con 3 no hubo pérdida, §4 del plan) y la librería trunca
            # en silencio a los 223 tokens. Se avisa mucho antes de ambos.
            console.print(
                "[yellow]Lista larga de hotwords: 25 términos degradaron la cobertura "
                "9 puntos sobre 240 s de audio real (2026-08-03); entran como texto "
                "previo, no como léxico.[/yellow]"
            )

    console.print(
        f"[bold]Modelo[/bold] [cyan]{model}[/cyan] · "
        f"[bold]device[/bold] [cyan]{device}[/cyan] · "
        f"[bold]compute[/bold] [cyan]{compute_type}[/cyan]"
    )
    if engine != ENGINE_FASTER:
        # El motor no-default se declara en el header (G2); el default queda byte a byte.
        from speechtotext.core.enginepin import ENGINE_PIN

        console.print(
            f"[bold]Motor[/bold] [cyan]whisper.cpp {ENGINE_PIN['version']}[/cyan] · "
            f"[cyan]{model} {compute_type}[/cyan] · [cyan]{device}[/cyan]"
        )
    from speechtotext.core.chunked import run_chunked, should_chunk, probe_duration

    opts = _transcribe_opts(lang, beam_size, vad, hotwords, word_timestamps=diarize)
    try:
        if should_chunk(probe_duration(audio), chunk):
            console.print(f"[bold]Troceado[/bold] (jobs={jobs}) · {model}")
            segments, info = run_chunked(
                audio, opts, jobs, model_name=model, device=device,
                compute_type=compute_type, log=lambda m: console.print(f"  {m}", markup=False),
                engine=engine,
            )
        elif engine == ENGINE_WHISPERCPP:
            # whisper-cli es main(char**): el path del usuario jamás viaja en argv.
            # Se transcodifica a wav 16k mono temporal (ASCII) como hace la ruta troceada.
            from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav

            dur = probe_duration(audio)
            try:
                wav = transcode_to_wav(audio.read_bytes())
            except (FfmpegMissingError, TranscodeError) as e:
                console.print(f"[red]No se pudo procesar el audio:[/red] {e}")
                raise typer.Exit(1)
            try:
                whisper = make_engine(engine, model, device, compute_type)
                with Progress(
                    SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(), console=console,
                ) as progress:
                    task = progress.add_task(f"Transcribiendo {audio.name}", total=None)
                    segments_iter, info = whisper.transcribe(str(wav), _duration_s=dur, **opts)
                    segments = list(segments_iter)
                    progress.update(task, completed=1)
            finally:
                wav.unlink(missing_ok=True)
            # Ley de orquestación: los segmentos pertenecen al motor; info.duration la
            # fabrica el orquestador (el adaptador no la emite, y :206/:231 la consumen).
            info.duration = dur
        else:
            whisper = WhisperModel(model, device=device, compute_type=compute_type)
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(), console=console,
            ) as progress:
                task = progress.add_task(f"Transcribiendo {audio.name}", total=None)
                segments_iter, info = whisper.transcribe(str(audio), **opts)
                segments = list(segments_iter)
                progress.update(task, completed=1)
    except RuntimeError as e:
        # El OOM real sube como RuntimeError del allocator ("mkl_malloc: failed to allocate
        # memory"), y este es el único punto por el que pasan los dos constructores del
        # camino de transcripción y además la decodificación. Reaccionar al fallo es más
        # barato y más exacto que sondear la RAM del sistema: nada de psutil ni ctypes.
        # ponytail: se decide por substring 'alloc', techo conocido; si algún backend
        # inventa otro texto para el OOM, se propaga crudo — que es la falla correcta.
        if "alloc" not in str(e).lower():
            raise
        console.print("[red]Se quedó sin memoria al transcribir.[/red]")
        # markup=False: el texto del allocator trae corchetes que rich intentaría parsear.
        console.print(f"  {e}", markup=False)
        if engine == ENGINE_WHISPERCPP:
            # El stderr de CUDA OOM también contiene 'alloc': aconsejar RAM de CPU aquí
            # sería un falso amigo. El consejo es de VRAM (4 GB en la 980).
            console.print(
                "La VRAM de la GPU se agotó. Cierra aplicaciones que usen la GPU, "
                "prueba un modelo menor o usa [cyan]--engine faster-whisper[/cyan] (CPU)."
            )
        else:
            console.print(
                "large-v3 pide ~3.5 GB sólo al cargar. Cierra procesos o usa "
                "[cyan]-m medium[/cyan]. No cubre la muerte nativa de --diarize."
            )
        raise typer.Exit(1)

    # La cobertura es la única medida que la herramienta tiene de sí misma: sin ella, `OK`
    # significa "write_text no lanzó" y una corrida que descartó media conversación se ve
    # igual que una buena. Se suma ANTES de _run_diarization porque esa partición por
    # palabras alteraría el conteo. Los segmentos no se solapan y los trozos son contiguos,
    # así que la suma cruda vale. gaps igual: una sola cantidad, calculada una vez aquí y
    # pasada al JSON (5.1.3), para que consola y disco no emitan dos números distintos.
    cov = sum(s.end - s.start for s in segments)
    # Se calcula SIEMPRE aquí, aunque el probe haya fallado y la consola vaya a callar:
    # mandarle None a write_json lo hace recalcular sobre los segmentos ya diarizados, que
    # es el mismo C-13 que este ítem cierra, entrando por la puerta de atrás. Con
    # duration==0 find_gaps igual devuelve los huecos interiores (el de cola no, porque
    # duration - cursor sale negativo). Lo que depende de duration es IMPRIMIRLO.
    gaps = find_gaps(segments, info.duration)
    if info.duration:
        ratio = cov / info.duration
        voz = f"con voz {cov / 60:.1f} de {info.duration / 60:.1f} min ({100 * ratio:.0f}%)"
    else:
        # duration==0 es el probe fallido. Reportar 0% sería inventar una medida sobre un
        # denominador que no existe, y encima se contradice con el conteo de segmentos:
        # el peor caso posible sería el único que no grita.
        voz = "[yellow]cobertura desconocida (duración no medida)[/yellow]"

    # "Idioma detectado" sobre el flag del propio usuario es una fabricación en el 100% de
    # las corridas por defecto (-l es): ahí no se midió nada. La probabilidad sólo se
    # imprime cuando de verdad hubo detección y la ruta la conservó.
    if lang is not None:
        idioma = f"Idioma: [bold]{info.language}[/bold] (forzado)"
    else:
        idioma = f"Idioma detectado: [bold]{info.language}[/bold]"
        if info.language_probability is not None:
            idioma += f" (prob={info.language_probability:.2f})"

    # El motor resuelto SIEMPRE en el resumen (G2): dos corridas del mismo audio con
    # texto distinto tienen que ser distinguibles a simple vista.
    console.print(
        f"{idioma} · duración {info.duration:.1f}s · "
        f"{len(segments)} segmentos · {voz} · motor {engine}"
    )

    # La línea de huecos sale incondicional (C-11: el umbral de cobertura se borró, no se
    # recalibró): es un hecho geométrico sobre la línea de tiempo, no depende de cómo el
    # VAD remapee los timestamps. Sin huecos se dice explícitamente — callar convertiría la
    # ausencia en una omisión indistinguible de un fallo del instrumento.
    # duration falsy = probe fallido: sin denominador no hay línea de tiempo sobre la que
    # afirmar ni los huecos ni su ausencia, así que ahí no se imprime nada de ellos.
    if info.duration:
        if gaps:
            # ponytail: se listan los primeros 5 huecos y luego "y N más (ver el JSON)"; con
            # 40 huecos la consola se inunda y el artefacto completo ya existe. Techo: si
            # alguien pide los 40 en consola, sale un --gaps-all.
            lista = ", ".join(f"{_fmt(a)}-{_fmt(b)} ({b - a:.0f} s)" for a, b in gaps[:5])
            if len(gaps) > 5:
                lista += f", y {len(gaps) - 5} más (ver el JSON)"
            # El consejo sólo aplica a quien tiene el VAD puesto: sugerir --no-vad a quien ya
            # lo apagó (o a whispercpp, que no trae VAD) es ruido, y ahí la pérdida viene de
            # otro lado (el modelo no emitió).
            consejo = " — prueba --no-vad" if vad else ""
            plural = "hueco" if len(gaps) == 1 else "huecos"
            console.print(f"{len(gaps)} {plural} sin texto: {lista}{consejo}")
        else:
            # find_gaps corta en >= 5.0, no en > 5.0: decir "> 5 s" declararía menos de lo
            # que la herramienta sabe, y este plan trata justo de eso (Q9).
            console.print("sin huecos de 5 s o más")

    if diarize:
        segments = _run_diarization(audio, segments, speakers, identify, threshold)

    # Post-proceso textual (horas 8.33->8:33). Uniforma a LabeledSegment: los Segment
    # de faster_whisper son inmutables y los writers solo leen start/end/text/speaker.
    segments = [
        LabeledSegment(s.start, s.end, normalize_hours(s.text), getattr(s, "speaker", None))
        for s in segments
    ]

    # Identidad del motor en el JSON (G2): dos corridas del mismo audio con texto
    # distinto tienen que ser distinguibles también a máquina, no solo en consola.
    # 'selection' es constante en v1 (no hay auto); mantener la clave evita otro
    # cambio de payload el día que exista.
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.core.enginepin import ENGINE_PIN

        version = f"whisper.cpp {ENGINE_PIN['version']}"
    else:
        from importlib.metadata import version as _pkg_version

        version = f"faster-whisper {_pkg_version('faster-whisper')}"
    engine_info = {
        "name": engine, "version": version, "model": model,
        "quant": compute_type, "device": device, "selection": "explicit",
    }
    if diarize:
        # La atribución gruesa (sin words) y la fina son resultados distintos y el
        # consumidor del JSON merece saber cuál recibió.
        engine_info["diarization"] = "segment" if engine == ENGINE_WHISPERCPP else "word"

    writers: dict[str, tuple[str, callable]] = {
        "txt": (".txt", lambda p: write_txt(segments, p)),
        "srt": (".srt", lambda p: write_srt(segments, p)),
        "vtt": (".vtt", lambda p: write_vtt(segments, p)),
        "json": (".json", lambda p: write_json(segments, info, p, engine_info=engine_info,
                                               speech_s=round(cov, 2), gaps=gaps)),
    }
    for fmt in sorted(requested):
        suffix, write_fn = writers[fmt]
        out_path = base.with_suffix(suffix)
        write_fn(out_path)
        console.print(f"  [green]OK[/green] {out_path}")


@app.command()
def transcribe(
    audio: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False, help="Archivo de audio o vídeo."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Carpeta o ruta base de salida (por defecto: junto al audio)."
    ),
    language: str = typer.Option(
        "es",
        "--language",
        "-l",
        help="Código ISO-639-1 (es, en, fr, ...). Usa 'auto' para detección automática.",
    ),
    model: str = typer.Option(
        "small",
        "--model",
        "-m",
        help="tiny | base | small | medium | large-v3 | distil-large-v3. "
        "large-v3 = máxima calidad (puntuación y nombres propios; ~1.1x tiempo real en CPU).",
    ),
    formats: str = typer.Option(
        "txt,srt,json", "--formats", "-f", help="Formatos separados por coma (txt, srt, vtt, json)."
    ),
    device: str = typer.Option("cpu", "--device", "-d", help="cpu | cuda | auto"),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help="auto | int8 | int8_float16 | float16 | float32. 'auto' elige int8 en CPU y float16 en GPU.",
    ),
    vad: bool = typer.Option(
        True, "--vad/--no-vad", help="Filtro VAD para descartar silencios largos."
    ),
    beam_size: int = typer.Option(5, "--beam-size", min=1, help="Tamaño del beam search."),
    diarize: bool = typer.Option(
        False, "--diarize", "-D", help=r"Marcar quién habla (diarización). Requiere el extra \[diarize]."
    ),
    speakers: Optional[int] = typer.Option(
        None, "--speakers", min=1, help="Número de hablantes (pista; auto si se omite)."
    ),
    identify: bool = typer.Option(
        True, "--identify/--no-identify", help="Poner nombre a las voces registradas."
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", min=0.0, max=1.0, help="Umbral de coincidencia de voz (coseno, 0-1)."
    ),
    hotwords: Optional[str] = typer.Option(
        None,
        "--hotwords",
        help="Términos difíciles separados por coma (nombres propios, jerga): entran a cada "
        "ventana como si fueran la conversación previa, así que una lista corta ayuda y una "
        "larga degrada. Escríbelos con mayúsculas y tildes.",
    ),
    hotwords_file: Optional[Path] = typer.Option(
        None,
        "--hotwords-file",
        exists=True,
        dir_okay=False,
        help="Archivo con términos difíciles (uno por línea o separados por coma), para un "
        "léxico por proyecto. Se combina con --hotwords. Techo: por encima de 223 tokens "
        "(~600-700 caracteres) la lista se trunca en silencio "
        "(faster_whisper/transcribe.py:1546-1547).",
    ),
    chunk: Optional[bool] = typer.Option(
        None, "--chunk/--no-chunk",
        help="Trocear el audio para checkpoint/resume + paralelismo. Auto si dura > 20 min.",
    ),
    jobs: int = typer.Option(
        4, "--jobs", "-j", help="Trozos en paralelo al trocear (comparten un modelo).",
    ),
    engine: str = typer.Option(
        ENGINE_FASTER,
        "--engine",
        help="faster-whisper (CPU, default) | whispercpp (whisper.cpp CUDA en la GPU).",
    ),
) -> None:
    """Transcribe un archivo de audio localmente con Whisper (sin enviar nada a internet)."""
    transcribe_file(
        audio, output, language, model, formats, device, compute_type,
        vad, beam_size, diarize, speakers, identify, threshold,
        hotwords=_resolve_hotwords(hotwords, hotwords_file),
        chunk=chunk, jobs=jobs, engine=engine,
    )


def _run_diarization(audio, segments, speakers, identify, threshold):
    from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav
    from speechtotext.speakers import diarization, registry
    from speechtotext.speakers.identify import assign_names

    try:
        wav = transcode_to_wav(audio.read_bytes())
    except (FfmpegMissingError, TranscodeError) as e:
        console.print(f"[red]No se pudo procesar el audio:[/red] {e}")
        raise typer.Exit(1)
    try:
        turns, clusters = diarization.diarize(str(wav), num_speakers=speakers)
    except ImportError:
        console.print(r'[red]Falta el extra de diarización:[/red] pip install -e ".\[diarize]"')
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]La diarización falló:[/red] {e}")
        console.print(
            "Revisa que aceptaste los términos de los modelos pyannote y que HF_TOKEN esté configurado."
        )
        raise typer.Exit(1)
    finally:
        wav.unlink(missing_ok=True)

    labeled = diarization.assign_segments(segments, turns)
    name_map: dict[str, str] = {}
    if identify:
        enrolled = registry.get_embeddings()
        if enrolled:
            name_map = assign_names(clusters, enrolled, threshold)
    return diarization.apply_names(labeled, name_map)


def _extract_region(audio, regions, region, output, language, model, formats,
                    diarize, speakers, identify, threshold, context, hotwords=None):
    import subprocess

    from speechtotext.core.finder import clip_window

    if region < 1 or region > len(regions):
        console.print(f"[red]Región {region} fuera de rango (hay {len(regions)}).[/red]")
        raise typer.Exit(1)

    r = regions[region - 1]
    begin, duration = clip_window(r.start, r.end, context)
    base_dir = output if output is not None else audio.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{audio.stem}_{_fmt_file(r.start)}-{_fmt_file(r.end)}"
    clip = base_dir / f"{stem}.wav"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(begin), "-t", str(duration), "-i", str(audio),
        "-ar", "16000", "-ac", "1", str(clip),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        console.print(
            "[red]ffmpeg no está en el PATH.[/red] "
            "Instálalo: [cyan]winget install Gyan.FFmpeg[/cyan]"
        )
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]No se pudo recortar el audio:[/red] {e.stderr.decode(errors='ignore')[:200]}")
        raise typer.Exit(1)

    console.print(f"  [green]Recorte[/green] {clip} ({_fmt(r.start)}–{_fmt(r.end)})")
    transcribe_file(
        clip, base_dir, language, model, formats,
        "cpu", "auto", True, 5, diarize, speakers, identify, threshold,
        hotwords=hotwords,
    )


@app.command()
def find(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio o vídeo a buscar."),
    query: str = typer.Argument(..., help="Palabras a buscar."),
    extract: bool = typer.Option(False, "--extract", "-e", help="Recortar + transcribir la región."),
    region: int = typer.Option(1, "--region", help="Qué región extraer (1 = la más densa)."),
    model: str = typer.Option("small", "--model", "-m", help="Modelo para la transcripción en calidad."),
    scan_model: str = typer.Option("tiny", "--scan-model", help="Modelo del índice."),
    language: str = typer.Option("es", "--language", "-l", help="Idioma de la transcripción del tramo."),
    formats: str = typer.Option("txt,srt", "--formats", "-f", help="Formatos de salida del tramo."),
    diarize: bool = typer.Option(False, "--diarize", "-D", help="Diarizar el tramo extraído."),
    speakers: Optional[int] = typer.Option(None, "--speakers", min=1, help="Nº de hablantes (pista)."),
    identify: bool = typer.Option(True, "--identify/--no-identify", help="Nombrar voces registradas."),
    threshold: float = typer.Option(0.5, "--threshold", min=0.0, max=1.0, help="Umbral de coincidencia de voz."),
    context: float = typer.Option(10.0, "--context", help="Segundos de margen al recortar."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Carpeta de salida del tramo."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Forzar reconstrucción del índice."),
    top: int = typer.Option(5, "--top", help="Cuántas regiones listar."),
    hotwords: Optional[str] = typer.Option(
        None, "--hotwords", help="Términos difíciles para la transcripción del tramo (ver transcribe)."
    ),
    hotwords_file: Optional[Path] = typer.Option(
        None, "--hotwords-file", exists=True, dir_okay=False,
        help="Archivo de léxico para la transcripción del tramo (ver transcribe).",
    ),
) -> None:
    """Busca contenido en un audio largo; con --extract recorta y transcribe el tramo."""
    from speechtotext.core import finder

    segments, cached = finder.load_or_build_index(audio, scan_model, rebuild)
    console.print(f"Índice: {'caché' if cached else 'construido'} ({scan_model}, {len(segments)} segmentos)")

    regions = finder.search(segments, query, top=top)
    if not regions:
        console.print(f'No se encontró "{query}" en el audio.', markup=False)
        raise typer.Exit(0)

    console.print(f"{len(regions)} regiones para \"{query}\":", markup=False)
    for i, r in enumerate(regions, start=1):
        console.print(f"  {i}.  {_fmt(r.start)} – {_fmt(r.end)}  ({r.hits})  \"{r.snippet}\"", markup=False)

    if extract:
        _extract_region(
            audio, regions, region, output, language, model, formats,
            diarize, speakers, identify, threshold, context,
            hotwords=_resolve_hotwords(hotwords, hotwords_file),
        )


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Nombre de la persona."),
    sample: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio de muestra de su voz."),
) -> None:
    """Registra la voz de una persona desde una muestra de audio (>=10s recomendado)."""
    import contextlib
    import wave

    from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav
    from speechtotext.speakers import diarization, registry

    try:
        wav = transcode_to_wav(sample.read_bytes())
    except (FfmpegMissingError, TranscodeError) as e:
        console.print(f"[red]No se pudo procesar el audio:[/red] {e}")
        raise typer.Exit(1)
    try:
        with contextlib.closing(wave.open(str(wav))) as w:
            seconds = w.getnframes() / float(w.getframerate())
        if seconds < 10:
            console.print(
                f"[yellow]Aviso:[/yellow] muestra corta ({seconds:.0f}s); >=10s es más fiable."
            )
        try:
            vec = diarization.embed_voice(str(wav))
        except ImportError:
            console.print(r'[red]Falta el extra de diarización:[/red] pip install -e ".\[diarize]"')
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]No se pudo registrar la voz:[/red] {e}")
            raise typer.Exit(1)
    finally:
        wav.unlink(missing_ok=True)

    registry.enroll(name, vec, seconds=seconds, model="pyannote/speaker-diarization-community-1")
    console.print(f"  [green]OK[/green] voz de {name} registrada.")


@app.command()
def voices() -> None:
    """Lista las voces registradas."""
    from speechtotext.speakers import registry

    vs = registry.list_voices()
    if not vs:
        console.print("Sin voces registradas. Usa: speechtotext enroll <nombre> <muestra.wav>")
        return
    table = Table("Nombre", "Segundos", "Registrada")
    for v in vs:
        table.add_row(v["name"], str(v.get("seconds", "")), v.get("enrolled_at", ""))
    console.print(table)


@app.command()
def forget(name: str = typer.Argument(..., help="Nombre de la voz a borrar.")) -> None:
    """Borra una voz registrada."""
    from speechtotext.speakers import registry

    if registry.remove(name):
        console.print(f"  [green]OK[/green] {name} borrada.")
    else:
        console.print(f"[red]No existe una voz llamada {name}.[/red]")
        raise typer.Exit(1)


def _trim_wav(wav: Path, seconds: float) -> Path:
    """Recorta el wav (ya 16k mono PCM) a los primeros `seconds` con ffmpeg -t.

    Medir las 7 configs sobre el audio completo sería eterno; el recorte acota el
    coste sin cambiar lo que se compara (todas miden el MISMO trozo).
    """
    import subprocess

    out = wav.with_name(wav.stem + "_bench.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(wav), "-t", str(seconds), "-c", "copy", str(out)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # Sin esto el usuario ve un CalledProcessError crudo; con esto, el mismo
        # patron de mensaje rojo que ya usa el resto del comando.
        stderr = (e.stderr or b"").decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg no pudo recortar el audio: {stderr[-300:]}") from e
    return out


def _print_bench(table: dict) -> None:
    """Tabla rich del bench + resumen. Misma vista al medir y con --show."""
    from rich.markup import escape

    def cell(v, fmt="{:.2f}"):
        if v is None:
            return "—"
        return fmt.format(v) if isinstance(v, float) else str(v)

    t = Table("Motor", "Modelo", "x_rt", "load_s", "RAM MB", "VRAM MB", "Caps", "WER", "Error")
    for r in table["results"]:
        caps = r.get("capabilities") or {}
        # H/W/S/V = hotwords/word_timestamps/native_signals/vad, compacto para caber.
        letras = "".join(
            l for l, k in (("H", "hotwords"), ("W", "word_timestamps"),
                           ("S", "native_signals"), ("V", "vad"))
            if caps.get(k)
        ) or "—"
        err = r.get("error")
        t.add_row(
            r["engine"], r["model"],
            cell(r.get("x_realtime")), cell(r.get("load_s")),
            cell(r.get("peak_ram_mb"), "{:.0f}"), cell(r.get("peak_vram_mb"), "{:.0f}"),
            letras, cell(r.get("wer_ref"), "{:.3f}"),
            # Las configs rotas se MARCAN, no se ocultan: la tabla no miente.
            # escape: el error puede traer corchetes que rich malinterpretaría.
            f"[red]{escape(err[:30])}[/red]" if err else "",
        )
    # ponytail: Console(width=120) fijo para que 9 columnas no se plieguen a 80;
    # si algún día molesta en terminales angostas, medir el ancho real.
    Console(width=120).print(t)

    ok = sum(1 for r in table["results"] if not r.get("error"))
    con_error = len(table["results"]) - ok
    skipped = table.get("skipped", [])
    console.print(f"{ok} viables · {con_error} con error · {len(skipped)} saltadas")
    for s in skipped:
        console.print(f"  saltada {s['engine']} {s['model']}: {s['reason']}", markup=False)

    recs = table.get("recommendations") or []
    if recs:
        r_t = Table("Caso de uso", "Config", "Motivo", title="¿Qué config para qué?")
        for rec in recs:
            e = rec.get("eleccion")
            config = f"{e['engine']} {e['model']}" if e else "— sin candidata —"
            r_t.add_row(rec["caso"], config, rec["motivo"])
        Console(width=120).print(r_t)


@app.command()
def bench(
    audio: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False,
        help="Audio a medir. Omítelo (o usa --show) para ver la última tabla.",
    ),
    seconds: float = typer.Option(
        60.0, "--seconds",
        help="Segundos del audio a medir (recorte inicial; acota el coste del bench).",
    ),
    quick: bool = typer.Option(
        False, "--quick",
        help="Salta medium y large-v3 de faster-whisper: son los lentos en CPU "
        "(minutos por config) y el resto basta para una primera decisión.",
    ),
    show: bool = typer.Option(
        False, "--show", help="Pinta la tabla guardada sin volver a medir."
    ),
) -> None:
    """Mide las configs ASR viables en ESTA máquina y guarda bench.json."""
    from speechtotext.core import benchmark

    if show or audio is None:
        table = benchmark.read_table()
        if table is None:
            console.print(
                "[red]No hay tabla de benchmark.[/red] "
                "Mídela con: [cyan]speechtotext bench <audio>[/cyan]"
            )
            raise typer.Exit(1)
        _print_bench(table)
        return

    from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav
    from speechtotext.core.chunked import probe_duration

    # Misma ruta que whispercpp directo: wav 16k mono temporal, el path del usuario
    # jamás viaja crudo a los motores.
    try:
        wav = transcode_to_wav(audio.read_bytes())
    except (FfmpegMissingError, TranscodeError) as e:
        console.print(f"[red]No se pudo procesar el audio:[/red] {e}")
        raise typer.Exit(1)
    try:
        clip = _trim_wav(wav, seconds)
    except RuntimeError as e:
        console.print(f"[red]No se pudo recortar el audio:[/red] {e}")
        raise typer.Exit(1)
    finally:
        wav.unlink(missing_ok=True)
    try:
        # duración REAL del recorte (el audio puede durar menos que --seconds).
        duration_s = probe_duration(clip)
        configs, _skipped = benchmark.available_configs()
        quick_saltadas = []
        if quick:
            lentas = [
                c for c in configs
                if c["engine"] == "faster-whisper" and c["model"] in ("medium", "large-v3")
            ]
            configs = [c for c in configs if c not in lentas]
            # Las quick-saltadas van a skipped: una tabla con filas ausentes sin razón
            # haría que aurelius eligiera sin saber que faltan candidatas.
            quick_saltadas = [
                {"engine": c["engine"], "model": c["model"], "reason": "saltada por --quick"}
                for c in lentas
            ]
        console.print(f"Midiendo {len(configs)} configs sobre {duration_s:.1f}s de audio...")

        def _progress(cfg, res):
            # Una fila al terminar cada config: el bench tarda minutos y el silencio
            # se confunde con un cuelgue. markup=False: el error trae corchetes.
            if res.get("error"):
                console.print(
                    f"  FALLO {cfg['engine']} {cfg['model']}: {res['error'][:120]}",
                    markup=False,
                )
            else:
                console.print(
                    f"  OK {cfg['engine']} {cfg['model']}: {res['x_realtime']}x "
                    f"(load {res['load_s']}s, transcribe {res['transcribe_s']}s)",
                    markup=False,
                )

        table = benchmark.run_benchmark(clip, duration_s, configs, progress=_progress)
        table["skipped"].extend(quick_saltadas)
        path = benchmark.write_table(table)
    finally:
        try:
            clip.unlink(missing_ok=True)
        except OSError:
            # ponytail: Windows puede retener el handle del wav unos ms tras morir el
            # ultimo hijo (WinError 32); un temporal huerfano no justifica tumbar un
            # bench de 15 minutos YA escrito en disco.
            pass
    console.print(f"Tabla escrita en {path}")
    _print_bench(table)


if __name__ == "__main__":
    app()
