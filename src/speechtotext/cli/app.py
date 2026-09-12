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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from types import SimpleNamespace

from speechtotext.asr.base import AsrError
from speechtotext.audio.io import AudioDecodeError
from speechtotext.core import probe as core_probe
from speechtotext.core.transcribe import (
    ENGINE_FASTER,
    ENGINE_WHISPERCPP,
    transcribe as core_transcribe,
)
from speechtotext.core.formats import (
    parse_formats,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)

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
    engine: str = "auto",
) -> None:
    """Transcribe un archivo (opcionalmente con diarización) y escribe los formatos pedidos.
    Todo el trabajo lo hace core.transcribe; aquí se parsean flags, se pinta y se escribe."""
    try:
        requested = parse_formats(formats)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    base = _resolve_output_base(audio, output)
    base.parent.mkdir(parents=True, exist_ok=True)
    language = "auto" if language.lower() == "auto" else language
    hotwords = (hotwords or "").strip() or None

    try:
        maquina = core_probe.machine()
        route = core_probe.choose_route(maquina, model, engine=engine, device=device,
                                        compute_type=compute_type)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    except AsrError as e:
        # insufficient_resources: el sondeo no cambia el modelo por su cuenta; lo dice y para.
        console.print(str(e), style="red", markup=False)
        raise typer.Exit(1)
    if route.reason:
        console.print(f"[yellow]{route.reason}[/yellow]")
    if route.engine == ENGINE_WHISPERCPP and maquina.whispercpp is None and maquina.platform == "win32":
        from speechtotext.core.enginepin import ENGINE_PIN

        # 650 MB por urllib sin barra: que al menos se anuncie (spec §5.3, "nunca en silencio").
        console.print(
            f"[yellow]whisper.cpp {ENGINE_PIN['version']} no está instalado: se descarga ahora "
            f"(~{ENGINE_PIN['zip_bytes'] / 1024 ** 2:.0f} MB, una sola vez)[/yellow]"
        )
    if route.engine == ENGINE_WHISPERCPP and jobs != 1:
        # 4 subprocesos × 1.28 GB contra 4096 MiB: WDDM no revienta, pagina 25x en silencio
        # (medido). El 19.3x de un solo proceso hace innecesario más.
        console.print("[yellow]la GPU no paraleliza; jobs=1[/yellow]")
        jobs = 1
    if hotwords:
        n_terms = len([t for t in hotwords.split(",") if t.strip()])
        # ponytail: se cuentan términos y caracteres, no tokens. El conteo exacto necesita
        # el tokenizer del modelo, que en este punto todavía no está cargado; 223 tokens
        # son del orden de 600-700 caracteres en español.
        console.print(
            f"Hotwords ({n_terms} términos, {len(hotwords)} caracteres): {hotwords}",
            markup=False,
        )
        if n_terms >= 10 or len(hotwords) >= 300:
            console.print(
                "[yellow]Lista larga de hotwords: 25 términos degradaron la cobertura "
                "9 puntos sobre 240 s de audio real (2026-08-03); entran como texto "
                "previo, no como léxico.[/yellow]"
            )
    terms = tuple(t.strip() for t in hotwords.split(",") if t.strip()) if hotwords else ()

    console.print(
        f"[bold]Modelo[/bold] [cyan]{model}[/cyan] · [bold]motor[/bold] [cyan]{route.engine}[/cyan] · "
        f"[bold]device[/bold] [cyan]{route.device}[/cyan] · "
        f"[bold]compute[/bold] [cyan]{route.compute_type}[/cyan]"
    )
    if route.engine != ENGINE_FASTER:
        from speechtotext.core.enginepin import ENGINE_PIN

        console.print(
            f"[bold]Motor[/bold] [cyan]whisper.cpp {ENGINE_PIN['version']}[/cyan] · "
            f"[cyan]{model} {route.compute_type}[/cyan] · [cyan]{route.device}[/cyan]"
        )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"Transcribiendo {audio.name}", total=None)

        avisado = False

        def on_progress(p):
            nonlocal avisado
            if p.stage == "decode" and p.total:
                dur_min = p.total / 60
                if route.eta_factor:
                    eta_min = max(1, round(dur_min * route.eta_factor))
                    fuente = "estimado" if route.estimated else "medido con bench"
                    console.print(f"Duración {dur_min:.1f} min · ETA ~{eta_min} min ({fuente})")
                else:
                    console.print(f"Duración {dur_min:.1f} min · ETA sin medir para esta ruta")
                return
            if p.stage == "transcribe" and p.total and p.total > 1:
                if not avisado:
                    avisado = True
                    console.print(f"[bold]Troceado[/bold] (jobs={jobs}) · {model}")
                linea = f"[{int(p.done)}/{int(p.total)}] {p.detail}"
                if console.is_terminal:
                    progress.update(task, description=linea, total=p.total, completed=p.done)
                else:
                    # Redirigido a archivo, Live no refresca: una linea por trozo o la corrida
                    # de horas queda muda (spec 2026-07-08, "el log mudo al redirigir").
                    console.print(f"  {linea}", markup=False)
            else:
                progress.update(task, description=f"{p.stage} {p.detail}".strip())

        try:
            t = core_transcribe(
                audio, model=model, language=language, engine=route.engine,
                device=route.device, compute_type=route.compute_type, vad=vad,
                beam_size=beam_size, hotwords=terms, diarize=diarize, speakers=speakers,
                identify=identify, threshold=threshold, chunk=chunk, jobs=jobs,
                on_progress=on_progress,
            )
        except AsrError as e:
            if e.code == "unsupported_option":
                console.print(str(e), markup=False)
                raise typer.Exit(2)
            if e.code == "out_of_memory":
                console.print("[red]Se quedó sin memoria al transcribir.[/red]")
                console.print(f"  {e}", markup=False)
                raise typer.Exit(1)
            if e.code == "diarize_unavailable":
                console.print(r'[red]Falta el extra de diarización:[/red] pip install -e ".\[diarize]"')
                raise typer.Exit(1)
            if e.code == "diarize_failed":
                console.print(f"[red]La diarización falló:[/red] {e}")
                console.print(
                    "Revisa que aceptaste los términos de los modelos pyannote y que HF_TOKEN esté configurado."
                )
                raise typer.Exit(1)
            console.print(str(e), markup=False)
            raise typer.Exit(1)
        except AudioDecodeError as e:
            console.print(f"[red]No se pudo procesar el audio:[/red] {e}")
            raise typer.Exit(1)

    for aviso in t.warnings:
        # markup=False: "la marca [?]" es texto, no una etiqueta de rich.
        console.print(aviso, style="yellow", markup=False)

    if t.duration:
        ratio = t.speech_s / t.duration
        voz = f"con voz {t.speech_s / 60:.1f} de {t.duration / 60:.1f} min ({100 * ratio:.0f}%)"
    else:
        voz = "[yellow]cobertura desconocida (duración no medida)[/yellow]"
    if language != "auto":
        idioma = f"Idioma: [bold]{t.language}[/bold] (forzado)"
    else:
        idioma = f"Idioma detectado: [bold]{t.language}[/bold]"
        if t.language_probability is not None:
            idioma += f" (prob={t.language_probability:.2f})"
            if t.language_probability < 0.5:
                idioma += " — dudoso: fíjalo con -l <código>"
    console.print(
        f"{idioma} · duración {t.duration:.1f}s · "
        f"{len(t.segments)} segmentos · {voz} · motor {route.engine}"
    )
    if t.duration:
        if t.gaps:
            # ponytail: se listan los primeros 5 huecos y luego "y N más (ver el JSON)".
            lista = ", ".join(f"{_fmt(a)}-{_fmt(b)} ({b - a:.0f} s)" for a, b in t.gaps[:5])
            if len(t.gaps) > 5:
                lista += f", y {len(t.gaps) - 5} más (ver el JSON)"
            # El consejo sólo aplica a quien tiene el VAD puesto de verdad (la petición
            # efectiva): bajo whispercpp no hay VAD que apagar.
            consejo = " — prueba --no-vad" if t.request.vad else ""
            plural = "hueco" if len(t.gaps) == 1 else "huecos"
            console.print(f"{len(t.gaps)} {plural} sin texto: {lista}{consejo}")
        else:
            console.print("sin huecos de 5 s o más")

    if t.diarization is not None:
        d = t.diarization
        partes = [f"{d.speakers} hablante{'s' if d.speakers != 1 else ''}",
                  f"{d.unattributed_pct}% sin atribuir"]
        if d.enrolled:
            parte = f"{d.identified} de {d.enrolled} voces identificadas"
            if d.best_score is not None:
                parte += f" (mejor score {d.best_score:.2f} < {threshold:.2f})"
            partes.append(parte)
        console.print(" · ".join(partes))
        if d.auto and d.speakers > 5:
            console.print(
                f"[yellow]{d.speakers} hablantes detectados en automático; si sabes cuántos son, "
                "fija el número con --speakers N[/yellow]"
            )

    info = SimpleNamespace(language=t.language, language_probability=t.language_probability,
                           duration=t.duration)
    writers: dict[str, tuple[str, callable]] = {
        "txt": (".txt", lambda p: write_txt(t.segments, p)),
        "srt": (".srt", lambda p: write_srt(t.segments, p)),
        "vtt": (".vtt", lambda p: write_vtt(t.segments, p)),
        "json": (".json", lambda p: write_json(t.segments, info, p, engine_info=t.engine.to_dict(),
                                               speech_s=t.speech_s, gaps=t.gaps)),
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
        "auto",
        "--language",
        "-l",
        help="'auto' (default) detecta con ≥ 30 s de audio; si la probabilidad sale baja, "
        "fíjalo con un código ISO-639-1 (es, en, fr, ...).",
    ),
    model: str = typer.Option(
        "large-v3",
        "--model",
        "-m",
        help="tiny | base | small | medium | large-v3 | distil-large-v3. large-v3 = el único "
        "que no perdió nada en lo medido; small = borrador rápido (5x más veloz, cambia lo "
        "que se dijo).",
    ),
    formats: str = typer.Option(
        "txt,srt,json", "--formats", "-f", help="Formatos separados por coma (txt, srt, vtt, json)."
    ),
    device: str = typer.Option(
        "auto", "--device", "-d",
        help="auto (sondea la GPU; ver `speechtotext probe`) | cpu | cuda",
    ),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help="auto | int8 | int8_float16 | float16 | float32. 'auto' elige int8 en CPU y float16 en GPU.",
    ),
    vad: bool = typer.Option(
        False, "--vad/--no-vad",
        help="Filtro VAD para descartar silencios largos. Apagado por defecto: medido, "
        "pierde frases cortas sin avisar.",
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
        "auto",
        "--engine",
        help="auto (según la máquina; ver `speechtotext probe`) | faster-whisper | whispercpp",
    ),
) -> None:
    """Transcribe un archivo de audio localmente con Whisper (sin enviar nada a internet)."""
    transcribe_file(
        audio, output, language, model, formats, device, compute_type,
        vad, beam_size, diarize, speakers, identify, threshold,
        hotwords=_resolve_hotwords(hotwords, hotwords_file),
        chunk=chunk, jobs=jobs, engine=engine,
    )


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
        "auto", "auto", False, 5, diarize, speakers, identify, threshold,
        hotwords=hotwords,
    )


@app.command()
def find(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio o vídeo a buscar."),
    query: str = typer.Argument(..., help="Palabras a buscar."),
    extract: bool = typer.Option(False, "--extract", "-e", help="Recortar + transcribir la región."),
    region: int = typer.Option(1, "--region", help="Qué región extraer (1 = la más densa)."),
    model: str = typer.Option(
        "large-v3", "--model", "-m",
        help="Modelo para la transcripción del tramo (small = borrador rápido).",
    ),
    scan_model: str = typer.Option("tiny", "--scan-model", help="Modelo del índice."),
    language: str = typer.Option("auto", "--language", "-l", help="Idioma de la transcripción del tramo."),
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

    registry.enroll(name, vec, seconds=seconds, model=diarization.EMBEDDING_MODEL)
    console.print(f"  [green]OK[/green] voz de {name} registrada.")


@app.command()
def voices() -> None:
    """Lista las voces registradas."""
    from speechtotext.speakers import registry

    vs = registry.list_voices()
    if not vs:
        console.print("Sin voces registradas. Usa: speechtotext enroll <nombre> <muestra.wav>")
        return
    table = Table("Nombre", "Segundos", "Registrada", "Modelo")
    for v in vs:
        table.add_row(v["name"], str(v.get("seconds", "")), v.get("enrolled_at", ""), v["model"])
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


def _wav_seconds(wav: Path) -> float:
    """Duración de un wav PCM con la stdlib (el recorte del bench ya es 16 kHz mono)."""
    import contextlib
    import wave

    with contextlib.closing(wave.open(str(wav))) as w:
        return w.getnframes() / float(w.getframerate())


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
        duration_s = _wav_seconds(clip)
        configs, _skipped = benchmark.available_configs()
        quick_saltadas = []
        if quick:
            lentas = [
                c for c in configs
                if c["engine"] == "faster-whisper" and c["model"] in ("medium", "large-v3")
            ]
            configs = [c for c in configs if c not in lentas]
            # Las quick-saltadas van a skipped: una tabla con filas ausentes sin razón
            # haría que el consumidor de la tabla eligiera sin saber que faltan candidatas.
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
