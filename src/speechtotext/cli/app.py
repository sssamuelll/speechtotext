"""Transcripción de audio a texto 100% local con faster-whisper.

Sin claves de API, sin subir audio a la nube. Solo necesita ffmpeg en el PATH
(en Linux/macOS: paquete `ffmpeg`; en Windows: https://ffmpeg.org/download.html).

Uso rápido:
    speechtotext transcribe reunion.m4a
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

app = typer.Typer(add_completion=False, help="Offline audio transcription with Whisper.")
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
            f"[yellow]whisper.cpp {ENGINE_PIN['version']} is not installed: downloading now "
            f"(~{ENGINE_PIN['zip_bytes'] / 1024 ** 2:.0f} MB, once)[/yellow]"
        )
    if route.engine == ENGINE_WHISPERCPP and jobs != 1:
        # 4 subprocesos × 1.28 GB contra 4096 MiB: WDDM no revienta, pagina 25x en silencio
        # (medido). El núcleo ya serializa whisper.cpp; aquí solo se corrige la etiqueta
        # "Troceado (jobs=N)" y se avisa ÚNICAMENTE a quien pidió el motor a mano: bajo
        # --engine auto el usuario no tocó nada y el aviso sería ruido.
        if engine == ENGINE_WHISPERCPP:
            console.print("[yellow]the GPU does not parallelize; jobs=1[/yellow]")
        jobs = 1
    if hotwords:
        n_terms = len([t for t in hotwords.split(",") if t.strip()])
        # ponytail: se cuentan términos y caracteres, no tokens. El conteo exacto necesita
        # el tokenizer del modelo, que en este punto todavía no está cargado; 223 tokens
        # son del orden de 600-700 caracteres en español.
        console.print(
            f"Hotwords ({n_terms} terms, {len(hotwords)} characters): {hotwords}",
            markup=False,
        )
        if n_terms >= 10 or len(hotwords) >= 300:
            console.print(
                "[yellow]Long hotwords list: 25 terms degraded coverage by 9 points "
                "over 240 s of real audio (2026-08-03); they enter as prior text, "
                "not as a lexicon.[/yellow]"
            )
    terms = tuple(t.strip() for t in hotwords.split(",") if t.strip()) if hotwords else ()

    console.print(
        f"[bold]Model[/bold] [cyan]{model}[/cyan] · [bold]engine[/bold] [cyan]{route.engine}[/cyan] · "
        f"[bold]device[/bold] [cyan]{route.device}[/cyan] · "
        f"[bold]compute[/bold] [cyan]{route.compute_type}[/cyan]"
    )
    if route.engine != ENGINE_FASTER:
        from speechtotext.core.enginepin import ENGINE_PIN

        console.print(
            f"[bold]Engine[/bold] [cyan]whisper.cpp {ENGINE_PIN['version']}[/cyan] · "
            f"[cyan]{model} {route.compute_type}[/cyan] · [cyan]{route.device}[/cyan]"
        )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"Transcribing {audio.name}", total=None)

        avisado = False

        def on_progress(p):
            nonlocal avisado
            if p.stage == "decode" and p.total:
                dur_min = p.total / 60
                if route.eta_factor:
                    eta_min = max(1, round(dur_min * route.eta_factor))
                    fuente = "estimated" if route.estimated else "measured with bench"
                    console.print(f"Duration {dur_min:.1f} min · ETA ~{eta_min} min ({fuente})")
                else:
                    console.print(f"Duration {dur_min:.1f} min · ETA not measured for this route")
                return
            if p.stage == "transcribe" and p.total and p.total > 1:
                if not avisado:
                    avisado = True
                    console.print(f"[bold]Chunked[/bold] (jobs={jobs}) · {model}")
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
                audio, model=model, language=language, engine=engine,
                device=device, compute_type=compute_type, route=route, vad=vad,
                beam_size=beam_size, hotwords=terms, diarize=diarize, speakers=speakers,
                identify=identify, threshold=threshold, chunk=chunk, jobs=jobs,
                on_progress=on_progress,
            )
        except AsrError as e:
            if e.code == "unsupported_option":
                console.print(str(e), markup=False)
                raise typer.Exit(2)
            if e.code == "out_of_memory":
                console.print("[red]Ran out of memory while transcribing.[/red]")
                console.print(f"  {e}", markup=False)
                raise typer.Exit(1)
            if e.code == "diarize_unavailable":
                console.print(r'[red]The diarization extra is missing:[/red] pip install -e ".\[diarize]"')
                raise typer.Exit(1)
            if e.code == "diarize_failed":
                console.print(f"[red]Diarization failed:[/red] {e}")
                console.print(
                    "Check that you accepted the pyannote model terms and that HF_TOKEN is set."
                )
                raise typer.Exit(1)
            console.print(str(e), markup=False)
            raise typer.Exit(1)
        except AudioDecodeError as e:
            console.print(f"[red]Could not process the audio:[/red] {e}")
            raise typer.Exit(1)

    for aviso in t.warnings:
        # markup=False: "la marca [?]" es texto, no una etiqueta de rich.
        console.print(aviso, style="yellow", markup=False)

    if t.duration:
        ratio = t.speech_s / t.duration
        voz = f"speech {t.speech_s / 60:.1f} of {t.duration / 60:.1f} min ({100 * ratio:.0f}%)"
    else:
        voz = "[yellow]coverage unknown (duration not measured)[/yellow]"
    if language != "auto":
        idioma = f"Language: [bold]{t.language}[/bold] (forced)"
    else:
        idioma = f"Language detected: [bold]{t.language}[/bold]"
        if t.language_probability is not None:
            idioma += f" (prob={t.language_probability:.2f})"
            if t.language_probability < 0.5:
                idioma += " — uncertain: set it with -l <code>"
    console.print(
        f"{idioma} · duration {t.duration:.1f}s · "
        f"{len(t.segments)} segments · {voz} · engine {route.engine}"
    )
    if t.duration:
        if t.gaps:
            # ponytail: se listan los primeros 5 huecos y luego "y N más (ver el JSON)".
            lista = ", ".join(f"{_fmt(a)}-{_fmt(b)} ({b - a:.0f} s)" for a, b in t.gaps[:5])
            if len(t.gaps) > 5:
                lista += f", and {len(t.gaps) - 5} more (see the JSON)"
            # El consejo sólo aplica a quien tiene el VAD puesto de verdad (la petición
            # efectiva): bajo whispercpp no hay VAD que apagar.
            consejo = " — try --no-vad" if t.request.vad else ""
            plural = "gap" if len(t.gaps) == 1 else "gaps"
            console.print(f"{len(t.gaps)} {plural} without text: {lista}{consejo}")
        else:
            console.print("no gaps of 5 s or more")

    if t.diarization is not None:
        d = t.diarization
        partes = [f"{d.speakers} speaker{'s' if d.speakers != 1 else ''}",
                  f"{d.unattributed_pct}% unattributed"]
        if d.enrolled:
            parte = f"{d.identified} of {d.enrolled} voices identified"
            if d.best_score is not None:
                parte += f" (best score {d.best_score:.2f} < {threshold:.2f})"
            partes.append(parte)
        console.print(" · ".join(partes))
        if d.auto and d.speakers > 5:
            console.print(
                f"[yellow]{d.speakers} speakers detected automatically; if you know how many "
                "there are, set the number with --speakers N[/yellow]"
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
        ..., exists=True, readable=True, dir_okay=False, help="Audio or video file."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output folder or base path (defaults to next to the audio)."
    ),
    language: str = typer.Option(
        "auto",
        "--language",
        "-l",
        help="'auto' (default) detects from 30 s of audio or more; if the probability "
        "comes back low, it prints it and suggests -l.",
    ),
    model: str = typer.Option(
        "large-v3",
        "--model",
        "-m",
        help="tiny | base | small | medium | large-v3 | distil-large-v3. large-v3 is the only "
        "one that lost nothing in testing; small is a fast draft (5x faster, changes what "
        "was said).",
    ),
    formats: str = typer.Option(
        "txt,srt,json", "--formats", "-f", help="Comma-separated formats (txt, srt, vtt, json)."
    ),
    device: str = typer.Option(
        "auto", "--device", "-d",
        help="auto (probes the GPU; see `speechtotext probe`) | cpu | cuda",
    ),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help="auto | int8 | int8_float16 | float16 | float32. 'auto' picks int8 on CPU, float16 "
        "on GPU, and q5_0 under whisper.cpp (where only auto and q5_0 are valid).",
    ),
    vad: bool = typer.Option(
        False, "--vad/--no-vad",
        help="VAD filter to drop long silences. Off by default: measured, it removes "
        "short sentences without saying so.",
    ),
    beam_size: int = typer.Option(5, "--beam-size", min=1, help="Beam search size."),
    diarize: bool = typer.Option(
        False, "--diarize", "-D", help=r"Mark who's speaking (diarization). Requires the \[diarize] extra."
    ),
    speakers: Optional[int] = typer.Option(
        None, "--speakers", min=1, help="Number of speakers (a hint; auto when omitted)."
    ),
    identify: bool = typer.Option(
        True, "--identify/--no-identify", help="Name enrolled voices."
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", min=0.0, max=1.0, help="Voice match threshold (cosine, 0-1)."
    ),
    hotwords: Optional[str] = typer.Option(
        None,
        "--hotwords",
        help="Hard terms, comma separated (proper nouns, jargon): they go into every "
        "window as if they were the previous conversation, so a short list helps and a "
        "long one hurts. Write them capitalized and accented.",
    ),
    hotwords_file: Optional[Path] = typer.Option(
        None,
        "--hotwords-file",
        exists=True,
        dir_okay=False,
        help="File with hard terms (one per line or comma separated), for a per-project "
        "lexicon. Combines with --hotwords. Ceiling: above 223 tokens "
        "(~600-700 characters) the list is truncated silently "
        "(faster_whisper/transcribe.py:1546-1547).",
    ),
    chunk: Optional[bool] = typer.Option(
        None, "--chunk/--no-chunk",
        help="Chunk the audio for checkpoint/resume plus parallelism. Automatic when longer than 20 min.",
    ),
    jobs: int = typer.Option(
        4, "--jobs", "-j", help="Chunks in parallel when chunking (they share one model).",
    ),
    engine: str = typer.Option(
        "auto",
        "--engine",
        help="auto (based on the machine; see `speechtotext probe`) | faster-whisper | whispercpp",
    ),
) -> None:
    """Transcribe an audio file locally with Whisper (nothing leaves your machine)."""
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
        console.print(f"[red]Region {region} out of range (there are {len(regions)}).[/red]")
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
            "[red]ffmpeg is not on the PATH.[/red] "
            "Install it: [cyan]winget install Gyan.FFmpeg[/cyan]"
        )
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Could not clip the audio:[/red] {e.stderr.decode(errors='ignore')[:200]}")
        raise typer.Exit(1)

    console.print(f"  [green]Clip[/green] {clip} ({_fmt(r.start)}–{_fmt(r.end)})")
    transcribe_file(
        clip, base_dir, language, model, formats,
        "auto", "auto", False, 5, diarize, speakers, identify, threshold,
        hotwords=hotwords,
    )


@app.command()
def find(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio or video to search."),
    query: str = typer.Argument(..., help="Words to search for."),
    extract: bool = typer.Option(False, "--extract", "-e", help="Clip and transcribe the region."),
    region: int = typer.Option(1, "--region", help="Which region to extract (1 = the densest)."),
    model: str = typer.Option(
        "large-v3", "--model", "-m",
        help="Model for transcribing the clip (small = fast draft).",
    ),
    scan_model: str = typer.Option("tiny", "--scan-model", help="Model for the index."),
    language: str = typer.Option("auto", "--language", "-l", help="Language for transcribing the clip."),
    formats: str = typer.Option("txt,srt", "--formats", "-f", help="Output formats for the clip."),
    diarize: bool = typer.Option(False, "--diarize", "-D", help="Diarize the extracted clip."),
    speakers: Optional[int] = typer.Option(None, "--speakers", min=1, help="Number of speakers (a hint)."),
    identify: bool = typer.Option(True, "--identify/--no-identify", help="Name enrolled voices."),
    threshold: float = typer.Option(0.5, "--threshold", min=0.0, max=1.0, help="Voice match threshold."),
    context: float = typer.Option(10.0, "--context", help="Seconds of margin when clipping."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output folder for the clip."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force a rebuild of the index."),
    top: int = typer.Option(5, "--top", help="How many regions to list."),
    hotwords: Optional[str] = typer.Option(
        None, "--hotwords", help="Hard terms for transcribing the clip (see transcribe)."
    ),
    hotwords_file: Optional[Path] = typer.Option(
        None, "--hotwords-file", exists=True, dir_okay=False,
        help="Lexicon file for transcribing the clip (see transcribe).",
    ),
) -> None:
    """Search a long audio file; with --extract, clip and transcribe the region."""
    from speechtotext.core import finder

    segments, cached = finder.load_or_build_index(audio, scan_model, rebuild)
    console.print(f"Index: {'cached' if cached else 'built'} ({scan_model}, {len(segments)} segments)")

    regions = finder.search(segments, query, top=top)
    if not regions:
        console.print(f'No match for "{query}" in the audio.', markup=False)
        raise typer.Exit(0)

    console.print(f"{len(regions)} regions for \"{query}\":", markup=False)
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
    name: str = typer.Argument(..., help="Name of the person."),
    sample: Path = typer.Argument(..., exists=True, dir_okay=False, help="Sample audio of their voice."),
) -> None:
    """Enroll a person's voice from a sample recording (>=10s recommended)."""
    import contextlib
    import wave

    from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav
    from speechtotext.speakers import diarization, registry

    try:
        wav = transcode_to_wav(sample.read_bytes())
    except (FfmpegMissingError, TranscodeError) as e:
        console.print(f"[red]Could not process the audio:[/red] {e}")
        raise typer.Exit(1)
    try:
        with contextlib.closing(wave.open(str(wav))) as w:
            seconds = w.getnframes() / float(w.getframerate())
        if seconds < 10:
            console.print(
                f"[yellow]Note:[/yellow] short sample ({seconds:.0f}s); >=10s is more reliable."
            )
        try:
            vec = diarization.embed_voice(str(wav))
        except ImportError:
            console.print(r'[red]The diarization extra is missing:[/red] pip install -e ".\[diarize]"')
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Could not enroll the voice:[/red] {e}")
            raise typer.Exit(1)
    finally:
        wav.unlink(missing_ok=True)

    registry.enroll(name, vec, seconds=seconds, model=diarization.EMBEDDING_MODEL)
    console.print(f"  [green]OK[/green] {name}'s voice enrolled.")


@app.command()
def voices() -> None:
    """List enrolled voices."""
    from speechtotext.speakers import registry

    vs = registry.list_voices()
    if not vs:
        console.print("No enrolled voices. Use: speechtotext enroll <name> <sample.wav>")
        return
    table = Table("Name", "Seconds", "Enrolled", "Model")
    for v in vs:
        table.add_row(v["name"], str(v.get("seconds", "")), v.get("enrolled_at", ""), v["model"])
    console.print(table)


@app.command()
def forget(name: str = typer.Argument(..., help="Name of the voice to delete.")) -> None:
    """Delete an enrolled voice."""
    from speechtotext.speakers import registry

    if registry.remove(name):
        console.print(f"  [green]OK[/green] {name} deleted.")
    else:
        console.print(f"[red]No voice named {name}.[/red]")
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
        raise RuntimeError(f"ffmpeg could not clip the audio: {stderr[-300:]}") from e
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

    t = Table("Engine", "Model", "x_rt", "load_s", "RAM MB", "VRAM MB", "Caps", "WER", "Error")
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
    errores = "error" if con_error == 1 else "errors"
    console.print(f"{ok} working · {con_error} with {errores} · {len(skipped)} skipped")
    for s in skipped:
        console.print(f"  skipped {s['engine']} {s['model']}: {s['reason']}", markup=False)

    recs = table.get("recommendations") or []
    if recs:
        r_t = Table("Use case", "Config", "Reason", title="Which config for what?")
        for rec in recs:
            e = rec.get("choice")
            config = f"{e['engine']} {e['model']}" if e else "— no candidate —"
            r_t.add_row(rec["case"], config, rec["reason"])
        Console(width=120).print(r_t)


@app.command()
def bench(
    audio: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False,
        help="Audio to measure. Omit it (or use --show) to see the last table.",
    ),
    seconds: float = typer.Option(
        60.0, "--seconds",
        help="Seconds of audio to measure (initial clip; bounds the cost of the bench).",
    ),
    quick: bool = typer.Option(
        False, "--quick",
        help="Skips faster-whisper's medium and large-v3: they are the slow ones on CPU "
        "(minutes per config) and the rest are enough for a first decision.",
    ),
    show: bool = typer.Option(
        False, "--show", help="Print the saved table without measuring again."
    ),
) -> None:
    """Measure the viable ASR configs on THIS machine and save bench.json."""
    from speechtotext.core import benchmark

    if show or audio is None:
        table = benchmark.read_table()
        if table is None:
            console.print(
                "[red]No benchmark table yet.[/red] "
                "Measure it with: [cyan]speechtotext bench <audio>[/cyan]"
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
        console.print(f"[red]Could not process the audio:[/red] {e}")
        raise typer.Exit(1)
    try:
        clip = _trim_wav(wav, seconds)
    except RuntimeError as e:
        console.print(f"[red]Could not clip the audio:[/red] {e}")
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
                {"engine": c["engine"], "model": c["model"], "reason": "skipped by --quick"}
                for c in lentas
            ]
        console.print(f"Measuring {len(configs)} configs over {duration_s:.1f}s of audio...")

        def _progress(cfg, res):
            # Una fila al terminar cada config: el bench tarda minutos y el silencio
            # se confunde con un cuelgue. markup=False: el error trae corchetes.
            if res.get("error"):
                console.print(
                    f"  FAILED {cfg['engine']} {cfg['model']}: {res['error'][:120]}",
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
    console.print(f"Table written to {path}")
    _print_bench(table)


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


@app.command()
def probe() -> None:
    """Probe this machine and show the route `transcribe` would choose (paste it into an issue)."""
    m = core_probe.machine()
    console.print(f"platform   {m.platform}", markup=False, soft_wrap=True)
    console.print(f"cpu_count  {m.cpu_count}", markup=False, soft_wrap=True)
    console.print(f"ram_gb     {m.ram_gb if m.ram_gb is not None else 'not measured'}",
                 markup=False, soft_wrap=True)
    console.print(f"cuda       {m.cuda}", markup=False, soft_wrap=True)
    console.print(f"gpu        {m.gpu_name or '-'}", markup=False, soft_wrap=True)
    console.print(f"vram_free  {f'{m.vram_free_gb} GB' if m.vram_free_gb is not None else '-'}",
                 markup=False, soft_wrap=True)
    console.print(f"whispercpp {m.whispercpp or 'not installed'}", markup=False, soft_wrap=True)
    for model in ("large-v3", "small"):
        try:
            r = core_probe.choose_route(m, model)
        except AsrError as e:
            console.print(f"{model:9} {e}", markup=False, soft_wrap=True)
            continue
        if r.eta_factor:
            eta = f"~{1 / r.eta_factor:.1f}x real time{' (estimated)' if r.estimated else ' (bench)'}"
        else:
            eta = "not measured"
        console.print(
            f"{model:9} {r.engine} · {r.device} · {r.compute_type} · {eta} · {r.reason or 'no notices'}",
            markup=False, soft_wrap=True,
        )


models_app = typer.Typer(help="Local models: list, pull, and remove (rm).")
app.add_typer(models_app, name="models")


@models_app.callback(invoke_without_command=True)
def models_list(ctx: typer.Context) -> None:
    """List installed models (faster-whisper in the HF cache, whisper.cpp in data_dir)."""
    if ctx.invoked_subcommand is not None:
        return
    from speechtotext.core import models

    rows = models.installed()
    if not rows:
        console.print("No installed models. Pull one: [cyan]speechtotext models pull large-v3[/cyan]")
        return
    t = Table("engine", "model", "size", "verified", "path")
    for mi in rows:
        t.add_row(mi.engine, mi.name, _gb(mi.size_bytes), "yes" if mi.verified else "-", str(mi.path))
    Console(width=140).print(t)
    console.print(f"Data in {models.data_dir()}", markup=False)


@models_app.command("pull")
def models_pull(
    name: str = typer.Argument(..., help="tiny | base | small | medium | large-v3 | distil-large-v3"),
    engine: str = typer.Option(ENGINE_FASTER, "--engine", help="faster-whisper | whispercpp"),
) -> None:
    """Download a model. Announces the size beforehand; huggingface_hub draws the progress bar."""
    from speechtotext.core import models

    try:
        size = models.remote_size(engine, name)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    console.print(f"Downloading {name} ({engine}{', ~' + _gb(size) if size else ''})...", markup=False)
    try:
        path = models.ensure(engine, name)
    except Exception as e:   # frontera del CLI: sha que no cuadra, sin red... se imprime y sale 1
        console.print(str(e), style="red", markup=False)
        raise typer.Exit(1)
    console.print(f"  [green]OK[/green] {path}")


@models_app.command("rm")
def models_rm(
    name: str = typer.Argument(..., help="Model name (see `speechtotext models`)."),
    engine: str = typer.Option(ENGINE_FASTER, "--engine", help="faster-whisper | whispercpp"),
) -> None:
    """Remove a local model."""
    from speechtotext.core import models

    try:
        models.remove(engine, name)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    except FileNotFoundError as e:
        console.print(str(e), style="red", markup=False)
        raise typer.Exit(1)
    console.print(f"  [green]Removed[/green] {name} ({engine})")


@app.command()
def mcp() -> None:
    """Serve the tools over MCP on stdio (Claude Desktop and compatible clients)."""
    from speechtotext.cli import mcp_server

    mcp_server.serve()


if __name__ == "__main__":
    app()
