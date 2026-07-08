"""Transcripción de audio a texto 100% local con faster-whisper.

Sin claves de API, sin subir audio a la nube. Solo necesita ffmpeg en el PATH
(en Linux/macOS: paquete `ffmpeg`; en Windows: https://ffmpeg.org/download.html).

Uso rápido:
    speechtotext transcribe src/static/audio.wav
    speechtotext transcribe charla.mp3 --model medium --language auto --formats txt,srt
"""
from __future__ import annotations

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
    los hotwords son sesgo probabilístico, un léxico global envenenaría todo otro audio."""
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
) -> None:
    """Transcribe un archivo (opcionalmente con diarización) y escribe los formatos pedidos."""
    try:
        requested = parse_formats(formats)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    base = _resolve_output_base(audio, output)
    base.parent.mkdir(parents=True, exist_ok=True)

    lang = None if language.lower() == "auto" else language
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    hotwords = (hotwords or "").strip() or None
    if hotwords:
        # markup=False: los términos pueden traer corchetes/acentos que rich malinterpretaría.
        console.print(f"Hotwords: {hotwords}", markup=False)

    console.print(
        f"[bold]Modelo[/bold] [cyan]{model}[/cyan] · "
        f"[bold]device[/bold] [cyan]{device}[/cyan] · "
        f"[bold]compute[/bold] [cyan]{compute_type}[/cyan]"
    )
    whisper = WhisperModel(model, device=device, compute_type=compute_type)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Transcribiendo {audio.name}", total=None)
        segments_iter, info = whisper.transcribe(
            str(audio),
            language=lang,
            beam_size=beam_size,
            vad_filter=vad,
            hotwords=hotwords,
        )
        segments = list(segments_iter)
        progress.update(task, completed=1)

    console.print(
        f"Idioma detectado: [bold]{info.language}[/bold] "
        f"(prob={info.language_probability:.2f}) · duración {info.duration:.1f}s · "
        f"{len(segments)} segmentos."
    )

    if diarize:
        segments = _run_diarization(audio, segments, speakers, identify, threshold)

    writers: dict[str, tuple[str, callable]] = {
        "txt": (".txt", lambda p: write_txt(segments, p)),
        "srt": (".srt", lambda p: write_srt(segments, p)),
        "vtt": (".vtt", lambda p: write_vtt(segments, p)),
        "json": (".json", lambda p: write_json(segments, info, p)),
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
    beam_size: int = typer.Option(5, "--beam-size", help="Tamaño del beam search."),
    diarize: bool = typer.Option(
        False, "--diarize", "-D", help=r"Marcar quién habla (diarización). Requiere el extra \[diarize]."
    ),
    speakers: Optional[int] = typer.Option(
        None, "--speakers", help="Número de hablantes (pista; auto si se omite)."
    ),
    identify: bool = typer.Option(
        True, "--identify/--no-identify", help="Poner nombre a las voces registradas."
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Umbral de coincidencia de voz (coseno, 0-1)."
    ),
    hotwords: Optional[str] = typer.Option(
        None,
        "--hotwords",
        help="Términos difíciles separados por coma (nombres propios, jerga) para sesgar el "
        "modelo en cada ventana. Escríbelos con mayúsculas y tildes.",
    ),
    hotwords_file: Optional[Path] = typer.Option(
        None,
        "--hotwords-file",
        exists=True,
        dir_okay=False,
        help="Archivo con términos difíciles (uno por línea o separados por coma), para un "
        "léxico por proyecto. Se combina con --hotwords.",
    ),
) -> None:
    """Transcribe un archivo de audio localmente con Whisper (sin enviar nada a internet)."""
    transcribe_file(
        audio, output, language, model, formats, device, compute_type,
        vad, beam_size, diarize, speakers, identify, threshold,
        hotwords=_resolve_hotwords(hotwords, hotwords_file),
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
    speakers: Optional[int] = typer.Option(None, "--speakers", help="Nº de hablantes (pista)."),
    identify: bool = typer.Option(True, "--identify/--no-identify", help="Nombrar voces registradas."),
    threshold: float = typer.Option(0.5, "--threshold", help="Umbral de coincidencia de voz."),
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


if __name__ == "__main__":
    app()
