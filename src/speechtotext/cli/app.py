"""Transcripción de audio a texto 100% local con faster-whisper.

Sin claves de API, sin subir audio a la nube. Solo necesita ffmpeg en el PATH
(en Linux/macOS: paquete `ffmpeg`; en Windows: https://ffmpeg.org/download.html).

Uso rápido:
    speechtotext transcribe src/static/audio.wav
    speechtotext transcribe charla.mp3 --model medium --language auto --formats txt,srt
"""
from __future__ import annotations

import sys
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
        help="tiny | base | small | medium | large-v3 | distil-large-v3",
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
) -> None:
    """Transcribe un archivo de audio localmente con Whisper (sin enviar nada a internet)."""
    transcribe_file(
        audio, output, language, model, formats, device, compute_type,
        vad, beam_size, diarize, speakers, identify, threshold,
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
