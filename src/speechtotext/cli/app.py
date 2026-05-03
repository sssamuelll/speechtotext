"""Transcripción de audio a texto 100% local con faster-whisper.

Sin claves de API, sin subir audio a la nube. Solo necesita ffmpeg en el PATH
(en Linux/macOS: paquete `ffmpeg`; en Windows: https://ffmpeg.org/download.html).

Uso rápido:
    speechtotext src/static/audio.wav
    speechtotext charla.mp3 --model medium --language auto --formats txt,srt
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from speechtotext.core.formats import (
    parse_formats,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)

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
) -> None:
    """Transcribe un archivo de audio localmente con Whisper (sin enviar nada a internet)."""
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


if __name__ == "__main__":
    app()
