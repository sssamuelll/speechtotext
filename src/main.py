"""Transcripción de audio a texto 100% local con faster-whisper.

Sin claves de API, sin subir audio a la nube. Solo necesita ffmpeg en el PATH
(en Linux/macOS: paquete `ffmpeg`; en Windows: https://ffmpeg.org/download.html).

Uso rápido:
    python src/main.py src/static/audio.wav
    python src/main.py charla.mp3 --model medium --language auto --formats txt,srt
    python src/main.py reunion.m4a --diarize           # con identificación de hablantes
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False, help="Transcripción de audio offline con Whisper.")
console = Console()

VALID_FORMATS = {"txt", "srt", "vtt", "json"}


@dataclass(frozen=True)
class AnnotatedSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def label(self) -> str:
        return f"[{self.speaker}] " if self.speaker else ""


def format_timestamp(seconds: float, *, srt: bool) -> str:
    millis = max(0, round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1_000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def parse_formats(formats: str) -> set[str]:
    requested = {f.strip().lower() for f in formats.split(",") if f.strip()}
    invalid = requested - VALID_FORMATS
    if invalid:
        raise typer.BadParameter(
            f"Formatos no soportados: {sorted(invalid)}. Usa: {sorted(VALID_FORMATS)}."
        )
    return requested


def resolve_output_base(audio: Path, output: Path | None) -> Path:
    if output is None:
        return audio.with_suffix("")
    if output.exists() and output.is_dir():
        return output / audio.stem
    if str(output).endswith(("/", "\\")):
        output.mkdir(parents=True, exist_ok=True)
        return output / audio.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    return output if output.suffix == "" else output.with_suffix("")


def assign_speakers(
    segments: list[AnnotatedSegment],
    turns: list[tuple[float, float, str]],
) -> list[AnnotatedSegment]:
    """Para cada segmento, asigna el hablante con mayor solapamiento temporal.

    `turns` es una lista de `(start, end, speaker_label)` producida por la
    diarización. Si ningún turno solapa con el segmento, `speaker` queda en None.
    """
    annotated: list[AnnotatedSegment] = []
    for seg in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for t_start, t_end, speaker in turns:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        annotated.append(replace(seg, speaker=best_speaker))
    return annotated


def write_txt(segments: list[AnnotatedSegment], path: Path) -> None:
    lines = [f"{s.label}{s.text}".rstrip() for s in segments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments: list[AnnotatedSegment], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{format_timestamp(seg.start, srt=True)} --> {format_timestamp(seg.end, srt=True)}"
        )
        lines.append(f"{seg.label}{seg.text}".rstrip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[AnnotatedSegment], path: Path) -> None:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{format_timestamp(seg.start, srt=False)} --> {format_timestamp(seg.end, srt=False)}"
        )
        lines.append(f"{seg.label}{seg.text}".rstrip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(segments: list[AnnotatedSegment], info, path: Path) -> None:
    payload = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 2),
        "segments": [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                "speaker": s.speaker,
            }
            for i, s in enumerate(segments)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_diarization(audio_path: Path, hf_token: str | None) -> list[tuple[float, float, str]]:
    """Ejecuta pyannote/speaker-diarization-3.1 sobre el audio.

    Requiere haber aceptado los términos del modelo en huggingface.co y un token
    de acceso (HF_TOKEN env var o flag --hf-token). Es gratis pero la primera
    vez descarga ~30MB de pesos.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise typer.BadParameter(
            "Diarización requiere pyannote.audio. Instala con: pip install '.[diarize]'"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if pipeline is None:  # pragma: no cover
        raise typer.BadParameter(
            "No se pudo cargar el pipeline. ¿Aceptaste los términos del modelo y "
            "exportaste HF_TOKEN?"
        )
    diarization = pipeline(str(audio_path))
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


@app.command()
def transcribe(
    audio: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False, help="Archivo de audio o vídeo."
    ),
    output: Path | None = typer.Option(
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
        False,
        "--diarize/--no-diarize",
        help="Identificar hablantes con pyannote.audio (requiere HF_TOKEN).",
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        envvar="HF_TOKEN",
        help="Token de Hugging Face para descargar el modelo de diarización.",
    ),
) -> None:
    """Transcribe un archivo de audio localmente con Whisper (sin enviar nada a internet)."""
    from faster_whisper import WhisperModel

    requested = parse_formats(formats)
    base = resolve_output_base(audio, output)
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
        annotated = [
            AnnotatedSegment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter
        ]
        progress.update(task, completed=1)

    console.print(
        f"Idioma detectado: [bold]{info.language}[/bold] "
        f"(prob={info.language_probability:.2f}) · duración {info.duration:.1f}s · "
        f"{len(annotated)} segmentos."
    )

    if diarize:
        token = hf_token or os.environ.get("HF_TOKEN")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Diarizando hablantes", total=None)
            turns = run_diarization(audio, token)
            progress.update(task, completed=1)
        annotated = assign_speakers(annotated, turns)
        speakers = sorted({s.speaker for s in annotated if s.speaker})
        console.print(f"Hablantes detectados: [bold]{len(speakers)}[/bold] ({', '.join(speakers)})")

    writers: dict[str, tuple[str, Callable[[Path], None]]] = {
        "txt": (".txt", lambda p: write_txt(annotated, p)),
        "srt": (".srt", lambda p: write_srt(annotated, p)),
        "vtt": (".vtt", lambda p: write_vtt(annotated, p)),
        "json": (".json", lambda p: write_json(annotated, info, p)),
    }
    for fmt in sorted(requested):
        suffix, write_fn = writers[fmt]
        out_path = base.with_suffix(suffix)
        write_fn(out_path)
        console.print(f"  [green]OK[/green] {out_path}")


if __name__ == "__main__":
    app()
