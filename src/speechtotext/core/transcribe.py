"""Un archivo entra, una transcripcion sale. Una sola decodificacion, un solo backend.

El nucleo NUNCA imprime: progreso por callback, avisos en Transcript.warnings, errores
como AsrError con codigo. El CLI, el MCP y la desktop pintan cada uno lo suyo.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from speechtotext.asr.base import AsrBackend, AsrError
from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult
from speechtotext.core.chunked import (
    TimedSegment, TimedWord, chunk_path, clip_to_end, pick_cuts, plan_chunks,
    seg_from_dict, seg_to_dict, shift_segments, should_chunk,
)
from speechtotext.core.formats import find_gaps
from speechtotext.core.postprocess import normalize_hours
from speechtotext.core.probe import ENGINE_FASTER, ENGINE_WHISPERCPP, ENGINES  # noqa: F401 — reexport
from speechtotext.core.segments import LabeledSegment

SAMPLE_RATE = 16000

Stage = Literal["decode", "load", "transcribe", "diarize"]


@dataclass(frozen=True)
class Progress:
    stage: Stage
    done: float
    total: float | None      # None = indeterminado
    detail: str


ProgressCallback = Callable[[Progress], None]


@dataclass(frozen=True)
class Route:
    engine: str
    device: str
    compute_type: str
    reason: str              # una frase para imprimir; vacia si no hay nada que avisar


def resolve_route(engine: str = "auto", device: str = "auto", compute_type: str = "auto",
                  model: str = "large-v3") -> Route:
    """Resuelve engine/device/compute_type explicitos o 'auto'. ponytail: sin sondeo de la
    maquina todavia — 'auto' es faster-whisper en CPU int8; el sondeo llega en el plan 2b."""
    engine = ENGINE_FASTER if engine == "auto" else engine
    if engine not in ENGINES:
        raise ValueError(f"engine {engine!r} no existe; disponibles: {', '.join(ENGINES)}")
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.core.enginepin import _MODEL_ALIAS

        if model not in _MODEL_ALIAS:
            raise ValueError(
                f"modelo {model!r} no está pinneado para whispercpp; disponibles: "
                f"{', '.join(sorted(_MODEL_ALIAS))}"
            )
        if compute_type not in ("auto", "q5_0"):
            raise ValueError(
                f"compute_type={compute_type!r} no soportado con whispercpp; usa 'auto' o 'q5_0'. "
                "Motivo: fp16 = 0.53x tiempo real por paging WDDM en la 980 (medido 2026-07-27)."
            )
        # El binario pinneado es build CUDA y corre en la GPU SIEMPRE (medido en el smoke).
        # Etiquetar cpu seria mentir en el header, la llave y el JSON: se declara cuda y
        # el remapeo se AVISA — pisar un -d cpu en silencio seria la sustitucion callada.
        reason = "" if device == "cuda" else "whisper.cpp (build CUDA) corre en la GPU; device=cuda"
        return Route(ENGINE_WHISPERCPP, "cuda", "q5_0", reason)
    device = "cpu" if device == "auto" else device
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    return Route(ENGINE_FASTER, device, compute_type, "")


def make_backend(engine: str, model: str, device: str, compute_type: str, jobs: int = 1) -> AsrBackend:
    """Los DOS motores se construyen aqui y solo aqui. Imports perezosos: no pagar
    faster_whisper si el motor es whisper.cpp, ni el pin si es faster-whisper."""
    if engine == ENGINE_FASTER:
        import os

        from speechtotext.asr.faster_whisper import FasterWhisperBackend, FasterWhisperConfig

        cfg = FasterWhisperConfig(
            device=device, compute_type=compute_type,
            # Al trocear, N trozos en paralelo comparten un modelo: CT2 necesita N replicas
            # (num_workers) y los hilos se reparten. Con jobs=1, cpu_threads=0 deja decidir a CT2.
            cpu_threads=max(1, (os.cpu_count() or 1) // jobs) if jobs > 1 else 0,
            num_workers=jobs,
        )
        return FasterWhisperBackend(model, cfg)
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.asr.whispercpp import WhisperCppBackend

        return WhisperCppBackend(model)
    raise ValueError(f"engine desconocido: {engine!r}; disponibles: {ENGINES}")


def load_audio(path: Path) -> np.ndarray:
    """Decodifica UNA vez con PyAV a float32 mono 16 kHz. Lanza AudioDecodeError."""
    from speechtotext.audio.io import decode_audio

    with open(path, "rb") as stream:
        return decode_audio(stream, sample_rate=SAMPLE_RATE).samples


@dataclass(frozen=True)
class EngineInfo:
    name: str
    version: str
    model: str
    quant: str
    device: str
    selection: str = "explicit"
    diarization: str | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "version": self.version, "model": self.model,
             "quant": self.quant, "device": self.device, "selection": self.selection}
        if self.diarization is not None:
            d["diarization"] = self.diarization
        return d


@dataclass(frozen=True)
class DiarizationReport:
    speakers: int
    unattributed_pct: int
    identified: int
    enrolled: int
    best_score: float | None   # el mejor coseno cuando nadie alcanzo el umbral
    auto: bool                 # True si el numero de hablantes no lo fijo el usuario


@dataclass
class Transcript:
    segments: list[LabeledSegment]
    language: str
    language_probability: float | None
    duration: float
    speech_s: float
    gaps: list[list[float]]
    engine: EngineInfo
    request: TranscriptionRequest        # la efectiva, tras CAPS
    warnings: tuple[str, ...] = ()
    diarization: DiarizationReport | None = None


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _request(*, language, vad, hotwords, beam_size, word_timestamps) -> TranscriptionRequest:
    return TranscriptionRequest(language=language, hotwords=tuple(hotwords),
                                word_timestamps=word_timestamps, beam_size=beam_size, vad=vad)


def _apply_caps(backend: AsrBackend, request: TranscriptionRequest
                ) -> tuple[TranscriptionRequest, tuple[str, ...]]:
    """Contrato de capacidades: se aplica ANTES de construir modelo alguno."""
    warnings: list[str] = []
    eff = request
    if request.hotwords and backend.caps.hotwords == "rechazado":
        raise AsrError(
            "unsupported_option", False,
            f"--hotwords no tiene efecto con {backend.backend_id} (--prompt es inerte con "
            "-mc 0, medido 2026-07-27); usa --engine faster-whisper",
        )
    if request.vad and backend.caps.vad == "degradado":
        eff = replace(eff, vad=False)
        warnings.append(f"{backend.backend_id} no trae VAD; se transcribe sin filtro")
    if request.word_timestamps and backend.caps.word_timestamps == "degradado":
        eff = replace(eff, word_timestamps=False)
        warnings.append(
            "atribución por segmento (gruesa), sin cortes intra-segmento; la marca [?] queda activa"
        )
    return eff, tuple(warnings)


def _oom(exc: RuntimeError, engine: str) -> RuntimeError:
    """El OOM real sube como RuntimeError del allocator ('mkl_malloc: failed to allocate
    memory', 'ggml_cuda: failed to allocate'). ponytail: se decide por substring 'alloc';
    si un backend inventa otro texto, se propaga crudo, que es la falla correcta."""
    if "alloc" not in str(exc).lower():
        return exc
    if engine == ENGINE_WHISPERCPP:
        consejo = ("La VRAM de la GPU se agotó. Cierra aplicaciones que usen la GPU, "
                   "prueba un modelo menor o usa --engine faster-whisper (CPU).")
    else:
        consejo = ("large-v3 pide ~3.5 GB sólo al cargar. Cierra procesos o usa -m medium. "
                   "No cubre la muerte nativa de --diarize.")
    return AsrError("out_of_memory", False, f"{exc}\n{consejo}")


def _timed(result: TranscriptionResult) -> list[TimedSegment]:
    out: list[TimedSegment] = []
    for s in result.segments:
        words = [TimedWord(w.start, w.end, w.text) for w in s.words] or None
        out.append(TimedSegment(s.start, s.end, s.text, words,
                                no_speech=s.native_signals.no_speech,
                                avg_logprob=s.native_signals.avg_logprob,
                                compression_ratio=s.native_signals.compression_ratio))
    return out


def _identity(path: Path, backend: AsrBackend, request: TranscriptionRequest) -> str:
    st = path.stat()
    return "|".join(str(x) for x in (
        path.resolve(), st.st_size, int(st.st_mtime), backend.backend_id, backend.model_id,
        backend.quant, backend.device, json.dumps(request.to_dict(), sort_keys=True),
    ))


def _run_span(backend, samples, request, start, end, identity, cancel):
    """(segmentos globales, desde_cache, idioma, probabilidad). El checkpoint viejo puede
    traer un fantasma sobre el relleno: clip_to_end tambien al leer."""
    if cancel is not None and cancel.is_set():
        raise AsrError("cancelled", True, "transcripción cancelada")
    path = chunk_path(identity, start, end) if identity is not None else None
    if path is not None and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            segs = clip_to_end([seg_from_dict(d) for d in data["segments"]], end)
            return segs, True, data.get("language"), None
        except (json.JSONDecodeError, KeyError):
            pass  # checkpoint corrupto -> recomputar
    a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    result = backend.transcribe(samples[a:b], request)
    segs = clip_to_end(shift_segments(_timed(result), start), end)
    if path is not None:
        path.write_text(json.dumps({"language": result.language,
                                    "segments": [seg_to_dict(s) for s in segs]},
                                   ensure_ascii=False), encoding="utf-8")
    return segs, False, result.language, result.native_signals.language_probability


def _diarize(samples, segments, speakers, identify, threshold):
    from speechtotext.speakers import diarization, registry
    from speechtotext.speakers.identify import assign_names, cosine

    try:
        turns, clusters = diarization.diarize(samples, SAMPLE_RATE, num_speakers=speakers)
    except ImportError as exc:
        raise AsrError("diarize_unavailable", False,
                       'falta el extra de diarización: pip install -e ".[diarize]"') from exc
    except Exception as exc:
        raise AsrError("diarize_failed", False, str(exc)) from exc
    labeled = diarization.assign_segments(segments, turns)
    name_map: dict[str, str] = {}
    enrolled: dict = {}
    if identify:
        enrolled = registry.get_embeddings(diarization.EMBEDDING_MODEL)
        if enrolled:
            name_map = assign_names(clusters, enrolled, threshold)
    best = None
    if enrolled and not name_map and clusters:
        best = max(cosine(vec, ref) for vec in clusters.values() for ref in enrolled.values())
    sin = round(100 * sum(1 for s in labeled if s.speaker is None) / len(labeled)) if labeled else 0
    report = DiarizationReport(len(clusters), sin, len(name_map), len(enrolled), best, speakers is None)
    return diarization.apply_names(labeled, name_map), report


def transcribe(
    audio: Path | np.ndarray,
    *,
    model: str = "large-v3",
    language: str = "auto",
    engine: str = "auto",
    device: str = "auto",
    compute_type: str = "auto",
    vad: bool = False,
    beam_size: int = 5,
    hotwords: tuple[str, ...] = (),
    word_timestamps: bool = False,
    diarize: bool = False,
    speakers: int | None = None,
    identify: bool = True,
    threshold: float = 0.5,
    chunk: bool | None = None,
    jobs: int = 4,
    backend: AsrBackend | None = None,
    on_progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> Transcript:
    """Archivo (o muestras 16 kHz mono) -> Transcript. El corto y el largo son el mismo
    camino con n = 1 trozo. Con muestras sin archivo no hay checkpoint ni cortes por
    silencio (pick_cuts fijo): es la ruta de `find` y de la libreria."""
    emit = on_progress or (lambda p: None)

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise AsrError("cancelled", True, "transcripción cancelada")

    if isinstance(audio, Path):
        emit(Progress("decode", 0, None, audio.name))
        samples = load_audio(audio)
        source: Path | None = audio
    else:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        source = None
    duration = len(samples) / SAMPLE_RATE

    chunking = should_chunk(duration, chunk)
    if chunking:
        spans = plan_chunks(source, duration) if source is not None else pick_cuts([], duration)
    else:
        spans = [(0.0, duration)]

    route = resolve_route(engine, device, compute_type, model) if backend is None else None
    engine_id = route.engine if route is not None else backend.backend_id
    # N subprocesos contra una sola GPU paginan en silencio (medido: 2 concurrentes tardan
    # MAS que en serie): whisper.cpp va de uno en uno, estructuralmente.
    workers = 1 if engine_id == ENGINE_WHISPERCPP else max(1, min(jobs, len(spans)))
    if backend is None:
        backend = make_backend(route.engine, model, route.device, route.compute_type, jobs=workers)

    # word_timestamps solo se piden al diarizar (la asignacion palabra->hablante parte los
    # segmentos en el cambio de voz) o si el llamador los quiere; cuestan.
    request = _request(language=language, vad=vad, hotwords=hotwords, beam_size=beam_size,
                       word_timestamps=word_timestamps or diarize)
    eff, warnings = _apply_caps(backend, request)
    check_cancel()

    emit(Progress("load", 0, None, backend.model_id))
    try:
        backend.warm()
    except RuntimeError as exc:
        translated = _oom(exc, backend.backend_id)
        if translated is exc:
            raise
        raise translated from exc

    # Checkpoints solo al trocear (como hoy): un pase unico no deja nada en disco.
    identity = _identity(source, backend, eff) if (source is not None and chunking) else None

    results: list = [None] * len(spans)
    langs: list = [None] * len(spans)
    probs: list = [None] * len(spans)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_span, backend, samples, eff, s, e, identity, cancel): i
                for i, (s, e) in enumerate(spans)}
        for done, fut in enumerate(as_completed(futs), start=1):
            i = futs[fut]
            s, e = spans[i]
            try:
                results[i], cached, langs[i], probs[i] = fut.result()
            except RuntimeError as exc:   # AsrError tambien es RuntimeError
                # Al primer fallo se cancela lo pendiente: con motor roto y fallos LENTOS
                # (paging, timeout) drenar 17 trozos serian horas.
                pool.shutdown(wait=False, cancel_futures=True)
                if isinstance(exc, AsrError) and exc.code != "backend_failed":
                    raise                                   # cancelled, unsupported_option: tal cual
                translated = _oom(exc, backend.backend_id)  # el mensaje envuelto conserva el texto del allocator
                if translated is not exc:
                    raise translated from exc
                if len(spans) == 1:
                    raise
                raise AsrError("backend_failed", True,
                               f"motor {backend.backend_id} fallo en el trozo {i + 1}/{len(spans)} "
                               f"({_mmss(s)}-{_mmss(e)}): {exc}") from exc
            span = e - s
            cov = 100 * sum(x.end - x.start for x in results[i]) / span if span > 0 else 0.0
            emit(Progress("transcribe", done, len(spans),
                          f"{_mmss(s)}-{_mmss(e)} {cov:.0f}% ({'cache' if cached else 'nuevo'})"))

    segments = [seg for chunk_segs in results for seg in chunk_segs]
    detected = next((l for l in langs if l), None)
    if language != "auto":
        lang_out, prob = language, 1.0   # fiel a faster-whisper: idioma impuesto -> 1.0
    else:
        lang_out = detected or "es"
        prob = probs[0] if len(spans) == 1 else None   # varios trozos: nadie midio una sola

    # La cobertura y los huecos se miden sobre lo que el ASR emitio, ANTES de diarizar:
    # la diarizacion recomprime cada span a sus palabras y publicaria otro numero.
    speech_s = round(sum(s.end - s.start for s in segments), 2)
    gaps = find_gaps(segments, duration)

    report = None
    if diarize:
        check_cancel()
        emit(Progress("diarize", 0, None, ""))
        labeled, report = _diarize(samples, segments, speakers, identify, threshold)
        final = [LabeledSegment(s.start, s.end, normalize_hours(s.text), s.speaker, src_dur=s.src_dur,
                                no_speech=s.no_speech, avg_logprob=s.avg_logprob,
                                compression_ratio=s.compression_ratio) for s in labeled]
    else:
        final = [LabeledSegment(s.start, s.end, normalize_hours(s.text), None,
                                no_speech=s.no_speech, avg_logprob=s.avg_logprob,
                                compression_ratio=s.compression_ratio) for s in segments]

    engine_info = EngineInfo(
        backend.backend_id, backend.engine_version, backend.model_id, backend.quant, backend.device,
        diarization=("word" if eff.word_timestamps else "segment") if diarize else None,
    )
    return Transcript(final, lang_out, prob, duration, speech_s, gaps, engine_info, eff,
                      warnings, report)
