"""One file in, one transcription out. A single decode, a single backend.

The core NEVER prints: progress goes through a callback, warnings through Transcript.warnings,
and errors as AsrError with a code. The CLI, MCP, and desktop each render their own output.
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
from speechtotext.core import probe
from speechtotext.core.probe import ENGINE_FASTER, ENGINE_WHISPERCPP, ENGINES, Route  # noqa: F401 — reexport
from speechtotext.core.segments import LabeledSegment

SAMPLE_RATE = 16000

# pyannote is the default: with the speaker count given it was the most accurate on a
# two-person call. Nemotron is ~30x faster on CPU, takes no count and gives no embeddings
# (so no names). Measured 2026-09-24; README, "Two diarizers".
DIARIZER_PYANNOTE, DIARIZER_NEMOTRON = "pyannote", "nemotron"
DIARIZERS = (DIARIZER_PYANNOTE, DIARIZER_NEMOTRON)

Stage = Literal["download", "decode", "load", "transcribe", "diarize"]


@dataclass(frozen=True)
class Progress:
    stage: Stage
    done: float
    total: float | None      # None = indeterminate
    detail: str


ProgressCallback = Callable[[Progress], None]


def resolve_route(engine: str = "auto", device: str = "auto", compute_type: str = "auto",
                  model: str = "large-v3") -> Route:
    """Probe the machine once (core.probe) and choose the route. ValueError for impossible
    flags; AsrError("insufficient_resources") if the model does not fit in RAM."""
    return probe.choose_route(probe.machine(), model, engine=engine, device=device,
                              compute_type=compute_type)


def make_backend(engine: str, model: str, device: str, compute_type: str, jobs: int = 1) -> AsrBackend:
    """Both engines are built here and only here. Lazy imports: do not pay for
    faster_whisper when the engine is whisper.cpp, or the pin when it is faster-whisper."""
    if engine == ENGINE_FASTER:
        import os

        from speechtotext.asr.faster_whisper import FasterWhisperBackend, FasterWhisperConfig

        cfg = FasterWhisperConfig(
            device=device, compute_type=compute_type,
            # When chunking, N parallel chunks share one model: CT2 needs N replicas
            # (num_workers) and the threads are divided among them. With jobs=1, cpu_threads=0 lets CT2 decide.
            cpu_threads=max(1, (os.cpu_count() or 1) // jobs) if jobs > 1 else 0,
            num_workers=jobs,
        )
        return FasterWhisperBackend(model, cfg)
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.asr.whispercpp import WhisperCppBackend

        return WhisperCppBackend(model)
    raise ValueError(f"unknown engine: {engine!r}; available: {ENGINES}")


def load_audio(path: Path) -> np.ndarray:
    """Decode ONCE with PyAV to float32 mono 16 kHz. Raise AudioDecodeError."""
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
    diarizer: str | None = None   # the diarization model's id, when there was diarization

    def to_dict(self) -> dict:
        d = {"name": self.name, "version": self.version, "model": self.model,
             "quant": self.quant, "device": self.device, "selection": self.selection}
        if self.diarization is not None:
            d["diarization"] = self.diarization
        if self.diarizer is not None:
            d["diarizer"] = self.diarizer
        return d


@dataclass(frozen=True)
class DiarizationReport:
    speakers: int
    unattributed_pct: int
    identified: int
    enrolled: int
    best_score: float | None   # best cosine when nobody reached the threshold
    auto: bool                 # True if the user did not set the number of speakers


@dataclass
class Transcript:
    segments: list[LabeledSegment]
    language: str
    language_probability: float | None
    duration: float
    speech_s: float
    gaps: list[list[float]]
    engine: EngineInfo
    request: TranscriptionRequest        # effective request, after CAPS
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
    """Capability contract: applied BEFORE building any model."""
    warnings: list[str] = []
    eff = request
    if request.hotwords and backend.caps.hotwords == "rejected":
        raise AsrError(
            "unsupported_option", False,
            f"--hotwords has no effect with {backend.backend_id} (--prompt is inert with "
            "-mc 0, measured 2026-07-27); use --engine faster-whisper",
        )
    if request.vad and backend.caps.vad == "degraded":
        eff = replace(eff, vad=False)
        warnings.append(f"{backend.backend_id} has no VAD; transcribing unfiltered")
    if request.word_timestamps and backend.caps.word_timestamps == "degraded":
        eff = replace(eff, word_timestamps=False)
        warnings.append(
            "per-segment attribution (coarse), no intra-segment cuts; the [?] mark stays active"
        )
    return eff, tuple(warnings)


def _oom(exc: RuntimeError, engine: str) -> RuntimeError:
    """The real OOM surfaces as a RuntimeError from the allocator ('mkl_malloc: failed to
    allocate memory', 'ggml_cuda: failed to allocate'). ponytail: decided by the 'alloc'
    substring; if a backend invents different text, it propagates raw, which is the correct failure."""
    if "alloc" not in str(exc).lower():
        return exc
    if engine == ENGINE_WHISPERCPP:
        advice = ("The GPU ran out of VRAM. Close applications that use the GPU, "
                  "try a smaller model, or use --engine faster-whisper (CPU).")
    else:
        advice = ("large-v3 needs ~3.5 GB just to load. Close other processes or use -m medium. "
                  "This does not cover --diarize's own native crash.")
    return AsrError("out_of_memory", False, f"{exc}\n{advice}")


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
    """(global segments, from_cache, language, probability, engine warnings). An old
    checkpoint can contain a phantom over the padding: clip_to_end also runs on read."""
    if cancel is not None and cancel.is_set():
        raise AsrError("cancelled", True, "transcription cancelled")
    path = chunk_path(identity, start, end) if identity is not None else None
    if path is not None and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            segs = clip_to_end([seg_from_dict(d) for d in data["segments"]], end)
            return segs, True, data.get("language"), None, ()
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt checkpoint -> recompute
    a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    result = backend.transcribe(samples[a:b], request)
    segs = clip_to_end(shift_segments(_timed(result), start), end)
    if path is not None:
        path.write_text(json.dumps({"language": result.language,
                                    "segments": [seg_to_dict(s) for s in segs]},
                                   ensure_ascii=False), encoding="utf-8")
    return segs, False, result.language, result.native_signals.language_probability, result.warnings


def _diarize(samples, segments, speakers, identify, threshold, diarizer=DIARIZER_PYANNOTE):
    from speechtotext.speakers import diarization, registry
    from speechtotext.speakers.identify import assign_names, cosine

    try:
        if diarizer == DIARIZER_NEMOTRON:
            from speechtotext.speakers import nemotron

            # No embeddings: the speaker count comes from the turns, and nobody gets a name.
            turns, clusters = nemotron.diarize(samples, SAMPLE_RATE), {}
        else:
            turns, clusters = diarization.diarize(samples, SAMPLE_RATE, num_speakers=speakers)
    except ImportError as exc:
        extra = "nemotron" if diarizer == DIARIZER_NEMOTRON else "diarize"
        raise AsrError("diarize_unavailable", False,
                       f'the diarization extra is missing: pip install -e ".[{extra}]"') from exc
    except Exception as exc:
        raise AsrError("diarize_failed", False, str(exc)) from exc
    labeled = diarization.assign_segments(segments, turns)
    name_map: dict[str, str] = {}
    enrolled: dict = {}
    if identify and diarizer != DIARIZER_NEMOTRON:
        enrolled = registry.get_embeddings(diarization.EMBEDDING_MODEL)
        if enrolled:
            name_map = assign_names(clusters, enrolled, threshold)
    best = None
    if enrolled and not name_map and clusters:
        best = max(cosine(vec, ref) for vec in clusters.values() for ref in enrolled.values())
    unattributed_pct = round(100 * sum(1 for s in labeled if s.speaker is None) / len(labeled)) if labeled else 0
    found = len({spk for *_, spk in turns}) if diarizer == DIARIZER_NEMOTRON else len(clusters)
    report = DiarizationReport(found, unattributed_pct, len(name_map), len(enrolled), best, speakers is None)
    return diarization.apply_names(labeled, name_map), report


def _diarizer_id(diarizer: str) -> str:
    """The model id the JSON records: which checkpoint drew the turns."""
    if diarizer == DIARIZER_NEMOTRON:
        from speechtotext.speakers.nemotron import MODEL_ID

        return MODEL_ID
    from speechtotext.speakers.diarization import EMBEDDING_MODEL

    return EMBEDDING_MODEL


def _nemotron_notes(identify) -> list[str]:
    """Names are on by default, so a run with enrolled voices is not refused: the transcript
    is still what was asked for, without the names. Said, never skipped in silence."""
    from speechtotext.speakers import diarization, registry

    if not identify:
        return []
    n = len(registry.get_embeddings(diarization.EMBEDDING_MODEL))
    if not n:
        return []
    return [f"nemotron gives no voice embeddings: {n} enrolled voice{'s' if n != 1 else ''} "
            "not compared; names need --diarizer pyannote"]


def transcribe(
    audio: Path | str | np.ndarray,
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
    diarizer: str = DIARIZER_PYANNOTE,
    speakers: int | None = None,
    identify: bool = True,
    threshold: float = 0.5,
    chunk: bool | None = None,
    jobs: int = 4,
    backend: AsrBackend | None = None,
    route: Route | None = None,
    on_progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> Transcript:
    """File (path as Path or str, or 16 kHz mono samples) -> Transcript. Short and long
    follow the same path with n = 1 chunk. With samples and no file there is no checkpoint
    or silence-based cutting (fixed pick_cuts): this is the code path for `find` and the library.

    `route`: an already resolved Route (the CLI probes, prints, and passes it); `engine="auto"`
    marks `engine.selection="auto"`."""
    emit = on_progress or (lambda p: None)

    if diarizer not in DIARIZERS:
        raise ValueError(f"unknown diarizer: {diarizer!r}; available: {DIARIZERS}")
    if diarize and diarizer == DIARIZER_NEMOTRON:
        if speakers is not None:
            # The model decides the count itself: the flag would be inert, and an inert knob
            # is rejected, never dropped in silence (design.md, "The probe fills in gaps").
            raise AsrError("unsupported_option", False,
                           f"--speakers {speakers} has no effect with --diarizer nemotron, which "
                           "counts speakers itself; drop --speakers or use --diarizer pyannote")
        # Its dependencies are a git install until transformers 5.18 ships: asking now costs
        # milliseconds, finding out after the ASR costs the whole transcription.
        from speechtotext.speakers import nemotron

        problem = nemotron.missing()
        if problem is not None:
            raise AsrError("diarize_unavailable", False, problem)

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise AsrError("cancelled", True, "transcription cancelled")

    # The route is resolved BEFORE decoding: a nonexistent --engine or a model that does not
    # fit stops in milliseconds, not after decoding an hour of audio. A caller that already
    # probed (the CLI) passes it in `route`, and the machine is inspected only once.
    if backend is not None:
        route = None
    elif route is None:
        route = resolve_route(engine, device, compute_type, model)

    if isinstance(audio, (str, Path)):
        audio = Path(audio)
        emit(Progress("decode", 0, None, audio.name))
        samples = load_audio(audio)
        source: Path | None = audio
        # Second event with done = total = duration: this is what the CLI needs for the ETA.
        emit(Progress("decode", len(samples) / SAMPLE_RATE, len(samples) / SAMPLE_RATE, audio.name))
    else:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        source = None
    duration = len(samples) / SAMPLE_RATE

    chunking = should_chunk(duration, chunk)
    if chunking:
        spans = plan_chunks(source, duration) if source is not None else pick_cuts([], duration)
    else:
        spans = [(0.0, duration)]

    engine_id = route.engine if route is not None else backend.backend_id
    # N subprocesses against a single GPU page silently (measured: 2 concurrent runs take
    # LONGER than serial runs): whisper.cpp runs one at a time, structurally.
    workers = 1 if engine_id == ENGINE_WHISPERCPP else max(1, min(jobs, len(spans)))
    if backend is None:
        backend = make_backend(route.engine, model, route.device, route.compute_type, jobs=workers)

    # word_timestamps are requested only for diarization (word-to-speaker assignment splits
    # segments at the speaker change) or if the caller wants them; they cost time.
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

    # Checkpoints only when chunking (as today): a single pass leaves nothing on disk.
    identity = _identity(source, backend, eff) if (source is not None and chunking) else None

    results: list = [None] * len(spans)
    langs: list = [None] * len(spans)
    probs: list = [None] * len(spans)
    extra: list[str] = []   # engine warnings (e.g. empty_transcript), deduplicated across chunks
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_span, backend, samples, eff, s, e, identity, cancel): i
                for i, (s, e) in enumerate(spans)}
        for done, fut in enumerate(as_completed(futs), start=1):
            i = futs[fut]
            s, e = spans[i]
            try:
                results[i], cached, langs[i], probs[i], span_warnings = fut.result()
            except RuntimeError as exc:   # AsrError is also a RuntimeError
                # On the first failure, pending work is canceled: with a broken engine and SLOW
                # failures (paging, timeout), draining 17 chunks would take hours.
                pool.shutdown(wait=False, cancel_futures=True)
                if isinstance(exc, AsrError) and exc.code != "backend_failed":
                    raise                                   # cancelled, unsupported_option: as-is
                translated = _oom(exc, backend.backend_id)  # the wrapped message preserves the allocator text
                if translated is not exc:
                    raise translated from exc
                if len(spans) == 1:
                    raise
                raise AsrError("backend_failed", True,
                               f"engine {backend.backend_id} failed on chunk {i + 1}/{len(spans)} "
                               f"({_mmss(s)}-{_mmss(e)}): {exc}") from exc
            for w in span_warnings:
                warning = f"{backend.backend_id}: {w}"
                if warning not in extra:
                    extra.append(warning)
            span = e - s
            cov = 100 * sum(x.end - x.start for x in results[i]) / span if span > 0 else 0.0
            emit(Progress("transcribe", done, len(spans),
                          f"{_mmss(s)}-{_mmss(e)} {cov:.0f}% ({'cache' if cached else 'nuevo'})"))

    segments = [seg for chunk_segs in results for seg in chunk_segs]
    detected = next((l for l in langs if l), None)
    if language != "auto":
        lang_out, prob = language, 1.0   # faithful to faster-whisper: forced language -> 1.0
    else:
        lang_out = detected or "es"
        prob = probs[0] if len(spans) == 1 else None   # multiple chunks: nobody measured a single one

    # Coverage and gaps are measured over what ASR emitted, BEFORE diarization:
    # diarization recompresses each span to its words and would publish a different number.
    speech_s = round(sum(s.end - s.start for s in segments), 2)
    gaps = find_gaps(segments, duration)

    report = None
    if diarize:
        check_cancel()
        emit(Progress("diarize", 0, None, ""))
        labeled, report = _diarize(samples, segments, speakers, identify, threshold, diarizer)
        if diarizer == DIARIZER_NEMOTRON:
            extra.extend(_nemotron_notes(identify))
        final = [LabeledSegment(s.start, s.end, normalize_hours(s.text), s.speaker, src_dur=s.src_dur,
                                no_speech=s.no_speech, avg_logprob=s.avg_logprob,
                                compression_ratio=s.compression_ratio) for s in labeled]
    else:
        final = [LabeledSegment(s.start, s.end, normalize_hours(s.text), None,
                                no_speech=s.no_speech, avg_logprob=s.avg_logprob,
                                compression_ratio=s.compression_ratio) for s in segments]

    engine_info = EngineInfo(
        backend.backend_id, backend.engine_version, backend.model_id, backend.quant, backend.device,
        selection="auto" if route is not None and engine == "auto" else "explicit",
        diarization=("word" if eff.word_timestamps else "segment") if diarize else None,
        diarizer=_diarizer_id(diarizer) if diarize else None,
    )
    return Transcript(final, lang_out, prob, duration, speech_s, gaps, engine_info, eff,
                      warnings + tuple(extra), report)
