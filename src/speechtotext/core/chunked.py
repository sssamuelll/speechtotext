"""Transcripción por trozos: durabilidad (checkpoint/resume) + paralelismo."""
from __future__ import annotations

import av
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel

# Import por modulo (no por nombre): el thunk de run_chunked llama engines.make_engine
# en el momento, asi los monkeypatch de tests sobre engines.make_engine surten efecto.
# No es perezoso porque engines.py ya importa faster_whisper perezosamente: es barato.
from speechtotext.core import engines
from speechtotext.core.finder import _home
from speechtotext.core.segments import native_signals


@dataclass
class TimedWord:
    start: float
    end: float
    word: str


@dataclass
class TimedSegment:
    start: float
    end: float
    text: str
    words: list[TimedWord] | None = None
    # Señales nativas de faster-whisper (Fase 2, G5), al final, mismo patrón que words.
    no_speech: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


def shift_segments(segments, offset: float) -> list[TimedSegment]:
    """Copia segmentos aplicando `offset` a start/end del segmento y de cada palabra.
    Los Segment de faster-whisper son inmutables; devolvemos TimedSegment nuevos."""
    out: list[TimedSegment] = []
    for s in segments:
        words = getattr(s, "words", None)
        tw = (
            [TimedWord(w.start + offset, w.end + offset, w.word) for w in words]
            if words
            else None
        )
        no_speech, avg_logprob, compression_ratio = native_signals(s)
        out.append(TimedSegment(s.start + offset, s.end + offset, s.text, tw,
                                no_speech=no_speech, avg_logprob=avg_logprob,
                                compression_ratio=compression_ratio))
    return out


_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)")


def parse_silences(stderr: str) -> list[tuple[float, float]]:
    starts = [float(m.group(1)) for m in _SIL_START.finditer(stderr)]
    ends = [float(m.group(1)) for m in _SIL_END.finditer(stderr)]
    return list(zip(starts, ends))  # zip corta el start final sin end


def pick_cuts(
    silences: list[tuple[float, float]],
    duration: float,
    target_len: float = 600.0,
    search: float = 60.0,
) -> list[tuple[float, float]]:
    mids = [(s + e) / 2 for s, e in silences]
    cuts: list[float] = []
    prev = 0.0
    boundary = target_len
    while boundary < duration - 1.0:
        near = [m for m in mids if abs(m - boundary) <= search and m > prev + 1.0]
        cut = min(near, key=lambda m: abs(m - boundary)) if near else boundary
        cuts.append(cut)
        prev = cut
        boundary = cut + target_len
    bounds = [0.0, *cuts, duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def plan_chunks(audio: Path, duration: float, target_len: float = 600.0) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio),
        "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
        silences = parse_silences(proc.stderr.decode("utf-8", errors="ignore"))
    except (FileNotFoundError, OSError):
        silences = []  # sin ffmpeg -> cortes fijos
    return pick_cuts(silences, duration, target_len)


def chunk_path(audio: Path, opts: dict, model: str, start: float, end: float,
               engine: str = "faster-whisper", quant: str = "", device: str = "") -> Path:
    """engine/quant/device entran al join (plan 2.3): sin ellos, faster-whisper int8 y
    whispercpp q5_0 con model='large-v3' darian el MISMO digest — hit silencioso que sirve
    texto de un motor bajo la firma del otro. `quant` es la cuantizacion EFECTIVA
    (quant_for), y `opts` deben ser los EFECTIVOS post-mapeo (effective_opts): asi --vad
    y --no-vad bajo whispercpp comparten digest. Kwargs con default para que los call
    sites viejos sobrevivan; la invalidacion unica de checkpoints esta asumida (PR-1)."""
    st = audio.stat()
    key = "|".join(str(x) for x in (
        audio.resolve(), st.st_size, int(st.st_mtime), model, engine, quant, device,
        opts["language"], opts["beam_size"], opts["vad_filter"],
        opts["hotwords"], opts["word_timestamps"], start, end,
    ))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    d = _home() / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"


def seg_to_dict(seg: TimedSegment) -> dict:
    d = {"start": seg.start, "end": seg.end, "text": seg.text}
    if seg.words is not None:
        d["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in seg.words]
    if seg.no_speech is not None:
        d["no_speech"] = seg.no_speech
    if seg.avg_logprob is not None:
        d["avg_logprob"] = seg.avg_logprob
    if seg.compression_ratio is not None:
        d["compression_ratio"] = seg.compression_ratio
    return d


def seg_from_dict(d: dict) -> TimedSegment:
    words = d.get("words")
    tw = [TimedWord(w["start"], w["end"], w["word"]) for w in words] if words is not None else None
    return TimedSegment(d["start"], d["end"], d["text"], tw, no_speech=d.get("no_speech"),
                        avg_logprob=d.get("avg_logprob"), compression_ratio=d.get("compression_ratio"))


def transcribe_chunk(audio, start, end, opts, get_model, model_name,
                     engine="faster-whisper", quant="", device=""):
    """Devuelve (segmentos_globales, from_cache, idioma_detectado). El idioma se guarda
    en el checkpoint para que run_chunked reporte el real bajo --language auto.
    `get_model` es un thunk, no el modelo: si el checkpoint sirve nadie paga el pico de
    carga (~3.5 GB en large-v3 int8), y si está corrupto el modelo aparece justo aquí.
    engine/quant/device solo se reenvían a chunk_path (la identidad del checkpoint);
    el motor mismo viaja dentro del thunk."""
    path = chunk_path(audio, opts, model_name, start, end, engine, quant, device)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [seg_from_dict(d) for d in data["segments"]], True, data.get("language")
        except (json.JSONDecodeError, KeyError):
            pass  # checkpoint corrupto -> recomputar

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(end - start), "-i", str(audio),
            "-ar", "16000", "-ac", "1", tmp,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        segments_iter, info = get_model().transcribe(tmp, **opts)
        segs = shift_segments(list(segments_iter), start)
        lang = getattr(info, "language", None)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    path.write_text(
        json.dumps({"language": lang, "segments": [seg_to_dict(s) for s in segs]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return segs, False, lang


def probe_duration(audio: Path) -> float:
    container = av.open(str(audio))
    if container.duration:
        return container.duration / 1_000_000.0
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(audio)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def run_chunked(audio, opts, jobs, model_name, device, compute_type, log=print,
                engine=engines.ENGINE_FASTER):
    duration = probe_duration(audio)
    chunks = plan_chunks(audio, duration)
    jobs = max(1, min(jobs, len(chunks)))
    if engine == engines.ENGINE_WHISPERCPP:
        # Defensa en profundidad: la CLI ya clampa, pero run_chunked es API publica y
        # N subprocesos contra una sola GPU paginan en silencio (medido: 2 concurrentes
        # tardan MAS que en serie, 15.7 s vs 11.4 s). Con jobs=1 la serializacion del
        # subprocess es estructural: max_workers=1, un solo whisper-cli vivo a la vez.
        jobs = 1

    # La llave del cache consume los opts EFECTIVOS post-mapeo, no los pedidos: bajo
    # whispercpp --vad y --no-vad producen el mismo digest (la salida del motor es
    # identica) y hotwords truthy revienta aqui como doble candado (la cli valida antes).
    opts = engines.effective_opts(engine, opts)
    # La cuantizacion EFECTIVA (q5_0 bajo whispercpp) es la que identifica el checkpoint.
    quant = engines.quant_for(engine, compute_type, device)

    # El modelo se construye la primera vez que un trozo lo pide de verdad: con el 100%
    # cacheado no se carga nada y el proceso no se juega la RAM por trabajo ya hecho.
    _lk, _box = threading.Lock(), []

    def get_model():
        with _lk:                       # ponytail: lock global, basta para jobs<=8
            if not _box:
                if engine == engines.ENGINE_WHISPERCPP:
                    # ponytail: sin _duration_s por trozo (opts es compartido); el
                    # adaptador usa su techo default de 3600 s = 6x el peor trozo de
                    # 600 s con JIT frio incluido. H6 ajusta la constante.
                    _box.append(engines.make_engine(engine, model_name, device, compute_type))
                else:
                    # Por el nombre local chunked.WhisperModel: los 7 monkeypatch de los
                    # tests (y el default byte a byte) dependen de esta costura.
                    _box.append(WhisperModel(
                        model_name, device=device, compute_type=compute_type,
                        cpu_threads=max(1, (os.cpu_count() or 1) // jobs), num_workers=jobs))
            return _box[0]

    results: list = [None] * len(chunks)
    langs: list = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(transcribe_chunk, audio, s, e, opts, get_model, model_name,
                        engine=engine, quant=quant, device=device): i
            for i, (s, e) in enumerate(chunks)
        }
        for done, fut in enumerate(as_completed(futs), start=1):
            i = futs[fut]
            s, e = chunks[i]
            try:
                results[i], from_cache, langs[i] = fut.result()
            except Exception as exc:
                # Al primer fallo se cancela lo pendiente: con motor roto y fallos LENTOS
                # (paging, timeout) drenar 17 trozos serian horas (plan H4). El mensaje
                # nombra motor y trozo — el dueño del contexto es el troceador (G6).
                pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    f"motor {engine} fallo en el trozo {i + 1}/{len(chunks)} "
                    f"({_mmss(s)}-{_mmss(e)}): {exc}"
                ) from exc
            tag = "cache" if from_cache else "nuevo"
            # "OK" sólo decía que el future no lanzó; el porcentaje dice cuánto del trozo
            # produjo texto. results[i] trae timestamps GLOBALES: el denominador es (e - s).
            span = e - s
            cov = 100 * sum(x.end - x.start for x in results[i]) / span if span > 0 else 0.0
            log(f"[{done}/{len(chunks)}] {_mmss(s)}-{_mmss(e)} {cov:.0f}% ({tag})")
    segments = [seg for chunk_segs in results for seg in chunk_segs]
    # Bajo --language auto (opts language=None) reporta el idioma REAL que detectó el
    # primer trozo, no un "es" hardcodeado; si se forzó idioma, ese manda.
    detected = next((l for l in langs if l), None)
    # Si se forzó idioma, 1.0 es fiel al upstream (faster-whisper devuelve 1 cuando se le
    # impone el idioma). Bajo --language auto la probabilidad real existió por trozo y no
    # se guardó en el checkpoint, así que None: no se fabrica un número que nadie midió.
    info = SimpleNamespace(language=opts.get("language") or detected or "es",
                           language_probability=1.0 if opts.get("language") else None,
                           duration=duration)
    return segments, info


CHUNK_THRESHOLD = 1200.0  # s (20 min): por encima, auto-trocea


def should_chunk(duration: float, chunk_flag: bool | None, threshold: float = CHUNK_THRESHOLD) -> bool:
    if chunk_flag is not None:
        return chunk_flag
    return duration > threshold
