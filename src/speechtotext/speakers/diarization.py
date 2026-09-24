"""Batch diarization: overlap-based assignment (pure) + pyannote (lazy)."""
from __future__ import annotations

from speechtotext.core.segments import LabeledSegment, native_signals

# The checkpoint that is loaded and, therefore, the vector space of the embeddings it
# produces: the voice registry archives and filters them by this key, not by dimension.
# A single literal on purpose — with two, anyone changing the checkpoint and forgetting
# the key leaves already-enrolled vectors labeled with a space that is no longer theirs.
EMBEDDING_MODEL = "pyannote/speaker-diarization-community-1"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_speaker(start: float, end: float, turns) -> str | None:
    """Speaker with the greatest TOTAL overlap with [start, end]. Aggregate by speaker,
    not by turn: pyannote splits the same speaker into several turns, and the largest
    individual turn may belong to the minority speaker. None if there is no overlap.
    Tie -> whichever appeared first."""
    totals: dict[str, float] = {}
    for t0, t1, spk in turns:
        ov = _overlap(start, end, t0, t1)
        if ov > 0.0:
            totals[spk] = totals.get(spk, 0.0) + ov
    return max(totals, key=totals.get) if totals else None


def assign_segments(segments, turns: list[tuple[float, float, str]]) -> list[LabeledSegment]:
    """Label each segment with its speaker. With word timestamps (word_timestamps), split
    the segment at internal speaker changes: without this, a Whisper segment that crosses
    a turn boundary is labeled entirely with one speaker and the tail carries over to the
    next one (cutting mid-phrase). Without words, fall back to coarse mode: one speaker per
    segment (maximum overlap).

    A word without a turn (it falls in a gap between turns —pyannote does not cover the
    entire timeline—or lasts 0s) inherits the speaker of the current run instead of
    starting a new one: otherwise, it would appear as a one-word 'Speaker ?' in the middle
    of a monologue."""
    out: list[LabeledSegment] = []
    for s in segments:
        no_speech, avg_logprob, compression_ratio = native_signals(s)
        words = getattr(s, "words", None)
        if not words:
            out.append(LabeledSegment(s.start, s.end, s.text, _best_speaker(s.start, s.end, turns),
                                      src_dur=s.end - s.start, no_speech=no_speech,
                                      avg_logprob=avg_logprob, compression_ratio=compression_ratio))
            continue
        run_spk: str | None = None
        run_words: list[str] = []
        run_start = run_end = None
        # The N runs from the same segment inherit the same src_dur (the span of the
        # segment emitted by ASR): this is the correct approximation —they come from the
        # same decoding window—and without it the is_suspect gate turns off under
        # --diarize, because each run is recompressed to the span of its words. The native
        # signals travel the same way: same origin, same window.
        src_dur = s.end - s.start
        for w in words:
            spk = _best_speaker(w.start, w.end, turns)
            if not run_words:
                run_start, run_spk = w.start, spk
            elif spk is None or spk == run_spk:
                pass  # inherit: word without turn or same speaker -> continue the run
            elif run_spk is None:
                run_spk = spk  # the run had no speaker -> adopt the first real one
            else:
                out.append(LabeledSegment(run_start, run_end, "".join(run_words), run_spk,
                                          src_dur=src_dur, no_speech=no_speech,
                                          avg_logprob=avg_logprob, compression_ratio=compression_ratio))
                run_words, run_start, run_spk = [], w.start, spk
            run_words.append(w.word)
            run_end = w.end
        if run_words:
            out.append(LabeledSegment(run_start, run_end, "".join(run_words), run_spk,
                                      src_dur=src_dur, no_speech=no_speech,
                                      avg_logprob=avg_logprob, compression_ratio=compression_ratio))
    return out


def humanize_speaker(speaker_id: str) -> str:
    try:
        n = int(speaker_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return speaker_id
    return f"Speaker {n + 1}"


def apply_names(
    labeled: list[LabeledSegment], name_map: dict[str, str]
) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for s in labeled:
        if s.speaker is None:
            spk: str | None = None
        else:
            spk = name_map.get(s.speaker) or humanize_speaker(s.speaker)
        out.append(LabeledSegment(s.start, s.end, s.text, spk, src_dur=s.src_dur,
                                  no_speech=s.no_speech, avg_logprob=s.avg_logprob,
                                  compression_ratio=s.compression_ratio))
    return out


# --- Model-backed part (pyannote 4.x). Lazy imports on purpose: torch/pyannote are
# heavy, and this module must remain importable in the base venv (without the [diarize]
# extra) to use the pure functions above. The community-1 pipeline returns, in one pass,
# diarization AND one embedding per speaker. The transcription already has the samples
# in memory; `enroll` reads them from a wav with `wave`. ---

_PIPELINE = None
# The default 32 is not pyannote's default (which is 1): it comes from the community-1
# checkpoint's config.yaml. At 32 the peak is 2620 MB; at 8, 1369 MB (-48%) for +7% wall
# clock over 180 s, with identical output. On a desktop machine, peak RAM is the scarce
# resource, not the 7 s.
# ponytail: constant, not a config option. Ceiling: if this ever runs on a GPU with ample
# VRAM, promote it to a parameter.
_BATCH = 8


def read_wav(wav_path) -> tuple["np.ndarray", int]:
    """Read a PCM16 wav as float32 mono in [-1, 1] at its native rate. This is the input
    to `enroll` (a voice sample on disk); the transcription already has the samples."""
    import wave

    import numpy as np

    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def _waveform(samples, sample_rate: int) -> dict:
    """{waveform, sample_rate} for pyannote, in memory (avoids torchcodec)."""
    import numpy as np
    import torch

    data = np.ascontiguousarray(samples, dtype=np.float32)
    if not data.flags.writeable:
        # The decoded samples arrive read-only, and torch.from_numpy warns
        # about a read-only buffer on every run, in the user's terminal. The
        # copy costs 4 bytes per sample, next to the gigabytes pyannote uses.
        data = data.copy()
    return {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sample_rate}


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        import os
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the torchcodec warning on import
            from pyannote.audio import Pipeline

        _PIPELINE = Pipeline.from_pretrained(
            EMBEDDING_MODEL, token=os.environ.get("HF_TOKEN")
        )
        # Both are read at call time, so assigning them here is enough.
        _PIPELINE.embedding_batch_size = _BATCH
        _PIPELINE.segmentation_batch_size = _BATCH
    return _PIPELINE


def diarize(samples, sample_rate: int, num_speakers: int | None = None):
    """Diarize float32 mono samples. Return (turns, embeddings).

    turns: list[(start, end, speaker_id)]. embeddings: dict[speaker_id, np.ndarray]
    (one vector per speaker, in the same space as embed_voice → comparable).
    """
    import warnings

    import numpy as np

    pipeline = _get_pipeline()
    wf = _waveform(samples, sample_rate)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pipeline(wf, **kwargs)
    ann = out.speaker_diarization
    turns = [(t.start, t.end, lbl) for t, _, lbl in ann.itertracks(yield_label=True)]
    labels = ann.labels()
    emb = np.asarray(out.speaker_embeddings)
    embeddings = {
        lbl: emb[i]
        for i, lbl in enumerate(labels)
        if i < len(emb) and not np.isnan(emb[i]).any()
    }
    return turns, embeddings


def embed_voice(wav_path):
    """Single-voice embedding (for enroll): force 1 speaker and return its vector."""
    samples, sample_rate = read_wav(wav_path)
    _, embeddings = diarize(samples, sample_rate, num_speakers=1)
    if not embeddings:
        raise ValueError(
            "could not extract a voice embedding (audio too short or no speech)"
        )
    return next(iter(embeddings.values()))
