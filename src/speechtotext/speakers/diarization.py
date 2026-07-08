"""Diarización batch: asignación por solape (pura) + pyannote (perezoso)."""
from __future__ import annotations

from speechtotext.core.segments import LabeledSegment


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_speaker(start: float, end: float, turns) -> str | None:
    """Hablante con más solape TOTAL con [start, end]. Agrega por hablante, no por turno:
    pyannote parte a un mismo hablante en varios turnos, y el turno individual más grande
    puede ser del minoritario. None si no solapa con ninguno. Empate -> el que apareció antes."""
    totals: dict[str, float] = {}
    for t0, t1, spk in turns:
        ov = _overlap(start, end, t0, t1)
        if ov > 0.0:
            totals[spk] = totals.get(spk, 0.0) + ov
    return max(totals, key=totals.get) if totals else None


def assign_segments(segments, turns: list[tuple[float, float, str]]) -> list[LabeledSegment]:
    """Etiqueta cada segmento con su hablante. Con timestamps de palabra (word_timestamps)
    parte el segmento en los cambios de hablante internos: sin esto un segmento de Whisper
    que cruza una frontera de turno se etiqueta entero con un solo hablante y la cola se
    arrastra al siguiente (corta a mitad de sintagma). Sin palabras cae al modo grueso:
    un hablante por segmento (máximo solape).

    Una palabra sin turno (cae en un hueco entre turnos —pyannote no cubre toda la línea de
    tiempo— o dura 0s) hereda el hablante del run en curso, no arranca uno nuevo: si no,
    saldría como 'Hablante ?' de una sola palabra en medio de un monólogo."""
    out: list[LabeledSegment] = []
    for s in segments:
        words = getattr(s, "words", None)
        if not words:
            out.append(LabeledSegment(s.start, s.end, s.text, _best_speaker(s.start, s.end, turns)))
            continue
        run_spk: str | None = None
        run_words: list[str] = []
        run_start = run_end = None
        for w in words:
            spk = _best_speaker(w.start, w.end, turns)
            if not run_words:
                run_start, run_spk = w.start, spk
            elif spk is None or spk == run_spk:
                pass  # hereda: palabra sin turno o mismo hablante -> sigue el run
            elif run_spk is None:
                run_spk = spk  # el run venía sin hablante -> adopta el primero real
            else:
                out.append(LabeledSegment(run_start, run_end, "".join(run_words), run_spk))
                run_words, run_start, run_spk = [], w.start, spk
            run_words.append(w.word)
            run_end = w.end
        if run_words:
            out.append(LabeledSegment(run_start, run_end, "".join(run_words), run_spk))
    return out


def humanize_speaker(speaker_id: str) -> str:
    try:
        n = int(speaker_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return speaker_id
    return f"Hablante {n + 1}"


def apply_names(
    labeled: list[LabeledSegment], name_map: dict[str, str]
) -> list[LabeledSegment]:
    out: list[LabeledSegment] = []
    for s in labeled:
        if s.speaker is None:
            spk: str | None = None
        else:
            spk = name_map.get(s.speaker) or humanize_speaker(s.speaker)
        out.append(LabeledSegment(s.start, s.end, s.text, spk))
    return out


# --- Parte con modelos (pyannote 4.x). Imports perezosos a propósito: torch/pyannote
# pesan y este módulo debe poder importarse en el venv base (sin el extra [diarize])
# para usar las funciones puras de arriba. El pipeline community-1 devuelve, en una
# sola pasada, la diarización Y un embedding por hablante. Cargamos el audio en memoria
# con `wave` (nuestro audio ya es wav 16 kHz mono) porque torchcodec no decodifica
# archivos de forma fiable en Windows con este stack. ---

_PIPELINE = None
_PIPELINE_NAME = "pyannote/speaker-diarization-community-1"


def _load_waveform(wav_path) -> dict:
    """Lee un wav PCM16 a un dict {waveform, sample_rate} para pyannote (evita torchcodec)."""
    import wave

    import numpy as np
    import torch

    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sr}


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        import os
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silencia el aviso de torchcodec al importar
            from pyannote.audio import Pipeline

        _PIPELINE = Pipeline.from_pretrained(
            _PIPELINE_NAME, token=os.environ.get("HF_TOKEN")
        )
    return _PIPELINE


def diarize(wav_path, num_speakers: int | None = None):
    """Diariza un wav 16 kHz mono. Devuelve (turns, embeddings).

    turns: list[(start, end, speaker_id)]. embeddings: dict[speaker_id, np.ndarray]
    (un vector por hablante, en el mismo espacio que embed_voice → comparables).
    """
    import warnings

    import numpy as np

    pipeline = _get_pipeline()
    wf = _load_waveform(wav_path)
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
    """Embedding de una sola voz (para enroll): fuerza 1 hablante y devuelve su vector."""
    _, embeddings = diarize(wav_path, num_speakers=1)
    if not embeddings:
        raise ValueError(
            "no se pudo extraer un embedding de voz (audio muy corto o sin voz)"
        )
    return next(iter(embeddings.values()))
