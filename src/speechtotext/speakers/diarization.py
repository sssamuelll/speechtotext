"""Diarización batch: asignación por solape (pura) + pyannote (perezoso)."""
from __future__ import annotations

from speechtotext.core.segments import LabeledSegment, native_signals

# El checkpoint que se carga y, por lo mismo, el espacio vectorial de los embeddings que
# produce: el registro de voces los archiva y filtra por esta clave, no por dimensión.
# Un solo literal a propósito — con dos, quien cambie el checkpoint y olvide la clave deja
# los vectores ya registrados etiquetados con un espacio que dejó de ser el suyo.
EMBEDDING_MODEL = "pyannote/speaker-diarization-community-1"


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
        # Los N runs de un mismo segmento heredan el mismo src_dur (la extensión del
        # segmento que el ASR emitió): es la aproximación correcta —vienen de la misma
        # ventana de decodificación— y sin ella el gate de is_suspect se apaga bajo
        # --diarize, porque cada run se recomprime a la extensión de sus palabras. Las
        # señales nativas viajan igual: mismo origen, misma ventana.
        src_dur = s.end - s.start
        for w in words:
            spk = _best_speaker(w.start, w.end, turns)
            if not run_words:
                run_start, run_spk = w.start, spk
            elif spk is None or spk == run_spk:
                pass  # hereda: palabra sin turno o mismo hablante -> sigue el run
            elif run_spk is None:
                run_spk = spk  # el run venía sin hablante -> adopta el primero real
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


# --- Parte con modelos (pyannote 4.x). Imports perezosos a propósito: torch/pyannote
# pesan y este módulo debe poder importarse en el venv base (sin el extra [diarize])
# para usar las funciones puras de arriba. El pipeline community-1 devuelve, en una
# sola pasada, la diarización Y un embedding por hablante. La transcripción ya trae las
# muestras en memoria; `enroll` las lee de un wav con `wave`. ---

_PIPELINE = None
# El 32 con el que corre por defecto no es el default de pyannote (que es 1): sale del
# config.yaml del checkpoint community-1. A 32 el pico son 2620 MB; a 8, 1369 MB (-48%)
# por +7% de reloj sobre 180 s, con salida idéntica. En una máquina de escritorio el pico
# de RAM es el recurso escaso, no los 7 s.
# ponytail: constante, no opción de config. Techo: si algún día esto corre en GPU con VRAM
# de sobra, sube a parámetro.
_BATCH = 8


def read_wav(wav_path) -> tuple["np.ndarray", int]:
    """Lee un wav PCM16 a float32 mono en [-1, 1] con su tasa nativa. Es la entrada de
    `enroll` (una muestra de voz en disco); la transcripción ya trae las muestras."""
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
    """{waveform, sample_rate} para pyannote, en memoria (evita torchcodec)."""
    import numpy as np
    import torch

    data = np.ascontiguousarray(samples, dtype=np.float32)
    return {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sample_rate}


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        import os
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silencia el aviso de torchcodec al importar
            from pyannote.audio import Pipeline

        _PIPELINE = Pipeline.from_pretrained(
            EMBEDDING_MODEL, token=os.environ.get("HF_TOKEN")
        )
        # Ambos se leen en tiempo de llamada, así que asignarlos aquí basta.
        _PIPELINE.embedding_batch_size = _BATCH
        _PIPELINE.segmentation_batch_size = _BATCH
    return _PIPELINE


def diarize(samples, sample_rate: int, num_speakers: int | None = None):
    """Diariza muestras float32 mono. Devuelve (turns, embeddings).

    turns: list[(start, end, speaker_id)]. embeddings: dict[speaker_id, np.ndarray]
    (un vector por hablante, en el mismo espacio que embed_voice → comparables).
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
    """Embedding de una sola voz (para enroll): fuerza 1 hablante y devuelve su vector."""
    samples, sample_rate = read_wav(wav_path)
    _, embeddings = diarize(samples, sample_rate, num_speakers=1)
    if not embeddings:
        raise ValueError(
            "could not extract a voice embedding (audio too short or no speech)"
        )
    return next(iter(embeddings.values()))
