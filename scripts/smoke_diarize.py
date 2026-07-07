"""Smoke manual de diarización.

Uso: python scripts/smoke_diarize.py <audio> [num_speakers]

Transcodea el audio a 16 kHz mono, lo diariza e imprime los turnos por hablante.
Requiere el extra [diarize] instalado, HF_TOKEN y los modelos gated aceptados.
"""
import sys
from pathlib import Path

from speechtotext.core.audio import transcode_to_wav
from speechtotext.speakers.diarization import diarize

num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
wav = transcode_to_wav(Path(sys.argv[1]).read_bytes())
try:
    turns, embeddings = diarize(str(wav), num_speakers=num_speakers)
finally:
    wav.unlink(missing_ok=True)

for t0, t1, spk in turns:
    print(f"{t0:6.2f} - {t1:6.2f}  {spk}")
print(f"\n{len(embeddings)} hablantes con embedding, {len(turns)} turnos")
