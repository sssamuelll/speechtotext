"""Manual diarization smoke test.

Usage: python scripts/smoke_diarize.py <audio> [num_speakers]

Transcodes the audio to 16 kHz mono, diarizes it, and prints the turns per
speaker. Requires the [diarize] extra installed, HF_TOKEN, and the gated
models accepted.
"""
import sys
from pathlib import Path

from speechtotext.core.audio import transcode_to_wav
from speechtotext.speakers.diarization import diarize, read_wav

num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
wav = transcode_to_wav(Path(sys.argv[1]).read_bytes())
try:
    samples, rate = read_wav(wav)
    turns, embeddings = diarize(samples, rate, num_speakers=num_speakers)
finally:
    wav.unlink(missing_ok=True)

for t0, t1, spk in turns:
    print(f"{t0:6.2f} - {t1:6.2f}  {spk}")
print(f"\n{len(embeddings)} speakers with embedding, {len(turns)} turns")
