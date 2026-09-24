# What the measurements say

Every default in this tool has a number behind it. This page is the
transcription benchmark those numbers come from: the recording, the method,
the full table, and what it cannot see. The speaker measurements are on
[the speakers page](speakers.md#two-diarizers), and the reasoning each number
led to is in [design.md](design.md).

## The recording and the metric

Fourteen minutes of a real meeting: two voices, Spanish from Spain and Spanish
from Venezuela, one microphone, technical jargon. Nine configurations over the
same audio. The metric is **25 concrete points**, not WER: points you had to be
able to write up without going back to the recording, each with its time window
and the words without which it makes no sense.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/benchmark-dark.svg">
  <img alt="Speed against errors per thousand words for six configurations: the large-v3 runs at the bottom, the small runs at the top" src="img/benchmark-light.svg" width="760">
</picture>

| Configuration | Points | Errors / 1000 words | Wall clock |
|---|---:|---:|---:|
| faster-whisper `large-v3`, no VAD, no chunking | **25 / 25** | **1.6** | 683 s |
| faster-whisper `large-v3` + VAD | 25 / 25 | 7.7 | 662 s |
| whisper.cpp `large-v3` CUDA | **25 / 25** | 6.0 | **109 s** |
| faster-whisper `large-v3` + hotwords | 24 / 25 | 3.8 | 667 s |
| faster-whisper `large-v3` chunked, with or without VAD | 24 / 25 | — | — |
| faster-whisper `large-v3` chunked + VAD + hotwords | 23 / 25 | — | ≈480 s |
| faster-whisper `small` | 22 / 25 | 29.1 | 135 s |
| whisper.cpp `small` CUDA | 21 / 25 | 27.5 | 55 s |

Errors are counted across the 66 places where the transcriptions disagree and the
answer is objective: a nonsense word that is not Spanish, a term, a number, a
confirmed omission. The other 100 places in disagreement (one demonstrative
swapped for another, filler words, commas) are left out on purpose. The chunked
configurations were compared in pairs against their unchunked twin, not in that
alignment. Wall clock on a Ryzen 9 5900X with a GTX 980, runs in series, model
load included.

## What this taught

- **`small` changes what was said.** Seventeen times more errors than
  `large-v3`, and they are not typos: *"todo se desordena"* ("everything falls
  out of order") came out as
  *"entonces ordenas"* ("then you tidy up"). <!-- # spanish-is-data: a quoted transcription is the evidence; translate it and the bullet proves nothing -->
  It saves nine minutes and costs three or four of the 25 points.
- **Hotwords fail in blocks.** With 4-5 terms, three blackouts of 28-30 s in two
  recordings — a whole window replaced by *"listo"* ("done") — and no
  improvement in the term they were meant to fix. n = 3, with no counterexample.
- **VAD deletes short sentences without saying so.** Four confirmed omissions
  and zero declared gaps: what it throws away before the model sees it leaves no
  hole in the timeline.
- **Chunking loses nothing at the seam.** It costs 2-3% of drift in every chunk
  after the first, because the 30 s windows end up shifted; that is where one
  point out of 25 fell.
- **whisper.cpp ties on points and loses on jargon.** Six times faster on the
  GTX 980 with 2 GB of VRAM, the same 25/25, and *Bézier* misspelled all ten
  times, measured on a single recording.

## What this does not prove

One recording, one domain, one machine. The reference was adjudicated by
whoever ran the benchmark, not by a human transcriptionist, and the omissions
were confirmed with a second Whisper configuration. This method cannot see an
error that every Whisper shares: *"clico la fecha"* for *flecha*, "date" where
the speaker said "arrow", in all nine. The recording is private and is not
published.

## Regenerating the chart

The two files under [`img/`](img/) are drawn from the table above by
`python scripts/benchmark_chart.py`; the light and dark variants share one
script and one set of numbers.
