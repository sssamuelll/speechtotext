# Why it is shaped this way

The README says what this tool does. This page says why it has the shape it
has.

None of what follows is taste. Every decision below was settled by a
measurement, by a bug that actually happened, or by a constraint nobody could
argue with, and the evidence sits next to the claim. Where a number appears it
came from one machine and one recording, and that limit is stated with it.

---

## One engine contract, under everything

For most of this project's life there were two ways in. The CLI built a
`WhisperModel` and passed dictionaries around. A typed `asr/` layer sat beside
it, written for a different consumer, and nothing in `core/` was allowed to
import it. When a second engine arrived, the work went around that wall instead
of through it: a factory and a capability dict landed in `core/`, because typing
the new engine then would have meant inventing fields whisper.cpp does not
produce, and lying with types is worse than lying with dicts.

The wall came down when the private half of the repository moved out. `asr/` is
now the only engine contract: one `AsrBackend` protocol, two implementations,
and a `Caps` record saying of each knob whether the engine will honor it,
degrade it, or reject it. A third engine is one class. Nothing above `asr/`
branches on which engine is underneath.

## `transcribe()` is a function

The thing with state is the model, and it already has an owner. The backend
object holds it, `warm()` loads it, and there is no registry and no singleton:
whoever wants the model hot between calls keeps the object and hands it back in
as `backend=`.

What is left is a pipeline. Probe, decode, plan the chunks, run them,
reassemble, diarize if asked. It holds nothing worth keeping between calls, so
it is a function. A short file and a long one are the same path with one chunk
instead of seven, which is how the CLI lost the second orchestration it used to
carry: two code paths that drifted apart every time somebody fixed one of them.

## `large-v3` is the default, and it costs five times the clock

On fourteen minutes of a real meeting, `large-v3` took 683 seconds and made 1.6
errors per thousand words. `small` took 135 seconds and made 29.1. The errors
are not typos. A sentence meaning "everything falls out of order" came back as
"then you tidy up": both are grammatical Spanish, both sit in the transcript
with the same authority, and only one of them was said.

A fast default that changes what was said is worse than a slow one, because the
reader cannot tell which sentences to distrust. `-m small` stays in the help as
a draft, described as what it is.

## VAD is off, and there is no dial for it

VAD deletes short sentences without saying so. In that same benchmark it caused
four confirmed omissions and declared zero gaps: what it throws away before the
model ever sees it leaves no hole in the timeline, so nothing downstream can
report it.

The dial does not exist either, and that was a decision against an earlier plan
of this project rather than an oversight. Silero over the same file found 87.0%
speech in 4.5 seconds of compute, sixty-seven seconds more than the decoder
turned into text. The VAD was running ahead of the decoder, not behind it.
Moving its threshold from 0.5 down to 0.2 buys 1.8% more speech, against a
measurement whose own bias is nine points (see the last section). An instrument
noisier than the effect it measures does not get built.

## Hotwords ship with the measurement against them

`--hotwords` is documented, wired, and measured to do damage. With lists of four
or five terms: three blackouts of 28 to 30 seconds across two different
recordings, a whole window replaced by a single word, and no improvement in the
term they were meant to fix. Three cases, no counterexample.

The mechanism explains it. Hotwords are not a bias over the vocabulary.
`faster-whisper` appends them to the prompt after `sot_prev`, the same channel
that carries the previous window's text, so a comma-separated list of proper
nouns reaches the decoder as the paragraph that came before. That is the exact
failure this CLI had already closed by setting
`condition_on_previous_text=False`, left open under a second name. Above 223
tokens the list is truncated with no warning.

None of this was theory here. The docstring in this repository said hotwords
were a probabilistic bias. Reading it, someone passed 25 terms, lost 9.1 points
of coverage and 13% of the words against passing nothing, and then spent an
afternoon building a four-pass pipeline to recover what the flag had cost. The
pipeline came back with fewer words than a bare run of the tool. The flag
stayed; the prose that lied about it was rewritten, and the CLI now warns at ten
terms.

## The probe fills in gaps; it never substitutes

`speechtotext probe` reads the machine before anything loads and picks engine,
device and quantization. What you asked for explicitly is honored. The probe
only fills in what you left on `auto`.

It never changes the model. If `large-v3` does not fit in RAM the run stops with
`insufficient_resources` and a message suggesting `-m small`. Quietly loading a
smaller model would produce a transcript that reads exactly like the one you
asked for and is not it.

The same rule governs the capability table. Under whisper.cpp, `--hotwords` is
rejected before anything is built rather than degraded with a notice: the
underlying `--prompt` is inert with `-mc 0`, bit-identical output across seven
runs with a positive control, and announcing a degradation over a knob that did
nothing would invent an effect that never happened. Degrade when the result is
still what was asked for with less precision. Reject when the knob would be
inert or would mean something else. Never silence, never a quiet substitution.

## A signal the engine does not emit is absent, not zero

whisper.cpp reports no `no_speech`, no `avg_logprob`, no language probability.
Those keys are left out of the JSON and out of the checkpoint. They never appear
as `null`, and never as `0.0`.

A zero is a measurement. Writing one where nothing was measured lets a consumer
average over invented numbers and never find out. The same rule caught an older
fabrication of our own: the CLI used to print "detected language" over a value
that came straight from the user's own `--language` flag. It now says whether
the language was measured or forced.

## Chunking cuts at silences, and the seam costs 2-3%

Above twenty minutes the audio is cut into chunks at silences, never mid-word,
and each chunk is checkpointed so that an interrupted run resumes.

Measured over fourteen minutes of real speech, nothing is lost at the seam: the
cut lands in a silence and the sentences on both sides arrive whole. The price
is elsewhere. Every chunk after the first decodes with its 30-second windows
shifted against the single pass and drifts 2-3%, and that is where one point out
of twenty-five fell.

The boundary has a second hazard, and it was a live bug. Whisper pads the last
window to 30 seconds and sometimes narrates over the padding: a video sign-off,
timestamped 598.6 to 628.6, on a chunk that ended at 598.7. Thirty seconds of
invention landing on top of the next chunk's real audio, formatted exactly like
speech. `clip_to_end` now drops a segment carrying more padding than audio and
trims one that barely protrudes, in global time and also when reading a
checkpoint written before the fix.

The checkpoint key is the whole identity of a run: file, size, mtime, engine,
model, quantization, device, and the effective request. Two engines over the
same audio never share a digest. `--vad` and `--no-vad` under whisper.cpp do
share one, because the effective request is identical.

## Two diarizers, and the slow one is still the default

In September 2026 a public leaderboard ranked pyannote's community-1, the
diarizer here, sixth of twelve at 30.6% DER, and NVIDIA's
Nemotron-3-Diarization first at 14.7%. That leaderboard is English, and it
never tells a system how many people are speaking. Measured on this
project's terms, most of the gap did not survive.

On a 64-minute two-person call in Spanish, scored word by word against the
call platform's own per-participant labels, pyannote told the count gave
1.15% of the words to the wrong speaker and Nemotron 1.55%. Not told the
count, pyannote gave 1.75%. For the recordings this tool was built for,
pyannote with `--speakers` is the more accurate of the two. On four AMI
meetings Nemotron as it ships scored worse, 30.1% against 20.1%, and the
cause was not the model. The transformers port hands out bare frames above
a threshold, so every pause ends a turn, and AMI's references bridge
pauses. Merging sub-second gaps took it to 17.4%.

What did survive is the clock: 55 seconds against 25 minutes for the same
hour of audio on the same CPU. That earns a flag, not the default, because
the default keeps the speaker count and the names, and because Nemotron's
install is a git pin until transformers 5.18 ships.

Wiring it settled three smaller things:

- **Its features are computed in pieces.** A single pass over 64 minutes
  peaked at 4.0 GB inside the STFT; in pieces the whole step peaks at 1.7
  GB. The pieces are allowed only because a test shows they change nothing
  the model sees, and on the real call the turns came out identical.
- **It checks its dependencies before the transcription starts.**
  Discovering a missing extra after an hour of ASR is the failure the
  pyannote path still has.
- **A count it cannot take is refused. Names it cannot give are a
  warning.** Under Nemotron, `--speakers` would be inert, and by the rule
  in the probe section an inert knob is rejected, so the run does not
  start. Names are different. Identification is on by default, so refusing
  would break every Nemotron run for anyone who ever enrolled a voice, and
  the transcript is still what was asked for without them. That one is
  degraded, and `warnings` says so.

## What a comparison against a commercial service settled

In July 2026 this tool was measured against a commercial transcription service
over the same audio, and lost on readability. The output had no punctuation, no
capitals, no question marks: walls of running text. Restoring punctuation and
truecasing was the largest item on the backlog, and it looked like a
post-processing project.

It was a model-size problem. The comparison had been running `small` against a
service that runs a large model. Re-running the same audio with `large-v3` and
no other change produced commas, periods, question marks, capitals and accents,
at parity or better. The biggest item on that backlog died without a line of
code, and that is the second reason `large-v3` is the default.

The finding that survived in the other direction: on regional proper nouns the
local model was already ahead of the commercial one, which mangled names the
local model got right. A large model on your own machine is not a downgrade from
a service, and the vocabulary nobody tunes for is where that shows.

## Known limits

- **Orphaned chunks are never collected.** Every change to the checkpoint key
  leaves the old checkpoints on disk under the data directory, forever. There is
  no `--prune`; deleting the folder is one command, and that is the whole story
  today.
- **Toggling `--diarize` invalidates every checkpoint.** `word_timestamps` is
  part of the effective request and therefore part of the key, so dropping a
  flag that does not change the text recomputes all of it. A checkpoint with
  words is a superset of one without, but nobody has confirmed that asking for
  words leaves the text untouched, and until somebody does, the key stays
  honest.
- **The coverage percentage is a hint, not a verdict.** Under VAD,
  `faster-whisper` remaps timestamps back onto the original axis, so a segment
  spanning two VAD regions carries the removed silence inside its span. One run
  yields three defensible numbers: 83% by spans as returned, 73% by spans with
  that silence taken out, 65% counted by words. Until the tool declares which
  quantity it means, no percentage it prints is evidence of anything.
- **The benchmark is one recording, one domain, one machine.** Two voices, one
  microphone, Spanish, technical jargon, a Ryzen 9 5900X with a GTX 980. The
  reference was adjudicated by whoever ran the benchmark. An error that every
  Whisper configuration makes is invisible to the method.
