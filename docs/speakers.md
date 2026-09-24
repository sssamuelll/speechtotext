# Who said what

With the `[diarize]` extra, `speechtotext` marks **who said what** in a
conversation recording and, if you enroll the voices, puts **names** on them.
All local. The `[nemotron]` extra adds a second diarizer that is about thirty
times faster on CPU and gives no names; [the two are compared below](#two-diarizers).

## Requirements (once)

The default diarizer uses [pyannote](https://github.com/pyannote/pyannote-audio)
models that download from Hugging Face and are _gated_:

1. Create a **Read** token at https://huggingface.co/settings/tokens and export it:
   ```bash
   export HF_TOKEN=hf_your_token        # Windows: setx HF_TOKEN "hf_your_token"
   ```
2. Signed in to HF, accept access to the model at
   https://huggingface.co/pyannote/speaker-diarization-community-1
   (if pyannote asks you to accept a dependent model on first use, accept that one too).

The first run downloads the models to `~/.cache/huggingface`; after that they
stay cached. Nemotron needs neither the token nor the click: see
[Installing Nemotron](#installing-nemotron).

## Enrolling voices

```bash
speechtotext enroll "Alice" alice_sample.wav   # >=10 s of a single clean voice
speechtotext voices                            # list the enrolled voices
speechtotext forget "Alice"                    # delete a voice
```

Voices are stored in `~/.speechtotext/` (override with `SPEECHTOTEXT_HOME`).

Each voice is filed under the model that produced its embedding, and is compared
only against vectors from that same model: the cosine between two different
vector spaces means nothing. If you consume the registry as a library,
`registry.get_embeddings(model)` demands that model for exactly this reason, and
returns an empty dictionary rather than an error when that model has no enrolled
voices. The on-disk format is in [`api.md`](api.md#voice-registry).

## Transcribing with speakers

```bash
# anonymous: Speaker 1 / Speaker 2
speechtotext transcribe conversation.mp3 --diarize

# a hint of 2 speakers (better accuracy) + names from the enrolled voices
speechtotext transcribe conversation.mp3 --diarize --speakers 2

# stricter about putting names on
speechtotext transcribe call.m4a -D --threshold 0.6
```

Sample `txt` output:

```
Alice: Good morning, can you hear me?
Speaker 2: Loud and clear. Go ahead.
```

In `json` every segment gains a `"speaker"` field and there is a top-level
`"speakers"`; in `srt`/`vtt` the speaker prefixes each line. Without
`--diarize`, the output is unchanged.

## Two diarizers

`--diarizer pyannote`, the default, and `--diarizer nemotron`
([Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization))
answer the same question, who spoke when, at very different cost:

| | `pyannote` (community-1) | `nemotron` |
|---|---|---|
| 64 minutes of a call, CPU | 25 min | 55 s |
| Peak RAM of that step | 1.7-1.9 GB | 1.7 GB |
| `--speakers` | honored | an error: the run does not start |
| Names from `enroll` | yes | no: it gives no voice embeddings |
| Install | `[diarize]` + a Hugging Face token | `[nemotron]`, no token |

Measured on 2026-09-24 on a Ryzen 9 5900X (12 threads, CPU only), both over the
same transcription:

- **A 64-minute two-person video call in Spanish**, scored against the call
  platform's own speaker labels (it receives one audio stream per
  participant) over the 6,970 words both transcripts agree on. Words given to
  the wrong speaker: pyannote with `--speakers 2` 1.15%, Nemotron 1.55%,
  pyannote without the count 1.75%. Told the count, pyannote is the more
  accurate; without it the two are level.
- **Four AMI test meetings** (four speakers, English), diarization error rate
  with no collar: pyannote 20.1%, Nemotron 30.1% as it ships. Almost all of
  Nemotron's error is missed speech: it ends a turn at every pause, and AMI's
  references bridge short pauses. Merging a speaker's gaps under one second
  brings it to 17.4%, a value tuned on those same meetings. A transcript does
  not feel this, because a word that falls in a pause keeps the speaker of
  the words around it.

What this does not prove: one call and four meetings. The call counts only
words both transcripts agree on, so crosstalk is under-represented, and in
crosstalk a third of the words went to the wrong speaker under either
diarizer. Why the slow one is still the default is in
[design.md](design.md#two-diarizers-and-the-slow-one-is-still-the-default).

To make Nemotron the default on a machine, set `SPEECHTOTEXT_DIARIZER`:

```bash
export SPEECHTOTEXT_DIARIZER=nemotron     # Windows: setx SPEECHTOTEXT_DIARIZER nemotron
speechtotext transcribe call.mp3 -D       # now diarized by nemotron
speechtotext transcribe call.mp3 -D --speakers 2   # pyannote: only it takes a count
```

It is a default, not a request. `--diarizer` wins over it. `--speakers N`
runs pyannote for that call and prints a line saying so, because a default
never overrides something asked for explicitly. `--diarizer nemotron` together
with `--speakers` is still an error. Only the CLI reads the variable: the
library and the MCP tool use pyannote unless told otherwise.

Nemotron downloads about 400 MB once, from a pinned revision. It is not gated
and its license ([OpenMDW 1.1](https://openmdw.ai/license/1-1/)) allows
commercial use. Its speakers are numbered in the order they first speak.

### Installing Nemotron

The `[nemotron]` extra needs transformers 5.18, which ships the model. Until
that release is on PyPI, install transformers from git first, in the same
environment:

```bash
pip install "transformers @ git+https://github.com/huggingface/transformers@f324707307757d9c0b8dac1c4462eceff911fa2f"
pip install "speechtotext[nemotron] @ git+https://github.com/sssamuelll/speechtotext@v0.6.0"
```

Once 5.18 is released, the second line alone is enough.

## Limits

- With `faster-whisper`, words are attributed one by one, so a turn change
  inside a segment splits it. Under `whispercpp` there are no word timestamps
  and a whole segment goes to one speaker; on the call above that doubled the
  words given to the wrong speaker (2.3-2.5% against 1.2-1.8%).
- Identification depends on the quality of the enrollment and on `--threshold`;
  very similar voices can be confused.
- It works on CPU, but diarization adds time on top of the transcription.
- Today only pyannote produces embeddings, and it runs the whole diarization
  pipeline to do it: seconds per sample, not milliseconds. Good for batch, not
  for a live path.
