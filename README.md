<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/hero-dark.svg">
    <img alt="speechtotext. The audio never leaves your machine." src="docs/img/hero-light.svg" width="1200">
  </picture>
</p>

<p align="center">
  Local speech to text for files, with who said what.<br>
  A Python library, a CLI and an MCP server over
  <a href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</a> and
  <a href="https://github.com/ggml-org/whisper.cpp">whisper.cpp</a>.
</p>

<p align="center">
  <a href="https://github.com/sssamuelll/speechtotext/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/sssamuelll/speechtotext/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/sssamuelll/speechtotext/releases"><img alt="release" src="https://img.shields.io/github/v/tag/sssamuelll/speechtotext?style=flat-square&label=release&sort=semver"></a>
  <img alt="python" src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsssamuelll%2Fspeechtotext%2Fmain%2Fpyproject.toml&style=flat-square">
  <img alt="platform" src="https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-555?style=flat-square">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/github/license/sssamuelll/speechtotext?style=flat-square"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="#measured-not-assumed">Measured</a> ·
  <a href="#from-python">Python</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="A terminal running speechtotext transcribe meeting.m4a --diarize --speakers 2: the route, the progress in minutes of audio, and the transcript with Speaker 1 and Speaker 2." src="docs/img/demo.gif" width="880">
  <br>
  <sub>Two synthetic voices from macOS text-to-speech. The run is real.</sub>
</p>

## Why it is local

This began as a script for a pile of interview recordings that could not leave
the machine they sat on. They were other people's words, recorded and not yet
published, and a transcription service would have meant handing them to a
stranger. So the transcription had to run where the audio already was.
Everything here follows from that: no API key, no per-minute meter, nothing
uploaded. Models download once; after that the machine works alone.

## Install

Python 3.11 or newer, and [ffmpeg](https://ffmpeg.org/) on the `PATH`
(`brew install ffmpeg`, `apt install ffmpeg`, or the official Windows build).
Then, as a tool:

```bash
uv tool install "speechtotext @ git+https://github.com/sssamuelll/speechtotext@v0.6.0"
# or: pipx install "speechtotext @ git+https://github.com/sssamuelll/speechtotext@v0.6.0"
```

Extras: `[diarize]` for speakers and names (pyannote + torch, about 2 GB),
`[nemotron]` for the [faster diarizer](docs/speakers.md#two-diarizers),
`[mcp]` for the server.

```bash
uv tool install "speechtotext[diarize] @ git+https://github.com/sssamuelll/speechtotext@v0.6.0"
```

The first `transcribe` downloads `large-v3`, about 3 GB, once;
`speechtotext models pull large-v3` does it ahead of time. Not on PyPI: the
name is taken there. A project that imports the library pins the same tag in
its requirements, and reads [`docs/api.md`](docs/api.md), the contract, and
[`CHANGELOG.md`](CHANGELOG.md), where every break is written down.

## Quick start

```bash
speechtotext transcribe meeting.m4a
```

```text
no usable GPU: CPU
Model large-v3 · engine faster-whisper · device cpu · compute int8
Duration 0.5 min · ETA ~1 min (estimated)
INFO:faster_whisper:Processing audio with duration 00:28.544
INFO:faster_whisper:Detected language 'en' with probability 0.99
  00:28 / 00:28 00:00-00:28 99% (new)
Language detected: en (prob=0.99) · duration 28.5s · 7 segments · speech 0.5 of 0.5 min (99%) · engine faster-whisper
no gaps of 5 s or more
  OK meeting.json
  OK meeting.srt
  OK meeting.txt
```

That is a MacBook Air with 8 GB of RAM: no GPU, so the probe picked the CPU
route, said what it would cost, and drew the progress in minutes of audio.
The same file with speakers, on a machine with the `[diarize]` extra:

```bash
speechtotext transcribe meeting.m4a --diarize --speakers 2
```

```text
Speaker 2: Morning. Did the overnight benchmark finish?
Speaker 1: It did. The large model lost nothing on the 25 points. The small one dropped 3.
Speaker 2: Then large stays the default. What did it cost?
Speaker 1: About 5 times the clock. 11 minutes for a 14 minute meeting. On the CPU.
```

Anything ffmpeg can open goes in, audio or video. Three more commands cover
most days:

```bash
speechtotext find lecture.mp3 "seismic vulnerability"   # where in three hours is this said
speechtotext enroll "Alice" alice.wav                    # from now on, Alice is Alice, not Speaker 1
speechtotext probe                                       # what your machine has and the route it gets
```

## What it does

| | | |
|---|---|---|
| **Transcribe** | `txt`, `srt`, `vtt` and `json`, with `large-v3` by default because `small` [changes what was said](#measured-not-assumed). | [cli.md](docs/cli.md#transcribe) |
| **Who said what** | Diarization marks the turns; a voice you enrolled once gets its name. Two diarizers, one 30× faster. | [speakers.md](docs/speakers.md) |
| **Find a stretch** | A fast pass with `tiny` indexes the recording; then the full model transcribes only the region you want. | [cli.md](docs/cli.md#find) |
| **Pick the route** | The probe reads the machine before loading anything: GPU with room, small GPU, or CPU. It fills in what you left on `auto` and never swaps the model in silence. | [Engines](#engines) |
| **Long audio** | Above 20 minutes it chunks at silences, runs the chunks in parallel and checkpoints each one; an interrupted run resumes. | [cli.md](docs/cli.md#long-audio) |
| **Say what not to trust** | The JSON carries the engine's own signals, `no_speech`, `avg_logprob`, `compression_ratio`, and marks a segment `suspect` when they say so. A suggestion to review, not a verdict. | [api.md](docs/api.md#json-schema) |
| **Serve it** | Four tools over MCP on stdio, for a client such as Claude Desktop. | [cli.md](docs/cli.md#mcp) |

What it deliberately does not do:

- **No cloud.** Models download once; after that the machine works alone.
- **No live path.** It reads files. There is no microphone, no streaming endpointer.
- **No verdicts.** It measures the signal and flags a segment for review. It never decides a recording is good enough.
- **No editing on top.** No summaries, no translation, no clean-up pass. The text is what the engine said.
- **No application.** No GUI, no project files. A program that needs those imports this one.

## Measured, not assumed

Every default has a number behind it. Fourteen minutes of a real meeting, two
voices, one microphone, technical jargon, nine configurations. The metric is
25 concrete points a reader had to be able to write up without going back to
the recording, not WER.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/benchmark-dark.svg">
  <img alt="Speed against errors per thousand words for six configurations: the large-v3 runs at the bottom, the small runs at the top" src="docs/img/benchmark-light.svg" width="760">
</picture>

| Configuration | Points | Errors / 1000 words | Wall clock |
|---|---:|---:|---:|
| faster-whisper `large-v3`, no VAD, no chunking | **25 / 25** | **1.6** | 683 s |
| faster-whisper `large-v3` + VAD | 25 / 25 | 7.7 | 662 s |
| whisper.cpp `large-v3` CUDA | **25 / 25** | 6.0 | **109 s** |
| faster-whisper `large-v3` + hotwords | 24 / 25 | 3.8 | 667 s |
| faster-whisper `small` | 22 / 25 | 29.1 | 135 s |

- **`small` changes what was said.** Seventeen times more errors than `large-v3`, and they are not typos: a sentence meaning "everything falls out of order" came back as "then you tidy up". It saves nine minutes and costs three or four of the 25 points. So `large-v3` is the default.
- **VAD deletes short sentences without saying so.** Four confirmed omissions and zero declared gaps. So it is off.
- **Hotwords fail in blocks.** Three blackouts of 28-30 s across two recordings and no gain in the term they were meant to fix. The flag ships with that warning on it.
- **Chunking loses nothing at the seam** and costs 2-3% of drift in every chunk after the first.
- **whisper.cpp ties on points and loses on jargon**, six times faster on a GTX 980.

One recording, one domain, one machine; the reference was adjudicated by whoever
ran the benchmark. The method, the full table and what it cannot see are in
[benchmark.md](docs/benchmark.md); why each number became a default is in
[design.md](docs/design.md).

## Engines

`--engine auto`, the default, probes the machine before loading anything and
picks:

| The machine | Route |
|---|---|
| NVIDIA GPU with 5 GB of VRAM free or more | `faster-whisper` · `cuda` · `float16` |
| NVIDIA GPU with 2-5 GB free and whisper.cpp at hand (on Windows it is downloaded, pinned by SHA-256) | `whispercpp` · `cuda` · `q5_0` |
| Anything else | `faster-whisper` · `cpu` · `int8` |

What you ask for explicitly is honored; the probe only fills in what is missing,
and it never changes the model. If `large-v3` does not fit in RAM, it stops and
suggests `-m small` rather than loading something else quietly. Under
whisper.cpp the flags the engine cannot honor are degraded with a notice or
rejected before the run starts; the table is in
[cli.md](docs/cli.md#engines).

## Speakers

```bash
speechtotext transcribe call.m4a --diarize                 # Speaker 1, Speaker 2, ...
speechtotext transcribe call.m4a --diarize --speakers 2    # a count, when you know it: measured, more accurate
speechtotext enroll "Alice" alice.wav                      # ten seconds of one clean voice, once
```

| | `pyannote`, the default | `nemotron` |
|---|---|---|
| 64 minutes of a call, CPU | 25 min | 55 s |
| Takes `--speakers` | yes | no |
| Names from `enroll` | yes | no |
| Needs | `[diarize]` and a Hugging Face token | `[nemotron]` |

On a 64-minute two-person call, pyannote told the count gave 1.15% of the words
to the wrong speaker, Nemotron 1.55%, pyannote without the count 1.75%. The
slow one stays the default because it keeps the count and the names; a machine
that prefers the clock sets `SPEECHTOTEXT_DIARIZER=nemotron`. The token, the
registry and the limits are in [speakers.md](docs/speakers.md).

## From Python

```python
from pathlib import Path

from speechtotext.core.formats import write_srt
from speechtotext.core.transcribe import transcribe

t = transcribe(
    "meeting.m4a",
    diarize=True,
    speakers=2,
    on_progress=lambda p: print(p.stage, p.done, p.total, p.detail),
)
for s in t.segments:
    print(f"{s.start:6.1f}  {s.speaker or '-'}: {s.text}")
write_srt(t.segments, Path("meeting.srt"))
```

A file or 16 kHz mono samples go in; a `Transcript` comes out, with the
segments, the language and its probability, the gaps, the engine that ran and
every warning. `on_segment` receives each segment as the engine produces it, a
live preview; `cancel` is a `threading.Event` that stops the run inside a
chunk; `backend=` keeps a warm model across calls. The core never prints,
which is what lets the same function sit under the CLI, the MCP server and a
desktop app. The contract, with every type and error code, is
[`docs/api.md`](docs/api.md#transcribe).

## MCP

`speechtotext mcp` serves `transcribe`, `find`, `voices` and `probe` over
stdio, with the `[mcp]` extra. One block in the client's configuration:

```json
{
  "mcpServers": {
    "speechtotext": {
      "command": "/path/to/your/venv/bin/speechtotext",
      "args": ["mcp"]
    }
  }
}
```

On Windows the executable is `...\Scripts\speechtotext.exe`. The tools and
their arguments are in [cli.md](docs/cli.md#mcp).

## Documentation

| Document | What is in it |
|---|---|
| [`docs/cli.md`](docs/cli.md) | Every subcommand and flag, the engines, long audio, output formats, environment variables, where things live. |
| [`docs/speakers.md`](docs/speakers.md) | Who said what: the token, enrolling, the two diarizers with their measurement, the limits. |
| [`docs/benchmark.md`](docs/benchmark.md) | The transcription benchmark: method, full table, what it does not prove. |
| [`docs/api.md`](docs/api.md) | The contract for consumers: JSON schema, public types, guarantees. |
| [`docs/design.md`](docs/design.md) | Why the product has this shape, with the measurement behind each decision. |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed between tags, and what breaks. |

## Contributing

`pip install -e ".[dev]"` and `pytest -q`; the suite runs without a GPU,
without network and without models, and CI runs it on Linux, macOS and Windows.
A bug report needs the command, what came out, and the output of
`speechtotext probe`. The layout of the code and the rules a change has to meet
are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[pyannote](https://github.com/pyannote/pyannote-audio) and
[Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization).
