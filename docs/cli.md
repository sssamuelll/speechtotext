# The command line

Every subcommand, every flag, and where things end up on disk. The
[README](../README.md) is the short version; this page is the whole one.
Speakers and names have [their own page](speakers.md), and the numbers behind
the defaults are in [benchmark.md](benchmark.md).

## The subcommands

| Command | What for |
|---|---|
| `transcribe` | Turn an audio file into `txt`/`srt`/`vtt`/`json`, with speakers if you ask for them. |
| `find` | Locate a topic inside a long recording without transcribing all of it. |
| `enroll` | Enroll a person's voice so that their name shows up. |
| `voices` | List the enrolled voices. |
| `forget` | Delete a voice from the registry. |
| `bench` | Measure which configuration suits your machine. |
| `probe` | Show what your machine has and which route `transcribe` would pick (paste it into an issue). |
| `models` | List, download (`pull`) and delete (`rm`) models. |
| `mcp` | Serve the tools over MCP on stdio, for a client such as Claude Desktop. |

`speechtotext <command> --help` prints the same tables as this page, one
command at a time.

---

## `transcribe`

```bash
speechtotext transcribe audio.wav
speechtotext transcribe talk.mp3 --model medium --language auto --formats txt,srt
speechtotext transcribe interview.m4a -o transcripts/ --device cuda
```

Anything ffmpeg can open goes in: audio or video, any container. Before it
starts, the CLI prints the duration and an ETA (estimated from a reference
table, or measured if you ran [`bench`](#bench)), then the progress in minutes
of audio, `mm:ss / mm:ss`. Redirected to a file, it prints one line per minute
of audio instead.

### Options

| Flag | Default | Description |
|---|---|---|
| `--language`, `-l` | `auto` | `auto` detects the language (well with ≥ 30 s of audio; if the probability comes back low, the CLI suggests fixing it) or an ISO-639-1 code (`es`, `en`, `de`, `fr`, …). |
| `--model`, `-m` | `large-v3` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`. `small` = fast draft. |
| `--formats`, `-f` | `txt,srt,json` | Any combination of `txt`, `srt`, `vtt`, `json`. |
| `--device`, `-d` | `auto` | `auto` probes the GPU (see [Engines](#engines)), `cpu`, `cuda`. |
| `--compute-type` | `auto` | `auto` picks `int8` on CPU and `float16` on GPU. |
| `--vad / --no-vad` | `--no-vad` | Filter for long silences. Off by default: measured, it drops short sentences without saying so. |
| `--beam-size` | `5` | Beam search size (minimum 1). |
| `--output`, `-o` | next to the audio | Output folder or base path. |
| `--engine` | `auto` | `auto` (based on the machine), `faster-whisper` or `whispercpp` (see [Engines](#engines)). |
| `--hotwords` | — | Terms the model should prefer, comma separated. |
| `--hotwords-file` | — | A file with those terms, one per line. |
| `--chunk / --no-chunk` | auto | Chunk the audio; automatic above 20 minutes. |
| `--jobs`, `-j` | `4` | Chunks transcribed in parallel. |
| `--diarize`, `-D` | off | Mark who is speaking (needs the `[diarize]` extra, or `[nemotron]` with `--diarizer nemotron`). |
| `--diarizer` | `pyannote`, or `SPEECHTOTEXT_DIARIZER` | `pyannote` or `nemotron`, about 30× faster on CPU (see [Two diarizers](speakers.md#two-diarizers)). |
| `--speakers` | auto | Number of speakers, as a hint (for example `2`); auto when omitted. |
| `--identify / --no-identify` | `--identify` | Put names to the voices enrolled with `enroll`. |
| `--threshold` | `0.5` | Voice match threshold (cosine, 0-1). |

### Which model

`large-v3` is the default: in int8 it runs on CPU at roughly 1.3× real time,
using about 3.5 GB of RAM, and it was the only one that lost nothing in
[what the measurements say](benchmark.md). `-m small` is the fast draft: five
times quicker, and it changes what was said. `tiny` and `base` are for testing
the pipeline, not for reading the result. On your machine:
[`probe`](#probe-and-models).

### Hotwords

```bash
speechtotext transcribe lecture.mp3 --hotwords "pyannote,diarization"
speechtotext transcribe lecture.mp3 --hotwords-file terms.txt
```

They bias the decoder toward terms the model does not know well: proper nouns,
jargon, acronyms. **Measured, they did damage**: with lists of 4-5 terms, three
blackouts of 28-30 s in two different recordings — a whole window replaced by
one word — and no improvement in the term they were meant to fix (see
[what the measurements say](benchmark.md)). If you use them, compare against a
run without them. The CLI warns at 10 terms or 300 characters, and
`faster-whisper` truncates silently around 223 tokens.

### Long audio

Above 20 minutes, `transcribe` chunks the audio by itself. It cuts at silences
(never mid-word), transcribes the chunks in parallel according to `--jobs`, and
saves each one under the data directory (see [Where things live](#where-things-live)).
If a run is interrupted, the next one resumes from the last finished chunk.

The checkpoint is by content: the key includes the file, the model, the engine,
the quantization, the device and the decoding flags. Changing any of those
invalidates the cache instead of reusing a result that does not match.

```bash
speechtotext transcribe podcast_3h.mp3 --jobs 6      # more parallelism
speechtotext transcribe interview.wav --no-chunk     # force a single pass
```

Chunking has a [measured](benchmark.md) price: nothing is lost at the seam, but
every chunk after the first decodes with its 30 s windows shifted and drifts
2-3% from the single pass.

Whisper pads the chunk's last window to 30 s and sometimes narrates over the
padding. A segment that runs past where the chunk really ended is dropped, or
trimmed back to it, so the invention never lands on top of the next chunk.
Chunking is not a reason to turn `--vad` on; the default holds here too.

### Output formats

`txt` is the flat transcription, `srt`/`vtt` are subtitles with times, and
`json` is the complete format: segments with times, detected language, gaps
without speech, and the engine's native signals (`no_speech`, `avg_logprob`,
`compression_ratio`) that let you decide whether a segment can be trusted.

The full schema, with which keys appear and when, is in
[`api.md`](api.md#json-schema). A segment marked `"suspect": true` is a
suggestion to review it, not a verdict.

With `--diarize`, `txt` prefixes each line with the speaker, `srt`/`vtt` do the
same inside the cue, and `json` gives every segment a `"speaker"` field plus a
top-level `"speakers"`. Without it, the output is unchanged.

---

## Engines

`--engine auto` (the default) probes the machine before loading anything, and
`speechtotext probe` shows the same thing the CLI sees. It picks:

| The machine | Route |
|---|---|
| NVIDIA GPU with ≥ 5 GB of VRAM free | `faster-whisper` · `cuda` · `float16` |
| NVIDIA GPU with 2-5 GB free, whisper.cpp installed (or Windows, where it is downloaded pinned) and model `large-v3` or `small` | `whispercpp` · `cuda` · `q5_0` |
| Anything else | `faster-whisper` · `cpu` · `int8` |

What you ask for explicitly (`--engine`, `-d`) is honored; the probe only fills
in what is missing, and it **never changes the model**: if `large-v3` does not
fit in RAM, it stops and suggests `-m small`. The thresholds were measured on a
single machine (5900X + GTX 980); `bench` is what tunes them, and its
`bench.json` overrides the estimated ETA.

`faster-whisper` is the normal path: CPU or CUDA, every flag honored.
`whispercpp` exists for old GPUs where CTranslate2 no longer performs. On
Windows it uses a binary pinned by SHA-256 that is downloaded and verified the
first time; on macOS and Linux it uses the `whisper-cli` on the `PATH`
(`brew install whisper-cpp`) and declares `device=native`, because the build
decides. The only things that change by themselves are the flags the engine
cannot honor:

| Flag | Under `whispercpp` |
|---|---|
| `--device` | Forced to `cuda` (the pinned binary is a CUDA build), with a notice. |
| `--compute-type` | `auto` resolves to `q5_0`; asking for `float16` explicitly is an error. |
| `--vad` | Turned off, with a notice. |
| `--jobs` | Forced to 1. |
| `--hotwords` | **Rejected**: the run does not start. |

`--hotwords` stops instead of degrading, on purpose: the knob was measured inert
under that engine, and pretending it had applied would be worse than saying so.

---

## `find`

Transcribing a long recording at full quality takes a while. If you only care
about one stretch (an interview, a talk), `find` locates it without transcribing
everything: it makes a fast pass with `tiny`, searches your query and returns
the **regions** where it appears. With `--extract` it also clips the chosen
stretch and transcribes it with the full model.

```bash
# locate: prints the regions (minutes) where the query appears
speechtotext find recording.mp3 "seismic vulnerability"

# extract: clips the densest region and transcribes it with the full model
speechtotext find recording.mp3 "seismic vulnerability" --extract

# pick another region, with diarization and names
speechtotext find recording.mp3 "interview" --extract --region 2 -D --speakers 4
```

| Flag | Default | Description |
|---|---|---|
| `--extract`, `-e` | off | Clip the chosen region and transcribe it with the full model. |
| `--region` | `1` | Which region to extract (1 = the densest). |
| `--model`, `-m` | `large-v3` | Model for transcribing the clip (small = fast draft). |
| `--scan-model` | `tiny` | Model for the fast indexing pass. |
| `--language`, `-l` | `auto` | Language for transcribing the clip. |
| `--formats`, `-f` | `txt,srt` | Output formats for the clip. |
| `--diarize`, `-D` | off | Diarize the extracted clip. |
| `--diarizer` | `pyannote`, or `SPEECHTOTEXT_DIARIZER` | `pyannote` or `nemotron` (see [Two diarizers](speakers.md#two-diarizers)). |
| `--speakers` | none | Number of speakers (a hint). |
| `--identify` / `--no-identify` | on | Name enrolled voices. |
| `--threshold` | `0.5` | Voice match threshold (cosine, 0-1). |
| `--context` | `10.0` | Seconds of margin around the clip. |
| `--output`, `-o` | next to the audio | Output folder for the clip. |
| `--rebuild` | off | Rebuild the index even if one exists. |
| `--top` | `5` | How many regions to list. |
| `--hotwords` | none | Hard terms for transcribing the clip (see `transcribe`). |
| `--hotwords-file` | none | Lexicon file for transcribing the clip (see `transcribe`). |

The first `find` on a file builds the index (slow, once); later searches on that
same file are instant. The index is kept under the data directory. Matching
ignores accents and case.

> `find --extract` transcribes with `device auto`, `compute-type auto`, VAD off
> and `beam-size 5` fixed. `--engine`, `--chunk` and `--jobs` aren't exposed
> either -- for those, extract first and then run `transcribe` on the clip.
> Everything else `find` accepts (`--model`, `--language`, `--formats`,
> `--diarize`, `--diarizer`, `--speakers`, `--identify`, `--threshold`,
> `--hotwords`) passes straight through.

---

## `bench`

```bash
speechtotext bench sample.wav              # measures the viable configurations
speechtotext bench sample.wav --quick      # fewer configurations
speechtotext bench sample.wav --seconds 90 # a longer stretch
speechtotext bench --show                  # repaints the last measurement
```

It measures, on **your** machine, the combinations of engine, model and
quantization your hardware can take, each one in an isolated subprocess, and
recommends one per use case (fast, balanced, quality). The result is kept in
`bench.json`, and `--show` repaints it without measuring again. From then on
the ETA that `transcribe` prints comes from that file, not from the estimate.

---

## `probe` and `models`

```bash
speechtotext probe                          # what the machine has and which route transcribe would pick
speechtotext models                         # installed models
speechtotext models pull large-v3           # download (announces the size first)
speechtotext models pull small --engine whispercpp
speechtotext models rm small
```

`probe` is what to paste into a bug report: it reads the machine in under a
second, loads nothing, and prints the route `transcribe` would take for
`large-v3` and for `small`, with the reason. The same API from Python:
`speechtotext.core.models` and `speechtotext.core.probe`
([contract](api.md#probe-and-models)).

---

## `enroll`, `voices` and `forget`

```bash
speechtotext enroll "Alice" alice_sample.wav   # >=10 s of a single clean voice
speechtotext voices                            # list the enrolled voices
speechtotext forget "Alice"                    # delete a voice
```

An enrolled voice gets its name in every transcript that runs with
`--diarize`. The whole story — the Hugging Face token, how voices are filed,
the two diarizers and their limits — is on [the speakers page](speakers.md).

---

## `mcp`

`speechtotext mcp` exposes four tools over stdio, for MCP clients such as Claude
Desktop. It needs the extra: `[mcp]`.

| Tool | What it does |
|---|---|
| `transcribe(path, language?, model?, diarize?, diarizer?)` | Transcribes and writes the JSON next to the audio; returns the text and the path. `diarizer` is `pyannote` (default) or `nemotron`. |
| `find(path, query)` | The regions of the audio where the query appears, without transcribing all of it. |
| `voices()` | The enrolled voices. |
| `probe()` | What the machine has and which route `transcribe` would pick. |

Client configuration (adjust the path to your environment's executable: on
Windows it is `...\Scripts\speechtotext.exe`, on macOS and Linux
`.../bin/speechtotext`):

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

The server prints nothing on its own: on stdio, stdout is the protocol. Long
transcriptions report no progress for that reason, and the client waits.

---

## Environment variables

| Variable | What it does |
|---|---|
| `SPEECHTOTEXT_HOME` | Moves the data directory (models under whisper.cpp, chunks, the `find` index, enrolled voices, `bench.json`) somewhere else. |
| `SPEECHTOTEXT_DIARIZER` | The CLI's default diarizer, `pyannote` or `nemotron`. `--diarizer` wins over it; `--speakers N` runs pyannote for that call and says so. Only the CLI reads it. |
| `HF_TOKEN` | The Hugging Face token pyannote's gated models need (see [speakers](speakers.md#requirements-once)). |
| `HF_HOME` | Hugging Face's own variable: where the `faster-whisper` and pyannote models are cached. |

## Where things live

| What | Where |
|---|---|
| `faster-whisper` models | The Hugging Face cache, `~/.cache/huggingface` unless `HF_HOME` says otherwise. |
| whisper.cpp binary and models | `%LOCALAPPDATA%\speechtotext` (Windows), `~/Library/Application Support/speechtotext` (macOS), `~/.local/share/speechtotext` (Linux). |
| Chunk checkpoints, the `find` index, enrolled voices, `bench.json` | `~/.speechtotext/` |

`SPEECHTOTEXT_HOME` overrides all of it if you set it.
