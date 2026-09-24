# Contributing

Thanks for looking. This page says how the tree is laid out, what the tests
guard, and what a change needs before it merges.

## Set up

```bash
git clone https://github.com/sssamuelll/speechtotext
cd speechtotext
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

The whole suite runs without a GPU, without network and without models:
`tests/conftest.py` gives every test a fixed probed machine and a temporary
`SPEECHTOTEXT_HOME`. It takes about a minute. CI runs the same suite on Linux,
macOS and Windows against Python 3.11 and 3.14, then builds the wheel and
installs it in a clean venv to check that the package works outside this tree.
The local gate is still `pytest -q`.

The `[diarize]`, `[nemotron]` and `[mcp]` extras are not needed to run the
tests; the tests that touch them seed a stand-in. Install them when you change
what they cover.

## How the code is laid out

```
src/speechtotext/
├── core/                 the transcription path
│   ├── transcribe.py     file → Transcript: route, single decode, chunks, progress
│   ├── enginepin.py      SHA-256 download and verification of the binary and the ggml
│   ├── chunked.py        chunking at silences, checkpoint and parallelism
│   ├── finder.py         fast index and region search (subcommand find)
│   ├── benchmark.py      measurement of configurations (subcommand bench)
│   ├── probe.py          machine probe and route choice (subcommand probe)
│   ├── models.py         models: where they live, list, pull, remove (subcommand models)
│   ├── formats.py        txt/srt/vtt/json writers + is_suspect + gaps
│   ├── segments.py       LabeledSegment and reading of native signals
│   ├── audio.py          transcode_to_wav() + typed errors
│   └── postprocess.py    normalization of clock times in the text
├── speakers/             diarization and identification (extras [diarize] and [nemotron])
│   ├── diarization.py    pyannote + assignment by overlap + embed_voice
│   ├── nemotron.py       Nemotron-3-Diarization, in pieces that fit in RAM
│   ├── identify.py       cosine + assign_names (a name per voice)
│   └── registry.py       voice registry, filed by model
├── audio/                measurements over the signal
│   ├── evidence.py       voice evidence by deterministic DSP
│   ├── quality.py        RMS, SNR, clipping, noise floor
│   ├── gate.py           pre-inference eligibility against thresholds
│   ├── io.py             decoding to mono float32
│   ├── level.py          fixed gain with a limiter
│   ├── fingerprint.py    cryptographic fingerprint of an audio pipeline
│   └── types.py          immutable domain models
├── asr/                  the engine contract and its two backends
│   ├── base.py           AsrBackend, Caps, AsrError
│   ├── types.py          TranscriptionRequest / TranscriptionResult
│   ├── faster_whisper.py FasterWhisperBackend
│   └── whispercpp.py     WhisperCppBackend (subprocess over a pinned whisper-cli)
└── cli/                  the user-facing surface
    ├── app.py            typer: transcribe / find / enroll / voices / forget / bench / probe / models / mcp
    └── mcp_server.py     the four MCP tools and the stdio server
```

Two rules shape all of it. The core never prints: the CLI paints from a
callback, the MCP server ignores it, and a desktop app draws a bar. And
nothing is substituted in silence: a flag the engine cannot honor is degraded
with a notice or rejected before the run starts, and a smaller model is never
loaded quietly in place of the one you asked for.
[design.md](docs/design.md) has the measurement behind each of those.

## What the tests guard

Three tests read the tree rather than the code, and a change that trips one
is usually a change to make on purpose:

- `tests/test_api_contract.py`: every name in [`docs/api.md`](docs/api.md)
  exists and imports, and every public name is in `api.md`. A change to that
  file is a change to the contract, and gets a line in
  [`CHANGELOG.md`](CHANGELOG.md), under *breaking* if it breaks.
- `tests/test_public_tree.py`: the public tree speaks English. A line that
  has to keep Spanish, because translating it would break a test or a
  migration, marks itself with `# spanish-is-data: <reason>`.
- `tests/test_docs.py`: every relative link and anchor in the README and
  `docs/` resolves, and every `python` block there imports names that exist.

## Adding an output format

1. Add `write_xxx(segments, path)` in `src/speechtotext/core/formats.py`.
2. Add `"xxx"` to `VALID_FORMATS` and to the `writers` table in
   `src/speechtotext/cli/app.py`.
3. Document the keys it writes in [`docs/api.md`](docs/api.md#json-schema) if
   it carries anything the others do not.

## Reporting a bug

Open an issue with the command you ran, what came out, and the output of
`speechtotext probe`: it says what your machine has and which route the CLI
took, and it is the first thing needed to reproduce anything. The issue form
asks for exactly that. Never attach a recording you would not publish; a
synthetic clip that reproduces the problem is worth more than a description.

## Pull requests

- Write the failing test first, then the change. The suite is the gate, and a
  claim about behavior that no test exercises is a claim nobody can check.
- Say the number. A default, a threshold or a flag ships with the measurement
  behind it and the limit of that measurement next to it; "faster" and
  "better" without a number do not merge.
- Keep the public tree in English, and keep real people out of it: no names,
  no private recordings, no machine paths in fixtures.
- A change to `docs/api.md` gets its CHANGELOG line in the same PR.
- One PR, one change. A refactor that rides along with a fix makes both
  harder to review and to revert.
