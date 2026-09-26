# Changelog

Consumers pin a tag, so this file exists for one question only: **is it
worth bumping the pin, and what breaks if I do?**

Anything marked **breaking** requires changes in the code that consumes the
library.

---

## Unreleased

---

## v0.6.0 — 2026-09-26

### Added

- `LICENSE`: the MIT text `pyproject.toml` had declared since 0.4.0 without
  the file. With it, `CONTRIBUTING.md`, `SECURITY.md`, a code of conduct,
  issue forms that ask for `speechtotext probe`, and a PR template.
- `transcribe(on_segment=...)` receives each `PartialSegment(start, end, text)`
  as the engine produces it, in seconds of the whole recording and before
  diarization: a live preview. Audio read from a checkpoint produces none.
- **New contract**: `core.transcribe.PartialSegment`.
- A second diarizer: `--diarizer nemotron` (and `transcribe(diarizer="nemotron")`)
  runs NVIDIA's Nemotron-3-Diarization, behind the new `[nemotron]` extra. On
  CPU it diarized 64 minutes in 55 s where pyannote took 25 minutes, with the
  same share of misattributed words when neither is told the speaker count.
  It counts speakers itself, so `--speakers` with it is an error and the run
  does not start. It gives no embeddings, so enrolled voices are not named,
  and `warnings` says so. A missing dependency also stops the run before the
  transcription starts. pyannote stays the default. Until transformers 5.18
  is released, the extra needs transformers installed from git (docs/speakers.md).
- The MCP `transcribe` tool gains `diarizer`, with the same two values.
- `SPEECHTOTEXT_DIARIZER` sets the CLI's default diarizer (`transcribe` and
  `find`). It is a default: `--diarizer` wins over it, and `--speakers N`
  runs pyannote for that call and prints a line saying so. The library and the
  MCP tool do not read it.
- **New contract**: `speakers.nemotron.diarize` and `speakers.nemotron.missing`.
- The JSON `engine` block gains `diarizer`, the id of the model that drew the
  turns, whenever there was diarization.
- `core.transcribe.transcribe()`: a file (or samples) goes in, a `Transcript`
  comes out, with progress by callback and cancellation. A single audio
  decode; the short file and the long one travel the same path. Contract in
  [`docs/api.md`](docs/api.md#transcribe).
- `WhisperCppBackend` implements the same contract as `FasterWhisperBackend`;
  the CLI, `bench` and `find` all build engines through one single place.
- `core.probe`: `machine()` probes the machine without loading models
  (`Machine`), and `choose_route()` picks engine/device/compute_type using
  the spec's table (`Route`, with `eta_factor` and `estimated`).
  `speechtotext probe` prints it.
- `core.models`: `data_dir()`, `installed()`, `ensure()`, `remove()`,
  `remote_size()`. `speechtotext models [pull|rm]` sits on top.
- The CLI prints the duration and an ETA before transcribing, and suggests
  pinning `-l` when detection comes out with probability < 0.5.
- `transcribe()` accepts the path as `str`; the engine's warnings
  (`empty_transcript`) land in `Transcript.warnings` with the engine's
  prefix.
- whisper.cpp on macOS/Linux: `ensure_engine()` uses the `whisper-cli` from
  `PATH`.
- `speechtotext mcp`: an MCP server over stdio with four tools (`transcribe`,
  `find`, `voices`, `probe`), behind the optional `[mcp]` extra. The core
  gains no dependencies.
- `docs/api.md` documents the entire `audio/` layer (20 symbols from
  `audio.__all__`, up from 3) and the `asr/` types that were missing.
  `tests/test_api_contract.py` watches it in both directions: nothing can be
  exported without being documented, and nothing documented without
  existing.
- **New contract** in `docs/api.md`: `speakers.diarization.diarize` and
  `embed_voice` (the "Diarizing" section), and `core.segments.native_signals`.
  They were already in the tree; what's new is that they are now watched
  contract, so renaming or moving them **breaks** and has to be recorded
  here.

### Changed

- The README is a front page. The reference moved whole into `docs/cli.md`
  (every flag, the engines, long audio, formats, environment variables),
  `docs/speakers.md` and `docs/benchmark.md`; the package layout into
  `CONTRIBUTING.md`. `tests/test_docs.py` keeps every link and anchor
  between them resolving, and every `python` block importing names that
  exist. The signature blocks in `docs/api.md` now parse as Python.
- The CLI shows progress in minutes of audio, `mm:ss / mm:ss`. Redirected to a
  file, it prints one line per minute of audio instead of one per chunk.
- whisper.cpp: a run that outlives its timeout is killed and raised as a
  `RuntimeError` — `backend_failed`, naming the chunk, in a chunked run —
  instead of escaping as a raw `subprocess.TimeoutExpired`.
- CI actually turned on: the suite runs on Linux, macOS and Windows against
  Python 3.11 and 3.14, and a separate job builds the wheel and installs it
  in a clean venv. Before, the workflow existed but installed an extra
  (`[evaluation]`) that no longer exists.
- `tests/fixtures/whispercpp_ojf.json` becomes synthetic: same shape as
  `-ojf` output, with no real human speech and no machine paths.

### Changed — breaking

- `Progress("transcribe")` counts seconds of audio, not chunks. `done` grows
  segment by segment, summed across the chunks running in parallel, and
  `total` is the duration; before, a file under 20 minutes jumped from 0 to 1.
  A finished chunk's `detail` ends in `(new)` instead of `(nuevo)`. <!-- # spanish-is-data: the old detail string being replaced -->
  `on_progress` may now be called from worker threads, one call at a time.
- `AsrBackend.transcribe` takes two keyword arguments, `on_segment` and
  `cancel`, and `transcribe()` passes both: a backend of your own has to accept
  them, and let exceptions from `on_segment` propagate unwrapped. The two
  built-in engines honor them between segments, so `cancel` now stops
  `transcribe()` inside a chunk, not only between chunks. Callbacks run on
  worker threads, one at a time.
- **The project's public surface moved wholesale, Spanish to English.** Every
  identifier, error message, warning, doc and default reachable through the
  library's public API now speaks English. The concrete breaks below come out
  of that move; if none of them touch your integration, bumping the pin is
  otherwise safe.
- `Cap` values are English now — `honored` / `degraded` / `rejected`, not
  `honrado` / `degradado` / `rechazado`. <!-- # spanish-is-data: the old Spanish enum values being replaced; translating them erases what changed -->
  Code comparing `caps.vad == "degradado"` stops matching. <!-- # spanish-is-data: the exact old comparison that stops matching; the string is the evidence -->
- The unnamed-speaker fallback in `txt` output is `Speaker ?` instead of `Hablante ?`, <!-- # spanish-is-data: the old placeholder string being replaced; translating it erases what changed -->
  and `humanize_speaker()` — which every output format uses once diarization hands it a numbered speaker — now emits `Speaker 1` rather than `Hablante 1`. <!-- # spanish-is-data: the old placeholder string being replaced; translating it erases what changed -->
  Both land in files a user opens.
- Outside Windows, whisper.cpp's `engine_version` in the JSON reads
  `"whisper.cpp (PATH, unpinned)"` instead of `"whisper.cpp (PATH, sin
  pin)"`. Same change as the rest of this list: a consumer matching the old
  Spanish string stops matching.
- `bench.json`'s `recommendations` block uses English keys (`case`,
  `description`, `requirements`, `criterion`, `choice`, `reason`) and
  English enum values. Nothing to migrate by hand: `read_table()` recognizes
  a file written with the old Spanish keys and regenerates the block on
  read, without re-measuring anything.
- `Transcript.warnings` now carries English prose. The machine codes in
  that same list (`empty_transcript`, `language_mismatch`,
  `no_speech_regions`, `gain_limited`) did not change — only the sentences
  around them did.
- **The evaluation harness and the chain of custody were extracted**:
  `evaluation/`, `security/`, `models/`, `confidence/`, `statistics.py` and
  the `[evaluation]` extra. They live in their one consumer now, the same
  as `api/` did in `0.4.0`. This library is left as what it is: audio →
  text, with or without speakers, and the measurements on the signal.
- `FasterWhisperBackend(model, config=None, *, model_version="unpinned")`
  now takes a name or a path, not a `VerifiedModelArtifact`. The
  `CalibratedAsrBackend` and `VerifiedLocalAsrBackend` Protocols left with
  the verification.
- `TranscriptionResult` no longer carries `confidence_target`,
  `calibrated_confidence` or `calibrator_version`, and `ConfidenceTarget`
  is gone. Whoever calibrates wraps the result.
- `PipelineProvenance` now takes `ModelRef(model_id, fingerprint)` in
  `models=`, not verified artifacts. The fingerprints do not change: the
  payload only ever stored `model_fingerprints`.

  Migration: import what was extracted from its new package; wrap
  `FasterWhisperBackend` with your own verification, passing
  `model=artifact.root` and `model_version=artifact.manifest.revision`;
  convert each artifact to `ModelRef(manifest.model_id, artifact.fingerprint)`
  before deriving provenance. With a path, the result's `model` becomes the
  directory name (`Path.name`), not the manifest's `model_id`; if your
  wrapper needs to keep that id, override the backend's `model_id`
  property.
- **`AsrBackend.transcribe` now takes samples, not an `AudioClip`**: float32
  mono at 16 kHz. Whoever has a clip passes `clip.view("asr").samples`. The
  Protocol also declares `caps`, `engine_version`, `quant` and `device`.
- `TranscriptionRequest` gains `vad: bool = False`; it enters the
  `fingerprint`, so request fingerprints change once. `language="auto"` is
  valid.
- The text of `TranscriptionSegment` and `TranscriptionWord` is now
  returned raw (with the engine's leading space); `result.text` is still
  trimmed.
- `speakers.diarization.diarize(samples, sample_rate, num_speakers=None)`
  now takes samples; `read_wav(path)` reads them from a wav.
  `embed_voice(wav_path)` does not change.
- `core.chunked`: `chunk_path(identity, start, end)`; `run_chunked`,
  `transcribe_chunk` and `probe_duration` are gone. Old checkpoints stop
  matching and get recomputed.
- `core/engines.py` is gone: the whisper.cpp adapter is
  `asr.whispercpp.WhisperCppBackend`, and the CAPS live in each backend.
- **CLI defaults:** `-m large-v3`, `--no-vad`, `-l auto`, `--engine auto`,
  `-d auto` (previously `small`, `--vad`, `es`, `faster-whisper`, `cpu`).
  Whoever relied on the default now passes it explicitly. Same for `find`
  (`large-v3`, `auto`).
- `Route` now lives in `core.probe` (still importable from
  `core.transcribe`) and gains `eta_factor` and `estimated`;
  `resolve_route()` now probes the machine and can raise
  `AsrError("insufficient_resources")` when the model does not fit in RAM.
- `Progress`: new `download` stage; with a file, `decode` is now emitted
  twice (before, indeterminate; after, `done = total = duration`).
- `enginepin.install_root()` now hangs off `models.data_dir()`: on Windows
  it is the same path as always; on macOS/Linux,
  `~/Library/Application Support/speechtotext` and
  `~/.local/share/speechtotext`. `SPEECHTOTEXT_HOME` wins if it is set.
- `benchmark._nvidia_smi/_ram_gb/_pinned_exe/machine_info` moved to
  `core.probe` (`machine_info()` stays in `benchmark` as an adapter for
  the v1 schema).
- `WhisperCppBackend.device` is `"native"` and `engine_version` is
  `"whisper.cpp (PATH, unpinned)"` outside Windows.
- `engine.selection` in the JSON is `"auto"` when the engine was chosen by
  the probe (previously always `"explicit"`); `transcribe()` accepts
  `route=` so it doesn't have to probe twice.

---

## v0.5.1 — 2026-09-12

### Added

- The engine's native signals reach the JSON: every segment can carry
  `no_speech`, `avg_logprob` and `compression_ratio` (#23). A non-finite
  value is treated as absence of measurement and the key is omitted,
  instead of writing `NaN`, which is not even valid JSON. Contract in
  [`docs/api.md`](docs/api.md#json-schema).
- Contract documentation for consumers (`docs/api.md`) and this changelog.

### Fixed

- With `--chunk`, Whisper could emit a segment over the zero-padding of a
  chunk's last window (a YouTube outro with a 30 s false mark), and that
  segment landed on top of the next chunk. Now it is clipped to the
  chunk's real end, including when reading old checkpoints that carry it.
- `speechtotext.__version__` had said `0.3.0` for two tags, and the
  package's docstring still advertised a pronunciation HTTP service that
  left in `0.4.0`.

---

## v0.5.0 — 2026-09-11

### Added

- **Voice evidence via deterministic DSP** (#22): `compute_voice_evidence`
  measures energy in the voice band, ratio of voiced frames, median F0 and
  spectral flatness. These are measurements, not verdicts — the module
  deliberately does not classify voice against non-voice, and it
  distinguishes "could not measure" (`None`) from "measured, and there is
  none" (`0.0`). Contract in [`docs/api.md`](docs/api.md#voice-evidence).

### Changed — breaking

- **The voice registry files and filters by model.** `registry.enroll` now
  requires `model`, and `registry.get_embeddings(model)` requires the
  model as a mandatory argument: the cosine between two different vector
  spaces means nothing, so a voice is only ever compared against vectors
  from the same extractor.

  Migration: the flat manifests from `0.4.x` are still readable — they are
  regrouped in memory on load, and the file on disk is rewritten to the
  new format on the next `enroll` or `remove`. What you do have to touch
  is the code: any `get_embeddings()` without an argument now raises
  `TypeError`.

  Watch out for the silent-failure mode: asking for a model with no
  enrolled voices returns `{}`, not an error. A misspelled name looks
  exactly like an empty registry.

---

## v0.4.0 — 2026-08-07

### Changed — breaking

- **The pronunciation-evaluation HTTP service was extracted** (FastAPI +
  Azure). It used to live in `api/`, and today it lives adapted in its one
  consumer. This library ended up as what it is: local speech to text, no
  server and no cloud dependencies.
- From here on the repo is consumed as a **library versioned by tags**.
  Consumers pin a tag or a SHA, never `@main`: a change a consumer needs
  goes in through a PR here, gets tagged, and only then does the pin move
  there.
