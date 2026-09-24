# Contract for consumers

What an app that depends on `speechtotext` can take as guaranteed, and what it
can't. If anything in this document changes, the change gets a line in
[`CHANGELOG.md`](../CHANGELOG.md) and ships in a new tag — nobody pins `@main`.

What isn't here is implementation: it can change without notice.

---

## JSON schema

Produced by `speechtotext transcribe -f json` and written by
`core.formats.write_json`. Optional keys **are omitted** when there is no
value; none of them ever appears as `null`.

```json
{
  "language": "en",
  "language_probability": 0.9987,
  "duration": 1843.2,
  "speech_s": 1502.7,
  "gaps": [[312.4, 340.1], [905.0, 913.8]],
  "speakers": ["Alice", "Speaker 2"],
  "engine": {
    "name": "faster-whisper",
    "version": "1.2.0",
    "model": "small",
    "quant": "int8",
    "device": "cpu",
    "selection": "explicit",
    "diarization": "segment",
    "diarizer": "pyannote/speaker-diarization-community-1"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.42,
      "text": "Hi, how are you?",
      "speaker": "Alice",
      "no_speech": 0.0142,
      "avg_logprob": -0.1877,
      "compression_ratio": 1.2044,
      "suspect": true
    }
  ]
}
```

### Top level

| Key | Type | Presence |
|---|---|---|
| `language` | `str` | Always. |
| `language_probability` | `float` | Omitted if the engine does not report it (happens with `--language auto` on the chunked route). |
| `duration` | `float` | Always. Duration of the file, in seconds. |
| `speech_s` | `float` | Always. Sum of the duration of the segments with speech. |
| `gaps` | `[[float, float], …]` | Always. Gaps with no speech of 5 s or more, as `[start, end]` pairs. |
| `speakers` | `[str, …]` | Only if the run produced speakers. |
| `engine` | `object` | Only if the CLI reports it. Includes `diarization: "segment"` or `"word"` when `--diarize` was used, and `diarizer`, the id of the diarization model that drew the turns (`pyannote/speaker-diarization-community-1` or `nvidia/Nemotron-3-Diarization`). `selection` is `"auto"` if the engine was chosen by the probe (`--engine auto`) and `"explicit"` if the user asked for it. |
| `segments` | `[object, …]` | Always. |

### Each segment

| Key | Type | Presence |
|---|---|---|
| `id` | `int` | Always. |
| `start`, `end` | `float` | Always. Seconds. |
| `text` | `str` | Always. |
| `speaker` | `str` | Only with diarization. |
| `no_speech` | `float` | Omitted if the engine did not measure it. Rounded to 4 decimals. |
| `avg_logprob` | `float` | Same. |
| `compression_ratio` | `float` | Same. |
| `suspect` | `true` | Only when the heuristic fires. **Never appears as `false`.** |

`whisper.cpp` does not emit any of the three native signals: under
`--engine whispercpp` those keys are never present.

### `suspect`

Flags a segment for you to review; it does not rule that it's wrong. It fires if:

- `no_speech > 0.6`, or
- the segment lasts 10 s or more and has **fewer than one character per second**.

The second criterion is an uncalibrated density heuristic. Treat it as a
review suggestion; if you need a decision with measured precision, use your
consumer's evaluation harness, not this field.

---

## The `audio/` layer: measurements on the signal, no verdict

`speechtotext.audio` measures and transforms audio; **it never decides
whether to transcribe**. Everything it exports is immutable, and whatever is
built by hand validates itself at construction: a malformed type blows up
where it is created, not three layers down. The types that only ever come
out of a function (`GainResult`, `PreInferenceDecision`) validate nothing —
nobody builds them by hand.

```python
from speechtotext.audio import decode_audio, AudioDecodeError

with open("meeting.m4a", "rb") as stream:
    view = decode_audio(stream, sample_rate=16000)   # -> AudioView
samples = view.samples                               # float32 mono, read-only
```

### Decoding

- **`decode_audio(stream, *, sample_rate, av_module=None) -> AudioView`** — takes a
  *seekable* binary stream (not a path: whoever opens it, closes it), returns a mono
  float32 `AudioView` at the requested `sample_rate`. `av_module` is the tests' seam.
- **`AudioDecodeError(RuntimeError)`** — the file could not be opened or carries no audio.

### A clip and its views

The same audio is measured, identified and transcribed with different preprocessing.
`AudioClip` keeps them together with their provenance, so every number can be traced
back to the samples it came from.

- **`AudioClip(started_at, ended_at, source_id, speech_regions, quality, views)`** —
  `.view(name)` returns the requested view and raises `KeyError` if it is missing or
  does not exist. Validates at construction that the views' durations and
  `quality.duration_ms` match the clip, and that the speech regions neither overlap
  nor run past the clip.
- **`AudioViews(capture, analysis, asr, identity=None, spoof=None)`** — the first
  three are mandatory.
- **`AudioViewName`** — `Literal["capture", "analysis", "identity", "spoof", "asr"]`.
- **`AudioView`** — `samples` (float32 mono, backed by `bytes`: cannot be written to),
  `sample_rate`, `provenance`. **No public constructor**: it is created with
  `AudioView.capture(samples, sample_rate, *, step)` or with
  `AudioView.derive(parent, samples, *, sample_rate=None, steps, models=(), thresholds=None)`.
  Properties: `duration_s` and `pipeline_fingerprint`.
- **`SpeechRegion(start_s, end_s)`** — orderable; requires finite times and
  `0 <= start_s < end_s`.

### Quality and gain

```python
from speechtotext.audio import apply_fixed_gain, compute_audio_quality

gain = apply_fixed_gain(samples, 6.0)
report = compute_audio_quality(samples, gain.samples, 16000, regions,
                               requested_gain_db=6.0, applied_gain_db=gain.applied_gain_db)
```

- **`compute_audio_quality(capture, processed, sample_rate, speech_regions, *, requested_gain_db, applied_gain_db, dropped_frames=0, discontinuities=0) -> AudioQualityReport`**
- **`AudioQualityReport`** — `duration_ms`, `effective_voice_ms`, `input_rms_dbfs`,
  `processed_rms_dbfs`, `peak_dbfs`, `clipping_ratio`, `noise_floor_dbfs`, `snr_db`,
  `requested_gain_db`, `applied_gain_db`, `dropped_frames`, `discontinuities`, `warnings`.
  `noise_floor_dbfs` is `None` when the clip is speech end to end and there is not one
  sample of silence left to measure; `snr_db` is `None` when either side of the
  subtraction is missing — no silence, or no speech region at all. **`None` is not
  zero**: the same rule that governs voice evidence.
- **`apply_fixed_gain(samples, gain_db, *, max_gain_db=18.0, peak_limit_dbfs=-1.0) -> GainResult`**
- **`GainResult(samples, requested_gain_db, applied_gain_db, limited)`** — `limited` is
  `True` when the limiter had to clip the requested gain.

### Quality gate

Decides whether a clip deserves inference, against thresholds the caller sets. The
four signal thresholds deliberately carry no default — measuring is the library's
job, deciding is the caller's. The two transport counters do carry one, at `0`: a
dropped frame is not acceptable by default.

- **`QualityThresholds(min_effective_voice_ms, min_processed_rms_dbfs, min_snr_db, max_clipping_ratio, max_dropped_frames=0, max_discontinuities=0)`**
- **`evaluate_pre_inference(report, thresholds) -> PreInferenceDecision`**
- **`PreInferenceDecision(eligible, reason_codes)`**
- **`QualityReason`** — a `Literal` with eight codes:
  `"silence"`, `"too_short"`, `"level_too_low"`, `"snr_unavailable"`, `"snr_too_low"`,
  `"clipping"`, `"dropped_audio"`, `"discontinuous_audio"`. The gate accumulates
  **all** the reasons that apply, not just the first.

### Provenance

The fingerprint of an audio pipeline: which steps, which models, which thresholds. Lets
you assert that two results came out of the same treatment without keeping the audio.

- **`PipelineStep(name, version, parameters)`** — `parameters` has to be JSON
  serializable; it freezes at construction.
- **`ModelRef(model_id, fingerprint)`** — `fingerprint` is a sha256 in lowercase hex,
  64 characters. Provenance knows nothing about manifests or filesystems: whoever
  has a verified model reduces it to this, and the fingerprint can then be computed
  on any machine.
- **`PipelineProvenance`** — `sample_rate`, `parent_fingerprint`, `steps`,
  `model_fingerprints`, `thresholds`. **No public constructor**:
  `PipelineProvenance.capture(*, sample_rate, step, models=(), thresholds=None)` and
  `PipelineProvenance.derive(parent, *, sample_rate, steps, models=(), thresholds=None)`.
  Property `fingerprint` (sha256 of the sorted payload), plus `to_dict()` and
  `from_dict(data, *, parent, models)`.

---

## Voice evidence

`speechtotext.audio.compute_voice_evidence` describes an audio signal. **It
decides nothing**: it does not classify voice against non-voice, it is not a
VAD, and it does not combine its own measurements into a boolean. The
threshold and the decision belong to the caller.

```python
from speechtotext.audio import compute_voice_evidence

evidence = compute_voice_evidence(window, sample_rate=16000)

if evidence.voice_band_ratio is None:
    ...  # could not measure: the window is below -80 dBFS
elif evidence.voice_band_ratio >= 0.4:
    ...  # there is energy where voice lives
```

### Input

`compute_voice_evidence(samples: np.ndarray, sample_rate: int) -> VoiceEvidence`

- `samples`: mono, 1-D, any dtype convertible to `float64`, all finite.
  Stereo, `NaN` or `inf` raise `ValueError`.
- `sample_rate`: positive integer, high enough to resolve F0 in 70-350 Hz. A
  rate that is too low raises `ValueError`.
- Minimum duration: one 64 ms frame. Below that it returns every measurement
  as `None` with `frames = 0`, no error.

### `VoiceEvidence` fields

| Field | Type | What it measures |
|---|---|---|
| `voice_band_ratio` | `float \| None` | Energy between 300 and 3400 Hz over total energy, weighted per frame (FFT with a Hann window). The telephone band where intelligibility lives. |
| `voiced_ratio` | `float \| None` | Fraction of **all** frames where F0 was detected between 70 and 350 Hz. |
| `f0_median_hz` | `float \| None` | Median F0 over the voiced frames. `None` if none were voiced. |
| `spectral_flatness` | `float \| None` | Geometric mean over arithmetic mean of the spectrum. 0 is pure tone, 1 is white noise. |
| `frames` | `int` | Frames analyzed. Only 0 if the audio is shorter than one frame. |

The three ratios fall in `[0, 1]`; `f0_median_hz` is positive.

### `None` is not zero

The four measurements are `None` **all at once** when no frame exceeds -80
dBFS: that is *could not measure*, and it differs from *measured, and there
is no voice*, which is reported as `0.0`. A consumer that treats `None` as
zero turns silence into a negative verdict, and that is exactly the mistake
this module exists to not make.

### Internal constants

| Constant | Value |
|---|---|
| `VOICE_BAND_HZ` | `(300, 3400)` |
| `F0_RANGE_HZ` | `(70, 350)` |
| `FRAME_S` | `0.064` (four periods of 70 Hz) |
| `HOP_S` | `0.016` |
| `VOICING_THRESHOLD` | `0.5` normalized autocorrelation |
| `OCTAVE_COST` | `0.05` per octave of lag |
| `SILENCE_RMS` | `1e-4` (-80 dBFS) |

They are exposed so you can reason about the measurements, not so you can
tune them: changing them changes the meaning of numbers you already stored.

### Known limits

- **A pure tone inside the band fools two measurements**: it produces a
  `voice_band_ratio` close to 1 and a `voiced_ratio` close to 1, with a
  fabricated subharmonic `f0_median_hz`. `spectral_flatness` below `1e-6`
  gives it away. If your input can carry tones (ringtones, feedback, dial
  tones), check the flatness.
- **F0 without interpolation**: the resolution is integer lag, so the
  median quantizes more at higher frequencies.
- **Fixed window and hop**: 64 ms and 16 ms, not configurable.
- Gain does not affect the ratios: measuring at -20 dB or at -40 dB gives
  the same result.

### Guarantees

- **Deterministic across processes**: the tests pin golden values in hex and
  verify them in a separate process, so even a change of FFT backend gets
  caught.
- **Memory bounded by batch**, not by duration: the peak stays under 80 MB
  over 20 s, and grows less than 1.5× when moving to 120 s.

> The `voice_band_ratio >= 0.4` threshold that appears in the example is
> what one consumer uses in production to decide whether there was voice
> under a transcribed text. It is a value calibrated against its microphone
> and its case, not a constant of this library. Calibrate yours with
> `scripts/validate_voice_evidence.py`, which compares a real-speech frame
> against an empty-room one and prints the overlap.

---

## Voice registry

`speechtotext.speakers.registry`. Lives in `~/.speechtotext`, or wherever
`SPEECHTOTEXT_HOME` points.

```python
from speechtotext.speakers import registry

registry.enroll("Alice", embedding, seconds=12.4, model="pyannote/speaker-diarization-community-1")
registry.list_voices()                                    # all of them, across every model
vectors = registry.get_embeddings("pyannote/speaker-diarization-community-1")
registry.remove("Alice")                                 # across every model
```

| Function | Contract |
|---|---|
| `home() -> Path` | `SPEECHTOTEXT_HOME` if it is set; otherwise `~/.speechtotext`. |
| `enroll(name, embedding, *, seconds, model) -> None` | `model` is mandatory and keyword-only. |
| `list_voices(model=None) -> list[dict]` | Rows `{"name", "model", **meta}`, sorted by name. Without `model`, mixes every one. |
| `get_embeddings(model) -> dict[str, np.ndarray]` | `model` is mandatory and positional. |
| `remove(name, *, model=None) -> bool` | Without `model`, deletes that person across every model. |

### The model is part of the identity

Every vector is filed under the model that produced it and is only compared
against vectors from the same model: the cosine between two different
vector spaces means nothing. That is why `get_embeddings` requires the
model and has no default.

**Trap**: asking for a model with no enrolled voices returns `{}`, not an
error. A misspelled model name looks exactly like an empty registry —
nobody gets identified, and there is no signal telling you why. If your app
depends on identifying someone, check that the dictionary does not come
back empty before moving on.

Entries whose `.npy` no longer exists on disk are silently ignored.

### On-disk format

```
~/.speechtotext/voices/
├── manifest.json
└── pyannote_speaker-diarization-community-1/
    └── alice.npy
```

`manifest.json` is a nested dictionary, `{model: {name: metadata}}`, with
`file`, `seconds` and `enrolled_at` per voice. Paths are stored with a
fixed `/` so the registry is portable across platforms. Vectors are
`float32`.

The flat manifests from `0.4.x` (not grouped by model) are still readable:
they are regrouped in memory on load, and the file on disk is rewritten to
the new format on the next `enroll` or `remove`.

### Producing embeddings

Today there is only one producer: `speakers.diarization.embed_voice`, with
pyannote, and it runs the full diarization pipeline to get one vector out —
**seconds per sample**. It works for batch, not for a live path.

The registry itself, though, is agnostic: `model` is a free-form string,
and filing vectors from a different extractor works today. What is tied to
pyannote is the CLI (`enroll` and the identification path), not the store.

---

## Identification

`speechtotext.speakers.identify` is the module — the `speakers` package
re-exports nothing, it is imported by submodule.
`assign_names(clusters, enrolled, threshold) -> dict[str, str]` maps each
anonymous `speaker_id` to a registered name, with a greedy algorithm: it
sorts every possible pair from highest to lowest cosine, cuts below the
threshold, and assigns without reusing a speaker or a name twice. The
CLI's default threshold is `0.5`. `cosine` is exposed but is
implementation: it can change without notice.

### Diarizing

`speakers.diarization.diarize(samples, sample_rate, num_speakers=None)`
returns `(turns, embeddings)`: `turns` is a list of
`(start, end, speaker_id)` and `embeddings` is one vector per
`speaker_id`, in the same space as `embed_voice` — comparable against
what is registered. `num_speakers` is a hint; if omitted, the pipeline
decides how many there are. A speaker whose embedding comes out `NaN`, or
that the pipeline does not return, shows up in `turns` but not in
`embeddings`. Requires the `[diarize]` extra; the pipeline loads once per
process.

`speakers.nemotron.diarize(samples, sample_rate)` is the second diarizer,
NVIDIA's Nemotron-3-Diarization at a pinned revision. It returns `turns`
only, in the same `(start, end, speaker_id)` shape, with `speaker_0`,
`speaker_1`, … numbered by first arrival and at most eight of them. There
are no embeddings, so its speakers cannot be compared against the registry,
and there is no speaker-count hint: the model decides. The turns are the
model's 10 ms frames above 0.5, unmerged, so a pause ends a turn and
overlapped speech gives overlapping turns. It takes 16 kHz audio and raises
`ValueError` for any other rate. Audio shorter than 80 ms gives `[]` without
loading the model. Requires the `[nemotron]` extra; the model loads once per
process, and anything transformers prints while loading goes to stderr,
never stdout.

`speakers.nemotron.missing()` returns `None` when that extra can run, or a
sentence naming what is missing with the command that installs it. It
imports nothing heavy, which is why `transcribe()` asks it before the
transcription starts rather than failing after it.

---

## Non-finite signals: two policies, on purpose

The same trio of signals (`no_speech`, `avg_logprob`, `compression_ratio`)
is treated differently depending on where it enters, and the difference is
deliberate:

| Path | On `NaN` or `inf` |
|---|---|
| `core.segments.native_signals` (the CLI's and the JSON's) | Converts it into absence of measurement: the key is omitted from the JSON. |
| `asr.types.NativeSignals` / `SegmentNativeSignals` | Raises `ValueError`. |

Halfway through the pipe, "this signal does not exist" is already a
correct answer, and forcing every caller to catch an exception for that
buys nothing. Inside a `TranscriptionResult`, on the other hand, a
non-finite value is data corruption and must abort.

Practical consequence: in the CLI's JSON, *absent* and *invalid* look the
same; in the `asr/` types, invalid never gets to exist.

---

## The `asr/` layer: the only engine contract

`speechtotext.asr` defines the speech-to-text engine: `AsrBackend`, `Caps`,
`TranscriptionRequest`, `TranscriptionResult`, `NativeSignals` and friends,
with strict validation and immutable dataclasses. **Every path goes through
here**: the CLI, `bench`, `find` and `transcribe()` build a backend and
talk to it the same way.

```python
class AsrBackend(Protocol):
    backend_id: str          # "faster-whisper" | "whispercpp"
    caps: Caps               # hotwords / vad / word_timestamps -> honored | degraded | rejected
    model_id: str; model_version: str; engine_version: str; quant: str; device: str
    def warm(self) -> None                                  # loads (once); the object is the cache
    def transcribe(self, samples: np.ndarray, request: TranscriptionRequest, *,
                   on_segment: Callable[[TranscriptionSegment], None] | None = None,
                   cancel: threading.Event | None = None) -> TranscriptionResult
```

`on_segment` receives each segment as soon as the engine has it, in order,
with times local to `samples`. `cancel` is checked between segments: once it
is set, the call raises `AsrError("cancelled")` instead of returning a partial
result. Both engines decode 30-second windows, so segments arrive in bursts,
one per window. faster-whisper checks `cancel` as it hands over each segment;
whisper.cpp ends its process as soon as `cancel` is set. An exception raised
by `on_segment` belongs to the caller: let it propagate unwrapped.

`TranscriptionResult.segments` are
`TranscriptionSegment(start, end, text, words, native_signals)`, and its
`words` are `TranscriptionWord(text, start, end, confidence)` —
`confidence` is `None` or a number between 0 and 1, never fabricated.
`Cap` is the `Literal["honored", "degraded", "rejected"]` that `Caps` is
built from.

Only **float32 mono at 16,000 Hz** goes in, nothing else; the caller
resamples (`core.transcribe.load_audio` does it from any file). The text
of segments and words is returned exactly as the engine emits it, leading
space included: trimming belongs to whoever presents it.

Two implementations, neither re-exported (importing `speechtotext.asr`
does not load engines):

- `speechtotext.asr.faster_whisper.FasterWhisperBackend(model, config=None, *, model_version="unpinned")`
  — `model` is a name (faster-whisper resolves it from the HF Hub) or a
  path to a CTranslate2 directory (local only). Honors hotwords, VAD and
  words.
- `speechtotext.asr.whispercpp.WhisperCppBackend(model)` — binary and GGML
  model pinned by SHA-256 (`core/enginepin.py`), always CUDA, `q5_0`.
  Rejects hotwords (the prompt is inert under `-mc 0`), degrades VAD and
  words, and emits no native signals.
  It reads whisper-cli's output as it decodes, one line per segment, for
  `on_segment`; the result still comes from its JSON. Cancelling ends the
  process.

`TranscriptionRequest(language="es", hotwords=(), word_timestamps=True, beam_size=5, context=None, vad=False)`;
`language="auto"` lets it detect. The `fingerprint` includes `vad`.

---

## `transcribe()`

```python
from speechtotext.core.transcribe import transcribe, Transcript, Progress, PartialSegment, AsrError

t = transcribe("meeting.mp4", model="large-v3", on_progress=print)
```

A file (or 16 kHz mono samples) goes in, a `Transcript` comes out:
`segments` (with speaker and native signals; the `suspect` mark is
computed by the JSON writer with `is_suspect`), `language`,
`language_probability`, `duration`, `speech_s`, `gaps`, `engine` (an
`EngineInfo`, with `.to_dict()` for the JSON), `request` (the effective
one, after CAPS), `warnings` and `diarization` (a `DiarizationReport` or
`None`). A single decode; the short file and the long one are the same
path with n chunks; chunks leave a checkpoint by content in
`~/.speechtotext/chunks`. The core never prints: `on_progress` receives
`Progress(stage, done, total, detail)` with stages
`decode → load → transcribe → diarize` (with a file, `decode` is emitted
twice: first with `total=None` and then with `done = total = duration`;
`download` is emitted by `models.ensure`).

During `transcribe`, `done` and `total` are seconds of audio: `done` grows as
the engine finishes each segment, summed across the chunks that run in
parallel, and never goes back; `total` is the duration. `on_segment` receives
each `PartialSegment(start, end, text)` as the engine produces it, in seconds
of the whole recording, trimmed and with no speaker yet: a live preview, not
the result. With parallel chunks, `PartialSegment` calls arrive interleaved,
not in time order. Audio read from a checkpoint produces none. Both callbacks
may be called from worker threads, never two at once. `cancel` is a
`threading.Event` checked each time the engine hands over a segment (a
stretch with no speech delays it; under whisper.cpp the process is ended at
once), and the call then raises `AsrError("cancelled")`. If a callback
raises, the run stops: pending chunks are cancelled, neither callback is
called again, and the callback's own exception comes out of `transcribe()`
unchanged, with its cause and context.

Errors: `AsrError(code, recoverable, message)` with `code` in
`unsupported_option`, `out_of_memory`, `insufficient_resources`,
`backend_failed`, `cancelled`, `diarize_unavailable`, `diarize_failed`;
`AudioDecodeError` if the file cannot be opened. The engine's warnings
(e.g. `empty_transcript`) land in `warnings` as `"<engine>: <warning>"`.
`backend=` lets you reuse a warm model across calls. `route=` takes an
already-resolved `Route` (the CLI probes, prints the reason and passes it
in: the machine gets looked at exactly once).

`diarizer=` picks who draws the turns when `diarize=True`: `"pyannote"`
(the default) or `"nemotron"`. Anything else is a `ValueError` before the
audio is decoded. With `"nemotron"`, a missing dependency is
`AsrError("diarize_unavailable")` before any transcription, carrying
`speakers.nemotron.missing()`'s sentence. Nemotron counts speakers itself,
so passing `speakers` with it is `AsrError("unsupported_option")`, also
before any work. It puts no names on anyone. When voices are enrolled and
`identify` is on, `warnings` says they were not compared, and
`DiarizationReport.enrolled` is `0`. Its `speakers` is the number of
distinct speakers in the turns.

---

## Probe and models

```python
from speechtotext.core import probe, models

m = probe.machine()                       # < 1 s, no models loaded
r = probe.choose_route(m, "large-v3")     # engine="auto", device="auto", compute_type="auto"
```

`Machine(platform, cpu_count, ram_gb, cuda, gpu_name, vram_free_gb, whispercpp)`:
what is there, with `None` where it could not be measured.
`Route(engine, device, compute_type, reason, eta_factor, estimated)`: the
choice; `reason` is a sentence to print (empty if there is nothing to warn
about); `eta_factor` multiplies the audio's duration (`None` = not
measured) and `estimated` is `False` only if it came from this machine's
`bench.json`. Rules: what's explicit is respected, the probe only fills in
`auto`; **it never changes the model** — if it does not fit,
`AsrError("insufficient_resources")`. Outside Windows, whisper.cpp is
labeled `device="native"`.

```python
models.data_dir() -> Path                                    # SPEECHTOTEXT_HOME or the system path
models.installed(engine=None) -> list[ModelInfo]             # ModelInfo(engine, name, path, size_bytes, verified)
models.ensure(engine, name, on_progress=None) -> Path         # downloads if missing; Progress("download", …)
models.remove(engine, name) -> None                           # FileNotFoundError if not there
models.remote_size(engine, name) -> int | None                # bytes ensure would download; None without network
```

`verified` is `True` only for whisper.cpp (sha256 against the pin);
faster-whisper's models are verified by Hugging Face by size. Valid names:
`tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`
(faster-whisper) and `large-v3`, `small` (whisper.cpp).

---

## What in `core/` is contract and what is not

`core/` is not imported as a package: it is imported by submodule, for
example `from speechtotext.core.transcribe import transcribe`. Of what
lives in there, **only what this document names** is contract, and the
contract is drawn symbol by symbol, not submodule by submodule:

| Submodule | What is contract |
|---|---|
| `core.transcribe` | `transcribe`, `Transcript`, `Progress`, `EngineInfo`, `DiarizationReport`, `load_audio` |
| `core.probe` | `machine`, `choose_route`, `Machine`, `Route` |
| `core.models` | `data_dir`, `installed`, `ensure`, `remove`, `remote_size`, `ModelInfo` |
| `core.formats` | `write_json`, `is_suspect` — the other writers are not |
| `core.segments` | `native_signals`, and nothing else |

That table is exactly the `CONTRACT` list in `tests/test_api_contract.py`:
if one moves without the other, the test fails. Whatever changes there
goes in `CHANGELOG.md`.

The rest of `core/` is internal — `chunked`, `finder`, `benchmark`,
`enginepin`, `postprocess`, and `core/audio.py`, which transcodes with
ffmpeg and has nothing to do with the `speechtotext.audio` package above
despite the name. Use it if it helps you, but it can move between
versions without notice and without an entry in the CHANGELOG.

`cli/` is not contract, `cli/mcp_server.py` included. The four tools that
`speechtotext mcp` serves are thin wrappers over `core.transcribe`,
`core.finder`, `speakers.registry` and `core.probe`. Calling them from
Python buys you nothing: it calls what they wrap, with whatever
guarantees this document gives each piece — `core.finder`, for instance,
has none.

The `evaluation/`, `security/`, `models/` and `confidence/` packages that
existed through `0.5.1` were extracted in `0.6.0`: they were one
consumer's evaluation harness and chain of custody, not part of
transcribing audio.
