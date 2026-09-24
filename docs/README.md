# The `docs/` index

The [README](../README.md) is the front page. These are the pages behind it:
three say how to use the tool, one says what you can depend on, and one says
why it ended up this way.

| Document | What is in it |
|---|---|
| [`cli.md`](cli.md) | Every subcommand and every flag: `transcribe`, `find`, `bench`, `probe`, `models`, `mcp`; the engines and the route the probe picks; long audio; output formats; environment variables; where things live on disk. |
| [`speakers.md`](speakers.md) | Who said what: the Hugging Face token, enrolling voices, the two diarizers with the measurement between them, and their limits. |
| [`benchmark.md`](benchmark.md) | The transcription benchmark behind the defaults: the recording, the method, the full table, what it does not prove. |
| [`api.md`](api.md) | The contract for consumers: the JSON schema, the whole `audio/` layer, `asr/`, `transcribe()`, the probe and models, the voice registry and identification. `tests/test_api_contract.py` watches it, and whatever a change to it breaks gets a line in [`CHANGELOG.md`](../CHANGELOG.md). |
| [`design.md`](design.md) | Why the product has this shape: the default model, the flags that ship with a measurement against them, the rule that nothing is ever substituted in silence, and the limits nobody has closed. |

The chart files under [`img/`](img/) are regenerated with
`python scripts/benchmark_chart.py`; the hero images there are drawn from the
project's design tokens with the text converted to outlines.
