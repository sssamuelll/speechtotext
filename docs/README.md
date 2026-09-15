# The `docs/` index

Two documents. One says what you can depend on, the other says why it ended up
that way.

| Document | What is in it |
|---|---|
| [`api.md`](api.md) | The contract for consumers: the JSON schema, the whole `audio/` layer, `asr/`, `transcribe()`, the probe and models, the voice registry and identification. `tests/test_api_contract.py` watches it, and whatever a change to it breaks gets a line in [`CHANGELOG.md`](../CHANGELOG.md). |
| [`design.md`](design.md) | Why the product has this shape: the default model, the flags that ship with a measurement against them, the rule that nothing is ever substituted in silence, and the limits nobody has closed. |

The two chart files under [`img/`](img/) are regenerated with
`python scripts/benchmark_chart.py`.
