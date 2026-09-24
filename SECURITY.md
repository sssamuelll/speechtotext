# Security

## Reporting a vulnerability

Report it privately, through this repository's
[Security tab](https://github.com/sssamuelll/speechtotext/security): *Report a
vulnerability* opens a draft advisory that only the maintainer can read. Do not
open a public issue for it. Expect an acknowledgement within a week, and a
fix or a written answer before anything is disclosed.

## What this tool does with the network and the disk

Knowing the shape helps decide what counts:

- **Nothing leaves the machine.** Transcription, diarization and search run
  locally. The only outbound connections are the downloads: Whisper models
  and pyannote's models from Hugging Face, Nemotron from a pinned revision,
  and on Windows the whisper.cpp binary from its GitHub release, verified
  against a SHA-256 pinned in `core/enginepin.py` before it runs. A download
  that fails verification is discarded, not used.
- **What it writes** stays under the data directory (`~/.speechtotext`, the
  platform application-support folder, or `SPEECHTOTEXT_HOME`): chunk
  checkpoints, the `find` index, enrolled voice embeddings, `bench.json`.
  Transcripts are written next to the audio, or where `-o` says.
- **The MCP server** runs the same code over stdio for a client on the same
  machine. It reads the files the client names and writes the JSON next to
  them; it has no network listener.

In scope, then: anything that makes a download run unverified, escapes the
data directory, or lets a file the user did not name be read or written.
Model quality, misattributed speakers and transcription errors are not
security issues; they go in a normal issue with the output of
`speechtotext probe`.

## Supported versions

The latest tag. Fixes ship as a new tag, and consumers move their pin.
