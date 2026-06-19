# Contributing

turbograph is meant to be read and modified. Patches are welcome.

## Ground rules

- **Standard library only.** The single allowed dependency is
  `golang.org/x/sys` (SIMD CPU detection). Anything that pulls in a third-party
  module will not be merged; shelling out to an optional external tool is fine.
- **Single binary.** The web UI is embedded; keep it that way. No build step for
  the frontend, no separate assets to ship.
- **Lean and hackable.** Prefer a small, readable implementation over a clever
  or general one. Each package should stand alone.

## Before you open a PR

```
make lint        # gofmt + vet, the CI gate
make test-race   # the suite under the race detector
make build       # confirm the binary still builds
go build -tags noasm ./...   # confirm the pure-stdlib path builds
```

New behavior needs tests. Retrieval-quality changes should be backed by a
measurement (see [docs/benchmarks.md](docs/benchmarks.md)); do not tune a default
to a single dataset.

## Layout

- `quant`, `index`, `graph`, `lexical` — standalone algorithm packages.
- `rag` — the store that composes them.
- `server` — HTTP API, OpenAI endpoint, embedded UI, hardening middleware.
- `cmd/turbograph` — the CLI.
- `ollama`, `extract`, `storage`, `entity`, `mcp`, `eval` — integrations and tools.

See [docs/architecture.md](docs/architecture.md) and
[docs/extending.md](docs/extending.md).
