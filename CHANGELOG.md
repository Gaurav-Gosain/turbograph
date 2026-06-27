# Changelog

All notable changes to turbograph are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow semantic versioning once it tags a first release.

There are no tagged releases yet, so everything to date sits under Unreleased.

## [Unreleased]

### Added

- Multi-hop query decomposition: an optional one-call step that splits a
  compositional question into focused subqueries, retrieves each, and unions the
  candidates so evidence in different documents surfaces. Opt-in (`decompose`),
  with the subquery retrievals run concurrently so latency is the slowest hop, not
  the sum.
- Knowledge-graph fact injection: the entity-graph relationships grounded in the
  retrieved chunks are rendered as short triplets and added to the prompt under a
  "Knowledge graph facts" heading, distinct from the numbered citable passages.
  Measured to improve answer F1 when retrieval is held fixed.
- Dense entity-graph seeding: the entity graph's Personalized PageRank is now
  seeded by embedding similarity (so paraphrases match), with the prior
  token-overlap seeding kept as a lexical prior and a backward-compatible
  fallback.
- Entity-graph canonicalization and pruning: near-duplicate entities are merged
  (edit-ratio and salient-token containment) with relation endpoints rewritten
  through the merge map, and generic or ghost nodes are dropped, reducing graph
  fragmentation. The extraction prompt was tightened with a closed type
  vocabulary and coreference rules.
- Contextual retrieval (Anthropic): an opt-in `contextual` ingest flag prefixes
  each chunk, for indexing only, with a short generated sentence situating it in
  its document. The body shown to the user and fed to the model is unchanged. It
  markedly improves retrieval of fragmented, anaphoric chunks; off by default
  since it costs one model call per chunk.
- Answer-quality metrics in the `eval` package: token-F1, exact match, a
  verbosity-robust cover match, and a bootstrap confidence interval, all
  deterministic and LLM-free, plus an optional gold `Answer` field on eval cases.
- A model-backed A/B benchmark harness (gated by `TG_AB=1`, skipped in CI) that
  measures the lift of these features end to end with a real embedder and chat
  model; see [docs/benchmarks.md](docs/benchmarks.md).
- Sentence-boundary lookahead in the sentence chunker, so a period is treated as
  a boundary only when the next non-space character starts a new sentence,
  avoiding splits inside decimals and lowercased abbreviations.
- Document metadata: arbitrary per-document JSON, propagated to every chunk and
  returned with each retrieved result, so callers can filter on it or feed
  selected fields to the model.
- Chunk-to-document offsets: each chunk records its `[start, end)` rune offsets
  in the source document, giving an exact mapping the UI uses to preview a
  document with its retrieved chunks highlighted in place.
- Document view, with the retrieved chunks highlighted, and per-document
  metadata shown alongside.
- Document delete, removing a document and its chunks from a bucket.
- JSON export: `ExportJSON` reads a `.tg` snapshot and writes an equivalent
  indented JSON view (config, chunks with offsets, embeddings, metadata, version
  history, entity graph) for cross-language interop, with an option to omit the
  embeddings when only text and structure are needed.
- The `.tg` store format spec, documenting the on-disk snapshot and what is
  stored versus rebuilt on load.
- Multimodal image support: an image document is described by the model into a
  caption, then the caption is embedded and indexed like text, with the source
  image kept as a content-addressed asset and referenced from the chunk.
- Community summaries and a global query path: one thematic summary per detected
  community, generated once at index time and opt-in, plus a global chat path
  that ranks summaries against a whole-corpus question and synthesizes a cited
  answer across them.
- Visual pipeline and free-form flow editor in the UI, with a tracing run,
  context menu, and inline answers.
- Command-palette submenus: the palette groups actions into drill-in submenus.
- Document version history: each document's content history is tracked and
  persisted, with a UI to browse prior versions.
- OpenAPI 3 spec served at `/openapi.json`.
- Official Python and TypeScript client libraries for the HTTP API.
- HTTP API reference documentation.

### Changed

- Flow editor layout redesigned for clarity.
- Bucket layout documentation rendered as a mermaid diagram.
- README and docs synced and professionalized, and the public API surface tidied.

### Fixed

- Mermaid parse errors in the docs.
- Document list on load and a graph glitch when switching modes.

[Unreleased]: https://github.com/Gaurav-Gosain/turbograph
