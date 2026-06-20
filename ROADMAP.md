# Roadmap

This roadmap is written to be honest about where turbograph is strong and where
it is not yet. Several items below are gaps, not features: they are listed
because they are real and because naming them is more useful than hiding them.
Read it alongside [docs/benchmarks.md](docs/benchmarks.md), especially the "How
turbograph compares" section, which grounds most of what follows.

The work is grouped by horizon, not by date. turbograph is a small project and
priorities shift with use, so no item here carries a delivery promise.

## Near-term

- Reproducible benchmark harness. The numbers in
  [docs/benchmarks.md](docs/benchmarks.md) are real but are local measurements,
  not a committed harness: the BEIR and MultiHop-RAG corpora and a running model
  server do not live in the repo. The near-term goal is a scripted harness that
  downloads a dataset, ingests it, retrieves the test queries, and scores them,
  so the results table can be reproduced from a clean checkout rather than
  trusted.
- Published results table. Pair the harness with a checked-in results table and
  the exact configuration that produced each row, so a reader can see both the
  number and how to get it.
- Head-to-head retrieval-quality numbers. The comparison in the benchmarks doc
  is architectural: it places turbograph against other GraphRAG systems by
  capability and component cost, but there is no controlled accuracy table
  against those systems on a shared corpus and model. That controlled comparison
  is the missing piece.

## Mid-term

- An honest scaling story and a disk-backed index path. turbograph is an
  in-memory engine today: the full embeddings and every index live in RAM, and
  a `.tg` snapshot is loaded in its entirety. This is the right design for
  corpora that fit comfortably in RAM on one machine, and it is fast there, but
  it sets a hard ceiling. A memory-mapped or otherwise disk-backed index path,
  so the working set can exceed RAM, is the way past that ceiling.
  [docs/limits.md](docs/limits.md) states the current envelope and the memory
  math in detail.
- HippoRAG-2-style entity-graph improvements for stronger multi-hop. The
  benchmarks show that a chunk-similarity graph does not help multi-hop, because
  multi-hop evidence lives in dissimilar documents the similarity graph never
  connects. The published winners seed Personalized PageRank from query entities
  over an entity graph. turbograph already has the opt-in entity graph; the
  improvements that follow the HippoRAG 2 design are query-to-triple linking
  (match a query against extracted relationships, not just entity names),
  passage-aware Personalized PageRank seeding (seed and score with passage
  context, not entity nodes alone), and triple filtering (drop low-value
  extracted relations before they dilute the graph).
- Reranking and query-expansion options. A pointwise LLM reranker and
  Rocchio-style pseudo-relevance feedback already exist and are off by default
  for the reasons the benchmarks record. The mid-term work is to broaden these
  into a small, documented set of opt-in quality knobs: more rerankers, more
  query-expansion strategies, each measured rather than assumed.

## Longer-term

- Automatic figure and table extraction from PDFs. The multimodal path can
  describe an image and embed the caption today, but it relies on images being
  supplied. The longer-term goal is to extract figures and tables from PDFs
  automatically and route them through the same describe-then-embed path, so a
  document's non-text content is retrievable without manual preparation.
- Observability. Production use wants metrics and tracing: per-stage retrieval
  timings, ingestion throughput, index sizes, and request traces, exported in a
  standard form rather than read off logs. turbograph has profiling hooks today
  but no first-class observability surface.
- Broader language clients. Python and TypeScript clients exist. The HTTP API
  and the OpenAPI spec make further clients straightforward; which languages to
  add will follow demand.

## What is intentionally not on this list

Some things are absent on purpose, not by oversight. turbograph is a single,
self-contained Go binary that runs locally against Ollama, and that footprint is
a deliberate design choice rather than a stage to grow out of. Features that
would require a mandatory external service, a hosted API, or a separate vector or
graph database are weighed against that choice and are not assumed to be
worthwhile simply because larger systems have them.
