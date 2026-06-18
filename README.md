# turbograph

A fast, local, hackable graph RAG engine in Go.

turbograph is the retrieval layer: you bring documents, an embedding model, and
(optionally) a language model, and it gives you hybrid graph-aware retrieval over
a quantized vector index, a similarity graph, and a streaming chat UI. It runs as
a single self-contained binary with an embedded web interface and a single small
dependency (golang.org/x/sys, for SIMD CPU feature detection; build with
`-tags noasm` for a pure standard-library binary).

It is built to be taken apart. Every external dependency sits behind a small
interface, every algorithm lives in its own package usable on its own, and the
moving parts (embedder, parsers, vector index, graph, lexical search) are
swappable without touching the rest.

## What it does

- Quantizes embeddings with TurboQuant for compact storage and fast estimation.
- Indexes them in an HNSW graph for sublinear nearest-neighbor search.
- Indexes the text with BM25 for exact and rare-term matching.
- Connects chunks into a similarity graph and detects communities, with an
  optional entity-relationship knowledge graph (GraphRAG style) on top.
- Retrieves by fusing dense and sparse hits, seeding Personalized PageRank, and
  optionally diversifying with MMR.
- Grounds answers with numbered inline citations, an evidence-sufficiency
  abstention gate, optional pointwise LLM reranking, and conversational query
  rewriting, each independently switchable.
- Serves a streaming chat UI with an interactive graph visualization, a command
  palette, and full keyboard control.
- Speaks an OpenAI-compatible chat endpoint and serves the corpus over MCP, so
  existing clients and agents connect without changes.
- Ships a deterministic eval harness (recall, precision, MRR, NDCG, context
  precision) for regression-gating retrieval quality.
- Ingests at volume: parallel, resumable, crash-tolerant, with pluggable parsers
  including PDF and OCR.
- Dedupes by content hash and versions documents: re-uploading a changed document
  updates it in place and only re-embeds the chunks that changed.
- Persists to the local disk or any S3-compatible service, and isolates corpora
  into named buckets.

## Design goals

- Modular. Embedding, parsing, the vector index, the graph, and lexical search
  are separate packages behind small interfaces. Swap any of them.
- Local first. Embeddings and generation come from a local Ollama server. No
  data leaves the machine.
- Self-contained. One binary with the UI embedded; effectively zero
  dependencies (only golang.org/x/sys for SIMD detection).
- Fast. AVX SIMD distance kernels, hand-tuned hot paths, parallel ingestion,
  sublinear search.
- Honest. The README states what is approximate and what is exact.

## Architecture

```mermaid
flowchart TB
  subgraph external [Bring your own]
    EMB[Embedder<br/>Ollama by default]
    EXT[Parsers<br/>text, pdftotext, PP-OCRv6]
    LLM[Language model<br/>any Ollama model]
  end

  subgraph core [turbograph]
    direction TB
    RAG[rag.Store<br/>orchestration, ingestion, retrieval]
    QUANT[quant<br/>TurboQuant codec]
    INDEX[index<br/>HNSW + flat ANN]
    LEX[lexical<br/>BM25 + RRF]
    GRAPH[graph<br/>CSR + PageRank + communities]
  end

  SRV[server<br/>JSON API + embedded UI]
  CLI[cmd/turbograph<br/>ingest, query, serve, stats]

  EXT --> RAG
  EMB --> RAG
  RAG --> QUANT --> INDEX
  RAG --> LEX
  RAG --> GRAPH
  RAG --> SRV
  RAG --> CLI
  SRV --> LLM
```

Each core package is independently useful. `quant` is a standalone vector
quantizer, `index` is a standalone ANN index, `graph` is a standalone PageRank and
community library, `lexical` is a standalone BM25. `rag` composes them.

## Retrieval pipeline

```mermaid
flowchart LR
  Q[query text] --> E[embed]
  E --> D[HNSW dense hits]
  Q --> S[BM25 sparse hits]
  D --> F[reciprocal rank fusion]
  S --> F
  F --> SEED[seed Personalized PageRank]
  SEED --> PR[propagate over similarity graph]
  PR --> BL[add graph boost on top of relevance]
  BL --> M[optional MMR diversity]
  M --> R[ranked chunks]
```

The graph step is the point: a chunk that is one hop from a strong hit, but not
directly similar to the query, still receives mass and can be retrieved. That is
what makes it graph RAG rather than plain nearest-neighbor search. The graph score
is added on top of direct relevance rather than blended into it, so it lifts
associated chunks without ever demoting a strong direct match; the boost defaults
to a modest level and can be turned off for pure retrieval. This is measured, not
assumed, on BEIR SciFact and NFCorpus; see [docs/benchmarks.md](docs/benchmarks.md).

Embeddings are asymmetric: instruction-tuned models (the default EmbeddingGemma,
and E5, BGE, Nomic) are fed the query and document prompts they were trained on,
which is worth several points of nDCG@10 over embedding both as raw text.

## Two graph modes

turbograph ships two kinds of graph, and you can use either or both.

- The **chunk-similarity graph** is built automatically and for free: nodes are
  chunks, edges are embedding similarity. It is deterministic and fast, and it
  reinforces clusters of related passages.
- The **entity-relationship knowledge graph** is the classic GraphRAG structure
  and is opt-in. A language model extracts typed entities (people, places,
  concepts) and relationships from each chunk; nodes are entities and edges are
  relationships. Two passages can then be connected because they mention the same
  thing, not because they read alike. Query entities are propagated over this
  graph with Personalized PageRank and projected back onto chunks.

Build the knowledge graph from the web UI (the "entities" toggle on the graph, or
the command palette), or during ingestion with `--entities --gen-model <model>`.
It is extra work because it calls the model per chunk, so it stays off by
default; the similarity path keeps working regardless. At query time the
`entity_mix` control (UI slider or API field) blends the entity signal in.

## Ingestion pipeline

Built for volume: parallel embedding, per-document error isolation, a durable
journal for resume, periodic checkpoints, and a single graph rebuild at the end.

```mermaid
flowchart TB
  WALK[walk source<br/>stream documents] --> SKIP{already done?<br/>journal or store}
  SKIP -- yes --> DROP[skip, do not read]
  SKIP -- no --> EXTRACT[extract text<br/>pluggable parser]
  EXTRACT --> POOL[worker pool<br/>embed concurrently]
  POOL --> COLLECT[serialized indexer<br/>HNSW + BM25 + dedup]
  COLLECT --> CKPT{every N docs}
  CKPT -- yes --> SAVE[save store, then mark journal done]
  COLLECT --> REIDX[reindex once at end<br/>edges + communities]
```

A document is marked done in the journal only after the store containing it has
been saved, so a "done" record always implies recoverable work. Re-ingestion is
idempotent (documents are deduped by id), so resuming after a crash or a pause
never duplicates or loses data. Interrupt with Ctrl-C to pause; re-run the same
command to resume.

## Quick start

Requires [Go](https://go.dev) 1.22+ and a running [Ollama](https://ollama.com).

```
ollama pull embeddinggemma            # the default embedding model
go build -o bin/turbograph ./cmd/turbograph

bin/turbograph serve --gen-model qwen3.5:2b
# open http://localhost:8080, drop in some .txt/.md/.pdf files, and chat
```

If the embedding model is not installed, the web UI shows a one-click pull with
progress, so you do not have to leave the page. Point at a remote Ollama with
`--ollama-url http://host:11434` (or the `OLLAMA_HOST` environment variable);
change the embedding model with `--embed-model`.

Or install the binary directly:

```
go install github.com/Gaurav-Gosain/turbograph/cmd/turbograph@latest
```

The binary embeds the entire web UI, so there is nothing else to deploy.

## The web UI

`serve` ships a self-contained interface (dark, JetBrains Mono, vanilla
JavaScript, no build step). It lets you:

- upload .txt, .md, and .pdf files, indexed incrementally,
- pick a local model and chat, with answers streamed and rendered as markdown,
- see retrieved chunks as source chips that highlight their nodes on hover and
  focus them on click,
- search the graph and watch matches light up,
- explore the similarity graph as an interactive force-directed map colored by
  community, with pan, zoom, drag, hover previews, and per-node detail.

Answers carry numbered citations: each `[n]` in the text is clickable, maps to
the matching source chip, focuses that chunk's node in the graph, and opens a
preview of the passage it rests on. When retrieval is too weak to ground an
answer, the assistant abstains instead of guessing.

It is built to be both approachable and fast to drive. Press `Ctrl K` for a
command palette, `/` to search the graph, `?` for help, and `Esc` to close or
stop. Retrieval settings live in a popover with plain-language explanations,
including a grounding floor (abstain below a cosine threshold) and a rerank
toggle (re-score candidates with the model), and a built-in "how it works" guide
explains the pipeline.

## Grounding

Four refinements sit between retrieval and the answer, each off by default and
independently switchable, so the cheap path stays identical to plain retrieval:

- **Numbered citations.** Passages are numbered `[1..k]` in the prompt and the
  model is asked to cite them; the UI links each `[n]` back to its source.
- **Abstention gate.** If the top hit's cosine similarity is below the grounding
  floor, turbograph abstains rather than answer from the model's memory.
- **Reranking.** A single pointwise LLM call re-scores the candidates and blends
  the model score with the retrieval score. It is fail-open: any error or
  unparseable reply falls back to the base ranking, so it can never do harm.
- **Query rewriting.** An elliptical follow-up ("what about its height?") is
  rewritten into a standalone query for retrieval only, using the recent turns,
  and falls back to the original on any weak rewrite.

## Storage

Buckets persist to the local filesystem by default (`serve --data <dir>`), or to
any S3-compatible service (AWS S3, MinIO, Cloudflare R2). The server saves a
bucket automatically after each ingest, so uploads are durable without a manual
step. The data directory is relative to where you launch `serve` unless you pass
an absolute path, and a bucket file appears only once it has content (an empty
bucket has nothing to write).

```
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
turbograph serve --s3-bucket my-bucket --s3-endpoint https://s3.us-east-1.amazonaws.com
```

The S3 client is implemented on the standard library with SigV4 signing, so there
is no AWS SDK dependency. Storage sits behind a small `storage.Blob` interface, so
adding another backend is one type.

## Command line

```
turbograph ingest --src <dir|file> --out store.tg [flags]   # parallel, resumable
turbograph query  --store store.tg --q "..." [--gen-model M] # retrieve or answer
turbograph serve  --store store.tg --addr :8080 [--gen-model M]
turbograph stats  --store store.tg
turbograph eval   --store store.tg --suite suite.jsonl [--k 10]  # score retrieval
turbograph mcp    --store store.tg [--gen-model M]          # serve over MCP stdio
```

Run any subcommand with `-h` for its flags. Ingestion highlights:
`--workers` (concurrency), `--checkpoint` (crash-recovery interval),
`--pdf-cmd` and `--ocr-cmd` (swap parsers).

## Integrations

### OpenAI-compatible API

`serve` exposes `POST /v1/chat/completions` (streaming and non-streaming). It
accepts the standard request shape, so existing OpenAI clients and SDKs point at
turbograph unchanged; every answer is retrieval-augmented from the selected
bucket. The last user message is the question and the earlier messages become
history for query rewriting. Retrieval knobs (`top_k`, `graph_mix`, `rerank`,
`min_sim`, ...) are accepted as extra fields and ignored by stock clients.

```
curl -s localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.5:2b",
  "messages": [{"role": "user", "content": "what does the corpus say about X?"}]
}'
```

### MCP server

`turbograph mcp --store store.tg` serves the corpus to MCP hosts (editors,
agents, Claude Desktop) over stdio as line-delimited JSON-RPC. It registers a
`search` tool (returns the top chunks as JSON) and, with `--gen-model`, an
`answer` tool (a grounded, cited answer). Add it to a host's MCP config as a
command entry; no network port is opened.

### Evaluation

`turbograph eval --store store.tg --suite suite.jsonl` scores retrieval against a
labeled suite (JSONL, one `{"query":..., "relevant":[chunk ids]}` per line) and
reports recall, precision, MRR, NDCG, and context precision at a cut-off `k`. It
is deterministic for a fixed store and embedder, so it gates retrieval
regressions in CI; `--json` emits the full per-case report.

## PDF and OCR

PDF support is on automatically when `pdftotext` (poppler) is on PATH, which
handles text-based PDFs immediately. For scanned documents and images, wire an
OCR engine such as PaddleOCR PP-OCRv6 through `--ocr-cmd`. turbograph treats
extraction as an external command that reads a file and writes text, so any
parser works. See [docs/ingestion.md](docs/ingestion.md).

## Extending

turbograph is meant to be modified. See [docs/extending.md](docs/extending.md)
for how to:

- swap the embedder (implement one method),
- add or replace a parser (register an extractor by extension),
- use `quant`, `index`, `graph`, or `lexical` as standalone libraries,
- tune quantization, graph construction, and retrieval.

The deeper design is in [docs/architecture.md](docs/architecture.md).

## Performance

Measured on 16 cores, 768-dimensional embeddings, 4 bits per coordinate.

| operation                                   | result                |
| ------------------------------------------- | --------------------- |
| encode one vector (TurboQuant)              | about 74 microseconds |
| HNSW search recall at 10                     | 0.99+ at efSearch 64  |
| HNSW build per insert (clustered)            | about 0.8 ms          |
| flat quantized search, 1k / 10k / 50k       | 0.55 / 2.3 / 8.7 ms   |

The hottest function, the high-dimensional distance, is hand-tuned with multiple
accumulators and bounds-check elimination (profiled with pprof for a 1.8x build
speedup). Index scans and graph edge discovery run across all cores. Ingestion
embeds documents in parallel.

## Tests

```
go test ./...            # full suite
go test -race ./...      # race detector
go test -short ./...     # skip the slow recall and QPS sweeps
```

The Ollama and OCR dependent tests skip automatically when those tools are
absent. Everything else is self-contained: the codebook is checked against
textbook Lloyd-Max distortion, estimators against brute force, HNSW recall against
exact search, BM25 and RRF against known rankings, communities against modularity,
and ingestion (parallel, dedup, resume, error tolerance, cancellation) end to end.

## License

See [LICENSE](LICENSE).
