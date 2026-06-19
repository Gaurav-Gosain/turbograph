# Extending turbograph

turbograph is built to be modified. The external dependencies sit behind small
interfaces, and the algorithms live in independent packages. This guide shows the
seams.

## Swap the embedder

The store depends on one interface:

```go
type Embedder interface {
    Embed(ctx context.Context, texts []string) ([][]float32, error)
}
```

Anything that turns strings into vectors qualifies: a different Ollama model, a
hosted API, an in-process model, or a deterministic stub for tests. The store
never assumes a dimension; it learns it from the first batch.

```go
type myEmbedder struct{ /* ... */ }

func (m *myEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    // call your model, return one vector per input in order
}

store := rag.New(&myEmbedder{}, rag.Config{})
```

The only contract is: return exactly one vector per input, in order, all the same
length. The bundled `ollama.Client` already satisfies this, and the CLI wraps it
in a batching embedder so large documents are split into bounded requests.

## Add or replace a parser

Parsing is a registry keyed by file extension:

```go
type Extractor interface {
    Extract(ctx context.Context, filename string, data []byte) (string, error)
}
```

Register your own for any extension. Built in are a plain-text extractor and a
`CommandExtractor` that shells out to an external tool, which is how PDF and OCR
work without pulling heavy parsers into the Go binary.

```go
reg := extract.DefaultRegistry()                       // text, plus pdf if pdftotext exists
reg.Register("pdf", extract.CommandFromTemplate(       // override with anything
    []string{"pdftotext", "-q", "{in}", "-"}))
reg.Register("docx", myDocxExtractor{})                // add new formats
```

`{in}` is replaced with a temp file holding the bytes; `{out}`, if present in the
template, is read back as the result. An empty result returns
`extract.ErrEmptyOutput` so a caller can tell "no text found" (a scanned PDF) from
a hard failure. See [ingestion.md](ingestion.md) for wiring PaddleOCR PP-OCRv6.

## Use the packages standalone

Each algorithm is a library with no dependency on the rest.

```mermaid
flowchart LR
  quant[quant<br/>TurboQuant codec] -.-> index[index<br/>HNSW + flat ANN]
  graph[graph<br/>PageRank + communities]
  lexical[lexical<br/>BM25 + RRF]
```

### quant: vector quantization

```go
q := quant.New(quant.Config{Dim: 768, Bits: 4, ResidualDims: 32, Seed: 1})
code := q.Encode(vec)            // compact code
qr := q.PrepareQuery(query)
score := qr.Score(code)          // low-variance ranking estimate
ip := qr.IP(code)                // unbiased inner product
```

`Score` is the low-variance estimator for ranking; `IP`, `L2`, and `Cosine` add
the QJL residual correction for accurate magnitudes. See
[architecture.md](architecture.md) for why both exist.

`quant.Benchmark(cfg, opts)` measures a configuration's recall, compression, and
throughput on synthetic clustered vectors; it backs `turbograph quant bench` and
is callable directly to pick a bit budget for your data.

### index: nearest-neighbor search

```go
h := index.NewHNSW(768, q, index.HNSWConfig{M: 16, EfConstruction: 200})
h.Add("id-1", vec)
hits := h.Search(query, 10, 64)               // top-10, efSearch 64
hits = h.SearchFiltered(query, 10, 64, keep)  // with a metadata predicate
```

A flat quantized index (`index.New`) is also available for exact re-ranking and
small corpora.

### graph: PageRank and communities

```go
b := graph.NewBuilder(n)
b.AddEdge(i, j, weight)
g := b.Build()
scores := g.PersonalizedPageRank(map[int]float32{seed: 1}, graph.DefaultPPR())
comm := graph.DetectCommunities(g, graph.CommunityOpts{Seed: 1})
```

### lexical: BM25 and fusion

```go
ix := lexical.New(lexical.DefaultConfig())   // k1=1.2, b=0.75
ix.Add("id-1", text)
hits := ix.Search("query terms", 10)
fused := lexical.RRF(60, denseHits, sparseHits) // reciprocal rank fusion
```

## Choose or swap the chunker

How documents are split is a one-method interface:

```go
type Chunker interface {
    Split(text string) []Piece   // Piece{Text, Headings}
}
```

Four strategies ship, selected by name:

```go
cfg := rag.Config{Chunk: rag.ChunkConfig{
    Strategy: rag.StrategyRecursive, // or Word, Markdown, Sentence
    TargetWords: 120, OverlapWords: 24,
}}
```

- `recursive` (default) splits on a separator hierarchy (paragraph, line,
  sentence, word), descending only when a piece is over budget, then packs the
  fragments to the target with overlap. It keeps paragraphs and sentences intact.
- `word` is fixed overlapping word windows: fast and uniform, weakest at
  boundaries.
- `markdown` splits on headings, tracks the heading path, and prepends it to each
  chunk as a contextual header (e.g. `Guide > Setup`), which situates the passage
  for both embedding and BM25.
- `sentence` packs whole sentences up to the budget without cutting one in half.

Bring your own by implementing `Chunker` and setting `Config.Chunker`; it
overrides the named strategy. A custom chunker is not persisted (interfaces do not
round-trip through the snapshot), so set it again after loading a store if you
ingest more documents. Strategies that need an embedder or LLM (semantic,
contextual) take it at construction and still satisfy the same one-method
interface, so they compose as decorators around a base chunker.

## Swap the entity extractor

The optional knowledge graph is built through one interface:

```go
type Extractor interface {
    Extract(ctx context.Context, text string) (entity.Extraction, error)
}
```

The bundled extractor prompts a language model and parses a line-delimited
format, but anything that returns entities and relations works: a different
model, a rules or NLP pipeline, or a spaCy or GLiNER service behind an HTTP call.
The `entity` package (graph accumulation, merge, persistence) is independent of
how extraction happens, so you only implement the one method.

## Tune the store

`rag.Config` exposes every knob with sensible defaults:

- Quantization: `Bits` (compression vs accuracy), `ResidualDims` (unbiased IP).
- Chunking: `Chunk.TargetWords`, `Chunk.OverlapWords`.
- Graph: `GraphKNN`, `MinSimilarity`, `SequentialWeight`.
- Search: `HNSW.M`, `HNSW.EfConstruction`, `EfSearch`.
- Hybrid: `DisableLexical`, `RRFK`.

Retrieval is tuned per call with `rag.RetrieveParams`:

- `TopK`, `SeedK` (candidate pool), `Filter` (metadata predicate).
- `LexicalWeight`: BM25 weight added to dense relevance. Zero uses
  `DefaultLexicalWeight` (a small, benchmark-chosen value); negative forces pure
  dense.
- `GraphMix`: PageRank boost added on top of relevance. Off by default (zero);
  similarity-graph reranking lowers precision on standard retrieval, so it is
  opt-in for thematic or associative queries.
- `EntityMix`: weight of the entity knowledge graph (built on demand).
- `PRF`, `PRFWeight`: pseudo-relevance feedback (off by default; helps some
  corpora, hurts multi-hop).
- `MMRLambda`: diversity.

The embedding dimension is a client knob: `ollama.Client.EmbedDim` truncates
Matryoshka embeddings (768/512/256/128 for EmbeddingGemma) for smaller, faster
vectors.

## Asymmetric embedding prompts

Instruction-tuned embedding models encode queries and documents differently. The
`ollama.Client` applies a `QueryPrefix` and `DocPrefix`, set from the model name
by `SetEmbedModel` with presets for EmbeddingGemma, E5, BGE, and Nomic. A custom
embedder can opt into the same split by implementing `rag.QueryEmbedder`
(`EmbedQuery`) alongside `Embed`; the store routes queries through it and falls
back to `Embed` for embedders that do not. Override `QueryPrefix`/`DocPrefix`
directly for a model the presets do not cover.

## Add a storage backend

Where buckets persist is a small interface:

```go
type Blob interface {
    Put(ctx context.Context, key string, data []byte) error
    Get(ctx context.Context, key string) ([]byte, error)
    Delete(ctx context.Context, key string) error
    List(ctx context.Context, prefix string) ([]string, error)
}
```

Two implementations ship: the local filesystem and an S3-compatible client
(SigV4 on the standard library, no SDK). Pass either to a manager:

```go
blob, _ := storage.NewS3(storage.S3Config{Endpoint: ..., Bucket: ..., AccessKey: ..., SecretKey: ...})
mgr, _ := rag.NewManagerBlob(blob, embedder, cfg)
```

A new backend (GCS, Azure, a database) is one type that satisfies `Blob`.

## Replace a whole layer

Because the layers are packages, replacing one is local work. To use a different
ANN index, implement search over `quant.Code` (or your own codes) and call it
from a fork of `rag.Store`. To use a different community algorithm, add it beside
`graph.DetectCommunities`. The store is roughly 400 lines of orchestration; it is
meant to be read and forked.
