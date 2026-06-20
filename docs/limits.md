# Limits and scaling

turbograph is an in-memory engine. A bucket's chunks, their raw embeddings, the
vector index, the lexical index, and the graph all live in RAM, and loading a
`.tg` file reads the whole snapshot into memory. This is a deliberate design
choice: it is what makes retrieval a sub-millisecond to low-millisecond CPU
operation with no external service, and it is the right trade for corpora that
fit comfortably on one machine. It also sets a clear ceiling, stated plainly
here.

## Where the memory goes

For a corpus of `N` chunks at embedding dimension `d` (`rag.Store` holds
`chunks []Chunk` and `embeds [][]float32`, plus the HNSW index, the BM25 index,
and the similarity graph):

| Component | Rough size | Notes |
|-----------|------------|-------|
| Raw embeddings (`embeds`) | `N * d * 4` bytes | float32 per coordinate; the source of truth for rebuilds and MMR, kept in full precision. This usually dominates. |
| Quantized codes (HNSW) | `~N * d * bits/8` bytes | the compact code per vector; at 4 bits and 768 dims that is about 384 bytes per chunk. |
| HNSW graph | `~N * M * 4` bytes | neighbor links per node (`M` defaults near 12 to 16), times a small layer factor. |
| BM25 index | proportional to total tokens | postings and term statistics over the chunk text. |
| Chunk text | the source bytes | every chunk's `Text` is retained for previews and the lexical index. |

The raw embeddings are the headline term. A worked example at the default 768
dimensions:

- one float32 vector is `768 * 4 = 3072` bytes,
- so `embeds` alone is about `3 GB` per million chunks,
- with the quantized codes, HNSW graph, BM25 postings, and text, budget on the
  order of `5` to `8 GB` of resident memory per million 768-dimension chunks.

A useful rule of thumb: roughly `150,000` to `200,000` chunks per gigabyte of
RAM at 768 dimensions, dominated by the raw vectors. Halve the dimension and you
roughly halve the vector memory.

## Knobs that lower memory

- `--embed-dim N` (Matryoshka truncation): models such as EmbeddingGemma support
  using a prefix of the embedding. Truncating 768 to 256 cuts the raw-vector
  memory to a third for a small accuracy cost. See
  [benchmarks.md](benchmarks.md) for the measured trade.
- Quantization bit rate: fewer bits per coordinate shrink the codes. See
  `quant bench` for the recall-versus-size trade across bit rates.
- Buckets: separate corpora are separate `.tg` files and are loaded
  independently, so only the buckets in use occupy memory.

## The design envelope

turbograph targets corpora that fit comfortably in RAM on a single machine, and
it is excellent across that range: tens of thousands to a few million chunks on
commodity hardware, with no vector database, no graph database, and no external
service. It is not designed today for the hundreds of millions of vectors that
require a disk-backed or sharded store; for that scale, an engine with a
memory-mapped or on-disk index is the right tool.

Past the comfortable in-RAM range the failure mode is graceful to reason about:
memory grows roughly linearly with chunk count, so the ceiling is set by
available RAM and the numbers above, not by a hidden cliff. A memory-mapped,
disk-backed index path that lifts this ceiling is on the [roadmap](../ROADMAP.md).

## Other operational notes

- Indexes are rebuilt deterministically on load from the stored embeddings, so
  load time scales with corpus size (indexing minus the embedding step). The
  embeddings themselves, the expensive part, are never recomputed.
- Ingestion is parallel and checkpointed; see [ingestion.md](ingestion.md) for
  resumable, crash-tolerant bulk loading.
- A single store serves concurrent reads safely; writes take a lock, so heavy
  concurrent ingestion and querying contend on the same store.
