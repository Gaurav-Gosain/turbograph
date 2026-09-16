# Performance

turbograph does two jobs that are easy to conflate: it **indexes** (chunk, embed,
build the vector and lexical indexes and the optional graph) and it **searches**
(the HNSW walk, the BM25 lookup, the fusion, the ranking). They have separate
knobs, and most knobs that sound like "speed" actually tune the search width, the
ingest parallelism, or the on-disk representation. This document maps each knob to
what it costs and buys. The measured numbers live in
[benchmarks.md](benchmarks.md); every relevant measurement is reproduced from the
committed harness, never from memory.

The headline shape, so the knob discussions below have a frame:

- Search is **sub-linear**: ~0.11 ms at 1k chunks, ~0.36 ms at 10k, ~0.55 ms at
  100k (768-dim, TopK 10, in-process embedder).
- The search index and the similarity graph are built **on demand**, not on
  open: `turbograph add`, `docs`, `forget`, `convert` never pay for structures
  they do not use.
- The **worst case is lexical**: a corpus where every document contains every
  query term makes BM25 score the whole corpus linearly (1.0 ms at 10k, 9.7 ms at
  100k). Real corpora do not look like that; templated benchmarks do.
- Memory at 100k chunks of 768 dimensions is about 750 MB resident once the store
  is searchable, and that is deliberate: the vectors are resident once (the
  store views the index's flat buffer), because holding them twice costs a third
  of the heap for nothing.

## Query-time knobs

These control the per-query path: how many vector comparisons, whether the
similarity graph and entity graph run, and whether expensive stages are skipped.

| knob (CLI / `Config` / `RetrieveParams`) | default | what it costs | what it buys |
| --- | --- | --- | --- |
| `--ef` / `Config.EfSearch` | 64 | lower ef = fewer nodes visited per search | lower latency; costs recall |
| `--ef-gather` / `Config.EfGather` | 0 (off) | a wider candidate-gathering search when seeding PageRank | higher PPR frontier quality; **off until recall evidence** shows it is free |
| `RetrieveParams.GraphMix` | 0 (off) | the expensive one: a full Personalized PageRank pass, ~2 ms and ~50% over plain hybrid | associative/contextual hits; measurably lowers precision on standard retrieval, so opt-in only |
| `RetrieveParams.PRF` | 0 (off) | one extra ANN search plus a centroid mix | recall on underspecified/multi-hop queries; can drift the query on single-hop ones |
| `RetrieveParams.MMRLambda` | 0 (off) | a redundancy pass over a pool of 4x TopK | diverse results |
| `RetrieveParams.EntityMix` | 0 (off) | first use builds the fact index (embeds the relations); each query then runs entity PageRank | entity-aware, multi-hop retrieval |
| `EarlyExitThreshold` / `--` | 0 (off) | nothing when off | skips the graph/entity stages when the dense top hit already clears the threshold and agrees with BM25; changes ranking, so **off until the R0 report** clears it |
| `--topk` / `TopK` | 8 | more seeds and a larger candidate pool | more retrieved chunks |

The defaults are the shape of what retrieval quality studies (see benchmarks.md)
support: the pure-hybrid fast path is both the default and the fastest path, and
every additional signal (graph, entity, MMR, PRF) is opt-in because all of them
either cost real latency, change ranking, or both.

## Ingest-time knobs

These control how fast a corpus is indexed. They do not change the retrieved
ranking except by changing what gets indexed.

| knob (CLI / config) | default | what it costs | what it buys |
| --- | --- | --- | --- |
| `--batch` / `EmbedBatchSize` | 64 | nothing at 64 | setting 1 makes the embedder take one text per HTTP call, which is dramatically slower to the point of being useless for a large corpus |
| `--workers` / `IngestWorkers` | 0 (= number of CPUs) | more workers = more parallel HTTP embedding | ingest wall-clock; embedding is the dominant ingest cost |
| `--checkpoint` | 200 | a periodic save | crash recovery without restarting the whole ingest; the save is the cost |
| `--knn` / `GraphKNN` (+ `MinSimilarity`) | 12 (0.5) | graph build is a k-NN search per chunk; only paid by graph consumers (GraphMix, communities, the graph view) | a denser/chosen similarity graph |
| `--chunk-words` / `Chunk.TargetWords` | 120 | a larger target = fewer chunks = fewer embedder calls and smaller indexes | chunk granularity, which is a retrieval-quality decision first and a size decision second |
| `--stem` / `LexicalStem` | off | extra tokenization work in the lexical rebuild, once per store open or index rebuild | English Porter + Russian Snowball stemming and Russian stopword filtering in BM25; set it consistently for a corpus, and a store remembers it (the lexical index rebuilds from the stored flag) |
| `--entity-batch` | 4 | chunks per LLM call during entity extraction; a higher batch is faster but can dilute a small model | less wall-clock on the explicit `--entities` pass |
| `--bits` / `--residual` | 4 / 32 | nothing at query time | the quantized storage ("codes") representation; see below |

## Storage knobs

These change how big the `.tg` is and how fast it opens, not how fast the query
path is (search always uses exact vectors in RAM).

| knob | default | effect |
| --- | --- | --- |
| `--lean exact` | yes | float32 vectors: 100% size, 139 ms open at the reference corpus |
| `--lean codes` | — | compact TurboQuant codes, decoded on load: ~41% size, same open cost, recall@10 0.982 vs exact; `--bits 8` lifts that to 0.996 at the same size (a code is one byte per dim regardless of bits) |
| `--lean text` | — | no stored vectors, re-embed from chunk text on load: ~24% size, lossless with a deterministic embedder, paid with a slow open (~1.2 s at the reference corpus) |
| `--embed-dim` | 0 (full) | Matryoshka truncation: 768→256 dims roughly halves the store and shaves per-query latency at about 2 points of nDCG@10 |

## How opening is fast

Opening a store used to rebuild the whole vector index and the whole similarity
graph on every command. Now none of that happens up front:

1. The HNSW **link structure is persisted** and restored on first use, so the
   vector index comes back with a wiring pass instead of an O(n·ef) reconstruction.
2. The vectors are read as **one raw contiguous block** (the layout the index
   wants) and adopted without a copy; the store's embeddings become **views into
   that buffer**, so the vectors are resident once.
3. The lexical index rebuilds tokenizing every chunk, across all cores
   (`TestAddBatchIsIdenticalToAdd` pins that the parallel build is byte-identical).
4. The **query-embedding cache** (process-local LRU, keyed by normalized query,
   cleared when the embedding fingerprint changes) avoids re-embedding repeated
   queries over the wire.

The effect at 100k chunks: open dropped from 4.65 s (rebuild everything) to
0.55 s in the final shape, and open+query from 7.70 s to 1.34 s. Against the
committed pre-optimization baseline (`bench/baseline-2026-09-15.json`), search is
~6-10% faster and open ~7% faster with fewer allocations; retrieve is at parity.
The full before/after is `bench/report-2026-09-16.json`.

## Measured and rejected: parallel PageRank

This is the style the whole document follows: a change is kept only while it
measures as a win. The similarity-graph PageRank was briefly parallelized (one
goroutine per CPU per iteration, with per-worker scratch buffers). On the
benchmarked 3000-node graph it was ~50% *slower* than the serial algorithm and
allocated ~9x more per query (the worker buffers and goroutine churn dwarf the
propagation work at that size). It was reverted to the serial algorithm; the one
bit that measured as a clean win -- reusing the two PPR buffers from a `sync.Pool`
instead of allocating them per query -- is kept. See the note in
`bench/report-2026-09-16.json`.

## Reproducing

Everything above is either a committed benchmark or a committed report:

```
go test ./rag/ -run xxx -bench Speed -benchtime 200x -benchmem   # search + open
go test ./index/ -run xxx -bench Dotf -benchmem                   # the SIMD kernel
go test ./rag/ -run xxx -bench "BenchmarkRetrieve|BenchmarkRetrieveGraph" -benchmem
TG_BENCH_BIG=1 go test ./rag/ -run xxx -bench Speed -benchmem     # adds 100k-chunk arms
```

`bench/report-2026-09-16.json` is the before/after against
`bench/baseline-2026-09-15.json` on the same host.