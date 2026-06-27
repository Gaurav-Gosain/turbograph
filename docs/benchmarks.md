# Benchmarks

turbograph is measured against standard retrieval benchmarks, not just unit
tests. This document records the methodology, the results, and the findings the
benchmarks forced, including changes to defaults that the data did not support.

## Methodology

- Datasets:
  - [BEIR](https://github.com/beir-cellar/beir) **SciFact** (5183 docs, 300 test
    queries) and **NFCorpus** (3633 docs, 323 queries): single-hop, document-level
    qrels. SciFact is dense-dominant; NFCorpus is keyword-heavy medical text.
  - **[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG)** (609 news
    articles, 2255 answerable queries): multi-hop queries whose evidence spans
    several documents. This is the benchmark that actually stresses the graph.
- Embedding model: EmbeddingGemma (300M) served locally by Ollama, 768 dims, the
  shipped default. Generation/reranking are not used; this measures retrieval.
- BEIR scoring is document-level (chunks collapsed to their best-ranked document)
  with nDCG@10, Recall@10, Recall@100. MultiHop-RAG follows the paper's protocol:
  a retrieved chunk is relevant if it contains a gold evidence sentence as a
  substring; metrics are MRR@10, MAP@10, Hits@4, Hits@10.
- All measurements run on CPU through the turbograph library exactly as shipped.
  The only benchmark-specific code is dataset parsing, which lives outside the
  module, so nothing benchmark-specific is in the binary.
- Test bench: Intel Core i7-10700 (8 cores / 16 threads, AVX2), Linux, Go
  toolchain. Component timings in the comparison section below use this machine.

## Headline results

EmbeddingGemma, default configuration (asymmetric prompts, lexical weight 0.25,
graph off), after the findings below.

| dataset       | metric  | turbograph | reference baseline                    |
| ------------- | ------- | ---------- | ------------------------------------- |
| SciFact       | nDCG@10 | **0.79**   | BM25 0.67; strong dense low-mid 0.70s |
| NFCorpus      | nDCG@10 | **0.40**   | BM25 0.33; strong dense mid-0.30s     |
| MultiHop-RAG  | MRR@10  | **0.56**   | best non-reranked embedding 0.43      |
| MultiHop-RAG  | Hits@10 | **0.54**   | best embedding+reranker 0.75          |

turbograph's retrieval is at or above the strong-dense baseline on every set, and
on MultiHop-RAG it beats the paper's best *non-reranked* embedding (bge-large
0.430, voyage-02 0.393) by a wide margin with no reranker, and lands within reach
of the best embedding+reranker configuration (0.586 MRR@10).

## What the benchmarks forced

### 1. Asymmetric embedding of an instruction-tuned model

EmbeddingGemma expects different prompts for queries and documents; turbograph fed
both raw text. The first SciFact run scored 0.564 nDCG@10, well under the model's
capability. Applying the documented prompts (`task: search result | query: ` and
`title: none | text: `) lifted it several points. The fix generalizes: the client
sets query/document prompts from the model name, with presets for EmbeddingGemma,
E5, BGE, and Nomic and a safe empty default otherwise. See
[architecture.md](architecture.md#asymmetric-embeddings).

### 2. Graph reranking lowers precision, on single-hop and multi-hop alike

The retrieval score originally folded a personalized-PageRank centrality signal
into the ranking with a large default weight (a convex blend at 0.6), which
collapsed nDCG@10 to 0.60 on SciFact. Rebuilding it as an additive boost
(`relevance + GraphMix * pagerank`, which cannot demote a strong direct hit) fixed
the collapse, but a sweep over the final scoring shows the graph still does not
*help* ranking, and it slows each query (it computes PageRank over the whole
graph):

| GraphMix          | SciFact nDCG@10 | SciFact ms/query | MultiHop-RAG MRR@10\* |
| ----------------- | --------------- | ---------------- | --------------------- |
| 0 (off, default)  | **0.790**       | **168**          | **0.48**              |
| 0.2               | 0.790           | 190              | 0.45                  |
| 0.5               | 0.768           | 197              | 0.40                  |
| 1.0               | n/a               | n/a                | 0.35                  |

\*MultiHop column is the dense-only sweep, where the decline is sharp; on SciFact
the graph is neutral at a low mix and harmful at a high one.

PageRank measures centrality, not relevance, so at best it is neutral and at worst
it buries the right answer. It also cannot do multi-hop, the one place it should
shine: multi-hop evidence lives in *dissimilar* documents that a similarity graph
does not connect (the published multi-hop graph wins, e.g. HippoRAG, use an entity
graph seeded by query entities, not a chunk-similarity graph). So similarity-graph
reranking is **off by default** and skipped entirely on the default path, which is
also faster; it stays opt-in via `GraphMix` for thematic and associative queries,
and the graph still powers communities and the visualization. The entity graph
(`EntityMix`) is the path intended for multi-hop and is built on demand.

### 3. Lexical fusion: additive, not rank-based

Reciprocal rank fusion is the usual dense+BM25 hybrid, but it is rank-only and
discards the dense cosine magnitude. On SciFact, where the dense model dominates,
RRF *lowered* nDCG@10 from 0.78 to 0.74. The fix is a magnitude-preserving
additive blend, `relevance = dense + LexicalWeight * bm25`, both normalized. A
weight sweep found a small value helps everywhere:

| LexicalWeight     | SciFact nDCG@10 | MultiHop-RAG MRR@10\* |
| ----------------- | --------------- | --------------------- |
| 0 (pure dense)    | 0.783           | 0.45                  |
| **0.25 (default)**| **0.790**       | **0.54**              |
| 0.5               | 0.774           | 0.56                  |
| 1.0 (≈ RRF)       | 0.743           | n/a                     |

\*MultiHop column is a fixed query subsample for the sweep; the full-set default
(weight 0.25) is MRR@10 **0.563**, Hits@10 0.545.

Weight 0.25 improves all three sets at once: SciFact +0.7 pt, NFCorpus +2 pt
(0.376 to 0.396), and MultiHop-RAG +8 pt (+17%). Larger weights keep helping
MultiHop but start to cost SciFact, so 0.25 is the conservative default. Lexical
fusion is on by default and disabled with `DisableLexical` or a negative weight.

### 4. Pseudo-relevance feedback helps some corpora, hurts multi-hop

Rocchio-style PRF (averaging top-hit vectors into the query) is implemented but
**off by default**: on MultiHop-RAG it *lowered* MRR@10 from 0.48 to 0.45, the
classic query-drift failure mode (the top hits cover one hop, so the centroid
pulls the query away from the bridging evidence). It remains an opt-in knob for
recall-bound, single-topic corpora where it helps.

### 5. Matryoshka truncation: a speed/memory knob

EmbeddingGemma is Matryoshka-trained, so embeddings can be truncated and
renormalized. Truncating 768 to 256 dims on SciFact cost ~2 points of nDCG@10
(0.790 to 0.768, Recall@100 unchanged at 0.98) while roughly halving the store on
disk (38.5 MB to 18.5 MB; the vector portion shrinks 3x) and shaving per-query
latency (168 to 145 ms). It is opt-in (`--embed-dim`), for when memory or
vector-scan speed matters more than the last points of accuracy.

## In-process feature A/B (local model, small corpus)

The headline numbers above are BEIR / MultiHop-RAG. The graph, decomposition, and
contextual-retrieval features are measured separately by a model-backed A/B
harness committed in `server/ab_test.go` and `server/ab_hard_test.go`, run with
`TG_AB=1` (skipped in CI) against a small self-contained fictional corpus with
distractors, using `nomic-embed-text` and `qwen3.5:4b`. These numbers are
**directional, not BEIR-scale**: a dozen questions on a tiny corpus with a 4B
local model. They exist to show *under which conditions* a feature helps, since
that is the part the headline aggregates hide. The harness is the durable asset;
re-run it to ground any claim.

What it showed:

- **A strong baseline is already at ceiling.** On a clean corpus where every
  document is one self-contained chunk with distinct entity names, dense+BM25
  alone hits recall@5 = 1.00 and answers every question. There is no headroom, so
  the advanced features are near-neutral to slightly negative there. This is why
  entity seeding (`entity_mix`) and multi-hop decomposition are **opt-in**: they
  are levers for harder corpora, and on an easy one they only add noise (recall@3
  fell from 0.96 to ~0.83-0.88 when forced on).

- **Knowledge-graph fact injection helps, measured cleanly.** Holding retrieval
  fixed and toggling only the injected triplets, answer token-F1 rose 0.911 to
  0.942 (cover stayed 1.00). An earlier run that looked like facts *hurt* turned
  out to be a confound: that arm had also switched to the degraded
  entity+decomposition retrieval, so the loss was the retrieval, not the facts.
  Isolating one variable at a time is the whole point of the harness.

- **Contextual retrieval earns its keep on fragmentation, not on clean text.**
  Its target case is a long document that names an entity once and then refers to
  it anaphorically ("the programme", "it"); chunking strips the name from the
  later chunks. On that hard corpus, scoring chunk-level retrieval of the
  anaphoric fact chunks:

  | metric          | plain dense+BM25 | contextual retrieval |
  | --------------- | ---------------- | -------------------- |
  | chunkRecall@1   | 0.20             | **0.60**             |
  | chunkRecall@3   | 0.80             | **1.00**             |
  | MRR             | 0.49             | **0.78**             |

  On a clean corpus it is near-neutral (a small recall@3 dip, answer-F1 slightly
  up), which is why it is a deliberate, cost-bearing opt-in (one model call per
  chunk at ingest), enabled with the `contextual` ingest flag.

- **Quantized HNSW traversal was rejected after measuring.** A tempting "speed"
  idea is to walk the graph on the compact TurboQuant codes instead of the exact
  vectors. Measured on this CPU it is the opposite of a win: the exact dot product
  has an AVX kernel (35 ns/op at dim 768) while the scalar quantized estimator is
  461 ns/op, **13x slower**. turbograph keeps exact vectors in RAM and searches
  with them precisely because they are both more accurate and, with AVX, faster.
  The codes' value is compact storage and asymmetric scoring when you do *not*
  have the exact vector, neither of which applies to in-memory search.

## Frontier benchmarks

Two newer benchmarks set the current bar and are noted for context, not run here
(their corpora are large or partly held out): **RTEB** (2025), a generalization-
focused retrieval benchmark with private splits to resist leaderboard
contamination, and **BRIGHT** (2024), reasoning-intensive retrieval where even the
best systems score ~0.22-0.40 nDCG@10. turbograph's local, single-binary footprint
makes the small reasoning split of BRIGHT (`pony`, ~7.9k docs) the natural next
target.

## How turbograph compares to other GraphRAG systems

This section places turbograph against the popular graph and graph-adjacent RAG
systems. Provenance matters here: turbograph's component numbers below are measured on the
test bench defined in the Methodology section, while the cross-system numbers are
taken from each project's own paper or repository, on their own hardware and
models. This is an architectural and capability comparison, not a controlled
head-to-head re-run, which would require standing up every system on a single
corpus and model.

### Measured turbograph component costs

| Operation | Cost | Notes |
|-----------|------|-------|
| `dotf` (768-dim dot) | 151 ns SIMD vs 344 ns Go | the hot path, AVX kernel |
| Encode a vector (TurboQuant) | 71 us | 4 bits/coord, once per chunk |
| HNSW search (100k) | ~0.68 ms | approximate nearest neighbors |
| Quantized flat search (100k) | 13.9 ms vs 77 ms brute float | 5.6x faster than float scan |
| Hybrid retrieve (dense+BM25+fusion) | 259 us | per query, no graph |
| Graph retrieve (+ Personalized PageRank) | 2.0 ms | per query, with PPR |
| Chunk-to-document offset mapping | 333 ns | per chunk, for highlighting |

The headline: turbograph builds its core retrieval graph with **zero LLM calls**
(it is the chunk-similarity graph, derived from embeddings), and a query is a
sub-millisecond to low-millisecond CPU operation with no per-query model call.

### The landscape

The systems split cleanly into two camps by how they answer corpus-wide,
thematic ("global") questions:

- **Summary-based** (Microsoft GraphRAG, RAPTOR, LightRAG's high-level mode):
  pre-generate natural-language summaries of clusters/communities and answer
  global questions from them. Best at sense-making; expensive to index.
- **PageRank-based** (HippoRAG/HippoRAG 2, fast-graphrag, and turbograph): seed
  Personalized PageRank from query-matched nodes and spread activation to
  connected passages. Best and cheapest at local/multi-hop factual questions;
  structurally weak at global questions, because there is no good seed set or
  pre-aggregated thematic node for a whole-corpus query.

| Dimension | MS GraphRAG | LightRAG | HippoRAG 2 | RAPTOR | **turbograph** |
|---|---|---|---|---|---|
| Indexing cost | very high (4-6 LLM calls/chunk + community reports) | low-med (1/chunk) | med-high (>=1 OpenIE call/passage) | low (1/cluster) | **lowest** (similarity graph is embedding-only; entity graph and summaries opt-in) |
| Query latency | high (global map-reduce) | low | low (1 PPR pass) | low | **low** (PPR + RRF, no per-query LLM) |
| Global/thematic queries | best | good | weak | good | **now supported** (community summaries) |
| Local/multi-hop | good | good | **best** | weak | strong (same PPR mechanism) |
| Hybrid lexical (BM25) fusion | no | partial | no | no | **yes** (RRF) |
| Deploy footprint | Python + LLM API + vector store | Python | Python + embed model + vLLM | Python | **single Go binary, stdlib-only, local Ollama** |
| External service required | LLM API | no | LLM | LLM API | **none** |

Sources: Microsoft GraphRAG (arXiv:2404.16130), LightRAG (arXiv:2410.05779),
HippoRAG / HippoRAG 2 (arXiv:2405.14831, arXiv:2502.14802), RAPTOR
(arXiv:2401.18059), fast-graphrag (github.com/circlemind-ai/fast-graphrag).

### Where turbograph wins, and where it trailed

**Wins.** Deployment footprint (a single static binary, no vector DB, no graph
DB, local by default) is category-leading; every competitor is a Python stack,
and most default to a hosted LLM API. Indexing is the cheapest of the group: the
core graph costs zero model calls. Local and multi-hop retrieval use exactly the
Personalized PageRank mechanism that has the strongest independent QA evidence
(HippoRAG 2). Hybrid dense+BM25 fusion with reciprocal rank fusion is a robust,
free quality boost that the pure-graph systems do not foreground.

**The gap that was real.** Global, thematic, sense-making questions ("what are
the main themes across this corpus?"). This is the one dimension where a
PageRank-based system is structurally beaten rather than merely lighter: PPR
spreads from local seeds, so a whole-corpus question has no good seed and no
thematic node to retrieve. Microsoft GraphRAG wins those questions specifically
because of its **community summaries** (72-83% comprehensiveness over a vector
baseline in their study), and RAPTOR gets a similar effect from its summary
tree.

### What was changed to close it

turbograph already partitioned the chunk graph into communities; it was missing
the summaries. So community summaries and a global query path were added (see
[primitives.md](primitives.md) and the `/api/build-communities` and global
`/api/chat` endpoints):

1. Generate one thematic summary per community with the language model, once at
   index time (far fewer calls than per-chunk extraction, and **opt-in**, so the
   zero-LLM default is intact).
2. For a global question, rank the summaries against the query and synthesize an
   answer across the most relevant ones, citing them.

This adopts the proven remedy from the summary-based camp while keeping
turbograph's PageRank strengths and light footprint. It reintroduces a re-summarize-on-update
cost (which LightRAG rightly criticizes in Microsoft GraphRAG); turbograph
mitigates it by keeping summaries opt-in and invalidating them only when content
changes, leaving the cheap default untouched.

The remaining gap is **published retrieval-quality numbers**: turbograph
has measured component costs and the quality studies above, but not yet a
head-to-head accuracy table against these systems on a shared corpus. That
controlled comparison is the natural next benchmark.

## Reproducing

The harness is committed; the datasets and a model server are not, since the
BEIR and MultiHop-RAG corpora are large and the model runs locally.

The `turbograph bench` command ingests a labeled dataset and scores retrieval
end to end. For a BEIR dataset such as SciFact:

```
turbograph bench --format beir \
  --corpus scifact/corpus.jsonl --queries scifact/queries.jsonl \
  --qrels scifact/qrels/test.tsv --k 10 --embed-model embeddinggemma
```

`scripts/bench/scifact.sh` downloads SciFact and runs that command, so the
headline SciFact row regenerates with one script and a local Ollama. The `bench`
package (loaders, the evaluation runner, and the deterministic embedder) backs
both the command and the offline regression suite.

A committed, deterministic quality gate runs in CI with no model and no network:
`bench.TestRetrievalRegression` ingests a small topically-separated corpus
through the full pipeline (chunk, embed with a deterministic hashing embedder,
index, fuse, rank, collapse to documents, score) and fails if recall, MRR, or
nDCG drop below a floor. It catches pipeline regressions everywhere; the absolute
scores are not comparable to the literature numbers above, which use a real
embedder. The `turbograph eval` command scores chunk-level JSONL suites the same
way.

The feature A/B numbers come from a separate model-backed harness in the `server`
package, gated so it never runs in CI:

```
TG_AB=1 go test ./server/ -run TestABRetrievalImprovements -v -timeout 40m
TG_AB=1 go test ./server/ -run TestABContextualHardMode     -v -timeout 40m
```

It needs a local Ollama; models are overridable with `TG_AB_CHAT`,
`TG_AB_EMBED`, and `TG_AB_URL`. Answer accuracy is scored with the `eval`
package's token-F1, exact-match, and the verbosity-robust cover-match metric.
