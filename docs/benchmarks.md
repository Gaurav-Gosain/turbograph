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

EmbeddingGemma, **the shipped defaults** (asymmetric prompts, lexical weight 0.25,
graph off, chunk target 120 words).

| dataset       | metric  | turbograph | reference baseline                    |
| ------------- | ------- | ---------- | ------------------------------------- |
| SciFact       | nDCG@10 | **0.772**  | BM25 0.67; strong dense low-mid 0.70s |
| NFCorpus      | nDCG@10 | **0.338**  | BM25 0.33; strong dense mid-0.30s     |
| MultiHop-RAG  | MRR@10  | **0.56**   | best non-reranked embedding 0.43      |
| MultiHop-RAG  | Hits@10 | **0.54**   | best embedding+reranker 0.75          |

An earlier version of this table reported **0.79** on SciFact and **0.40** on
NFCorpus and described them as the default configuration. They were not: they
require a larger chunk target. SciFact and NFCorpus documents are abstracts, and
the 120-word default splits an abstract in half, which costs real accuracy:

| chunk target | SciFact nDCG@10 |
| ------------ | --------------- |
| 120 (default)| 0.7723          |
| 256          | 0.7732          |
| 512          | 0.7856          |

The number in the table is now the one you get by running the shipped binary with
no flags, because that is the number a reader is entitled to assume it is. If your
documents are short and self-contained, raise `--chunk-words`.

### What is actually doing the work

`turbograph bench --ablate` scores every configuration against one index, so the
report says which part earns its place rather than crediting the whole to the sum.
SciFact, 5183 documents, 300 queries:

| arm                     | nDCG@10 | Δ vs default |
| ----------------------- | ------- | ------------ |
| dense only (BM25 off)   | 0.7660  | -0.0063      |
| **hybrid w=0.25 (default)** | **0.7723** | —       |
| hybrid w=0.50           | 0.7587  | -0.0136      |
| hybrid w=1.0            | 0.7388  | -0.0335      |
| lexical-dominant w=8    | 0.6627  | -0.1096      |
| + graph 0.2             | 0.7648  | -0.0075      |
| + MMR 0.5               | 0.7494  | -0.0229      |

Three things, and none of them flatter the feature list:

1. **The embedder does nearly all the work.** Dense alone is 0.766; the lexical
   fusion adds 0.006. It is a real gain and it is free, but the honest description
   of turbograph's retrieval quality on SciFact is "EmbeddingGemma's quality, plus
   a little".
2. **The harness validates itself.** Pushing the lexical weight until BM25
   dominates gives 0.663, which is the published Anserini BM25 baseline for
   SciFact (0.665) to within noise. An independent implementation landing on the
   reference number is the strongest evidence available that the metrics are right.
3. **The graph and MMR measurably hurt** on single-hop retrieval, which is what
   [finding 2](#2-graph-reranking-lowers-precision-on-single-hop-and-multi-hop-alike)
   already says and why both are off by default. Their case is MultiHop-RAG, not
   this.

NFCorpus tells the same story more starkly: dense alone 0.3379, default 0.3383,
graph +0.0003. On that corpus the lexical fusion buys essentially nothing.

turbograph's retrieval is at or above the strong-dense baseline on every set, and
on MultiHop-RAG it beats the paper's best *non-reranked* embedding (bge-large
0.430, voyage-02 0.393) by a wide margin with no reranker, and lands within reach
of the best embedding+reranker configuration (0.586 MRR@10).

## Speed and memory

These measure **turbograph**, not Ollama. The end-to-end query latency in a BEIR run is
about 140 ms and essentially all of it is the HTTP round trip to embed the query;
quoting that as "query latency" would credit the index with the embedder's cost. The
numbers below use an in-process embedder, so what they time is the search: the HNSW
walk, the BM25 lookup, the fusion, and the ranking. 768 dimensions, `TopK` 10.

Reproduce with `go test ./rag/ -run xxx -bench Speed -benchtime 300x`.

| chunks  | search  | with graph 0.2 |
| ------- | ------- | -------------- |
| 1,000   | 0.11 ms |                |
| 10,000  | 0.36 ms | 0.54 ms        |
| 100,000 | 0.55 ms |                |

Ten times the corpus costs about twice the search, which is the sub-linear scaling
HNSW is for. The graph signal costs about 50% more per query, on top of measuring
worse (above), which is the whole reason it is opt-in.

**The worst case is worth stating too.** When every document contains every query term,
BM25 cannot narrow anything and has to score the entire corpus: 1.0 ms at 10k chunks
and 9.7 ms at 100k, and it is linear. That shape is rare in practice and easy to
mistake for the typical case — a templated benchmark corpus produces exactly it, and
one did, and it briefly convinced me that turbograph's search was O(n). It is not. But
a corpus of near-identical documents will pay for it.

Opening a store (which every CLI command does before anything else):

| chunks  | store size | open   |
| ------- | ---------- | ------ |
| 20,000  | 22 MB      | 0.18 s |
| 100,000 | 300 MB     | 1.2 s  |

**Memory.** A 100,000-chunk store at 768 dimensions is about 750 MB of heap once it is
searchable, of which 307 MB is the vectors themselves.

It used to be 1055 MB, because the vectors were resident twice: once in the store, and
again inside the HNSW index, which keeps its own flat contiguous copy. That copy earns
its place, since contiguity is what makes the distance kernel fast; the store's second
copy did not. The store's embeddings are now views into the index's buffer, so the
vectors are resident once. `TestVectorsAreNotHeldTwice` pins it, because the failure mode
is silent: any path that appends to the index without re-pointing the views leaves the
old buffer alive and memory quietly doubles again.

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

### 6. A fixed reranker weight lets a small model throw out the best hit

The reranker blends the model's pointwise judgement with the retrieval score. That
blend used a single fixed weight (0.7 model / 0.3 retrieval) at every rank, and
the arithmetic of that is bad: a candidate the retriever ranked 15th, with weak
retrieval support, could beat the top hit on the model's word alone
(`0.7*1.0 + 0.3*0.25 = 0.775` against `0.7*0.6 + 0.3*1.0 = 0.72`). The head of a
hybrid ranking is a high-confidence signal; a pointwise score from a small local
model is noisy and uncalibrated. Weighting them equally everywhere is the wrong
shape.

The weight now ramps with the candidate's **normalized** position in the pool,
from 0.35 (model authority at the head) to 0.65 (at the tail). Normalized, not
absolute: rank 2 of 3 is the tail of its pool while rank 2 of 30 is the head of
its pool, and the weight has to mean the same thing in both. This keeps reranking
effective on a short candidate list while making a strong top hit hard, but not
impossible, to displace: it now takes a decisive model judgement rather than mere
noise to overturn the retriever.

Measured on a labelled corpus with lexical distractors, reranking a 20-candidate
pool down to 5 with a small local reranker (`qwen3.5:0.8b`, the regime turbograph
actually runs in):

| rerank policy        | recall@5 | MRR       | broke a correct top-1 |
| -------------------- | -------- | --------- | --------------------- |
| none                 | 1.000    | 1.000     | 0                     |
| fixed w=0.7 (old)    | 1.000    | **0.938** | **1 of 8 queries**    |
| position-aware (new) | 1.000    | **1.000** | **0**                 |

The old blend demoted a correct top hit on one query in eight and cost MRR; the
new one broke none. Note the reranker fixed nothing on this corpus (retrieval was
already correct at rank 1 everywhere), so here reranking had only downside, which
is consistent with the situational picture above. Reproduce with
`TG_RERANK=1 TG_RERANK_CHAT=qwen3.5:0.8b go test ./rag/ -run TestRerankBlendBenchmark -v`.
With a larger reranker (`qwen3.5:4b`) both policies scored 1.000: the protection is
latent until the model errs.

### 7. Keeping the contextual prefix OUT of BM25 makes retrieval worse

The July 2026 research sweep proposed splitting `Chunk.IndexText()` into an embed text
(context + body) and a lexical text (body only), on the argument that a contextual prefix
is prose a language model wrote, and putting it into the lexical postings dilutes the
corpus IDF and lets a query match the model's paraphrase instead of the words the passage
actually uses. It called this "probably the cheapest real win" in the set. It is not a win.
It is a loss, and the harness says so immediately:

| BM25 indexes                | chunkRecall@1 | @3    | MRR   |
| --------------------------- | ------------- | ----- | ----- |
| body + contextual prefix    | **0.700**     | 1.000 | 0.850 |
| body only (the proposal)    | 0.500         | 0.900 | 0.694 |

(`TG_AB=1 go test ./server/ -run TestABContextualHardMode`: 8 fragmented documents whose
later chunks lose their subject to a pronoun, plus 16 distractors.)

The reasoning was plausible and the conclusion was backwards. The prefix does not merely
paraphrase the body, it supplies the terms the body is MISSING: a chunk that says "the
cells offer high cycle life" and never says "Borealis" cannot be found by a lexical
search for Borealis unless the prefix puts the word there. That is the whole point of
contextual retrieval, and it is why Anthropic's original work pairs contextual embeddings
with contextual BM25 and reports the pair as the larger win. Removing the prefix from the
lexical channel removes exactly the recall the feature exists to add.

Recorded because the argument is a good one and will be made again.

## Low-storage snapshot modes

Stored embeddings dominate a `.tg` file: on a 173-chunk real-prose corpus at
768 dimensions, the exact float32 vectors are about half of the snapshot. LEANN
([arXiv:2506.08276](https://arxiv.org/abs/2506.08276)) attacks this by storing no
embeddings and recomputing every one per query; that trades away the speed
turbograph is built for, and over Ollama's HTTP embedding it would be far slower
than LEANN's GPU-batched figures. Because turbograph's corpora fit in RAM, two
modes that materialize vectors *once* (not per query) capture most of the storage
win without the latency. Measured with `nomic-embed-text`, recall is the top-10
overlap with the exact-mode ranking (the query vector is identical across modes,
so the delta is purely how much the stored representation perturbs the order):

| `--lean` mode | `.tg` size | load     | recall@10 vs exact |
| ------------- | ---------- | -------- | ------------------ |
| exact (default) | 100%     | 139 ms   | 1.000              |
| codes         | **41.6%**  | 139 ms   | **0.982**          |
| text          | **23.9%**  | 1189 ms  | 1.000              |

`codes` stores the compact TurboQuant codes and decodes them to approximate
vectors on load: ~60% smaller, no load or query penalty, negligible recall loss.
`text` stores no vectors and re-embeds from the chunk text on load: ~76% smaller
and lossless (a deterministic embedder reproduces the vectors exactly), paid for
by re-embedding the whole corpus at load time. Per-query recomputation, LEANN's
actual mechanism, is deliberately not adopted; it only wins past the RAM ceiling,
a scale regime turbograph does not target.

A code is one byte per dimension regardless of `--bits`, so codes-mode storage is
**independent of the bit width** while recall improves with it: raising `--bits`
to 8 lifts codes recall from 0.982 to **0.996** at the same ~41% size, so
`--lean codes --bits 8` is the recommended near-lossless low-storage setting.

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
