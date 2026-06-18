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
- Everything runs locally on CPU through the turbograph library exactly as
  shipped. The only benchmark-specific code is dataset parsing, which lives
  outside the module, so nothing benchmark-specific is in the binary.

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
| 1.0               | —               | —                | 0.35                  |

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
| 1.0 (≈ RRF)       | 0.743           | —                     |

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

## Frontier benchmarks

Two newer benchmarks set the current bar and are noted for context, not run here
(their corpora are large or partly held out): **RTEB** (2025), a generalization-
focused retrieval benchmark with private splits to resist leaderboard
contamination, and **BRIGHT** (2024), reasoning-intensive retrieval where even the
best systems score ~0.22-0.40 nDCG@10. turbograph's local, single-binary footprint
makes the small reasoning split of BRIGHT (`pony`, ~7.9k docs) the natural next
target.

## Reproducing

These are honest local measurements rather than a committed harness (BEIR and
MultiHop-RAG corpora and a running model server do not belong in the repo). To
reproduce: download a dataset, ingest the corpus, retrieve the test queries, score
against the labels (collapse chunks to documents for BEIR, or substring-match the
evidence for MultiHop-RAG). The shipped `turbograph eval` command performs the
same scoring for chunk-level suites.
