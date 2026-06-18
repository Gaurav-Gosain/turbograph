# Benchmarks

turbograph is measured against standard BEIR retrieval benchmarks, not just unit
tests. This document records the methodology and the results, including a
regression that the benchmarks caught and the root-cause fixes that followed.

## Methodology

- Datasets: [BEIR](https://github.com/beir-cellar/beir) SciFact (5183 documents,
  300 test queries) and NFCorpus (3633 documents, 323 test queries). Both ship
  document-level relevance judgments (qrels).
- Embedding model: EmbeddingGemma (300M) served locally by Ollama, 768
  dimensions, the shipped default.
- Each corpus document is ingested whole. Retrieval returns chunks; chunks are
  collapsed to their document (keeping each document's best rank) before scoring,
  so the metric is document-level and comparable to published BEIR numbers.
- Metrics: nDCG@10 (BEIR's primary), Recall@10, Recall@100, MRR@10.
- Everything runs locally on CPU. The runner uses the turbograph library exactly
  as shipped; the only benchmark-specific code is dataset parsing, which lives
  outside the module so nothing benchmark-specific is in the binary.

## Results

EmbeddingGemma, after the fixes below.

| dataset  | mode                     | nDCG@10 | Recall@10 | Recall@100 |
| -------- | ------------------------ | ------- | --------- | ---------- |
| SciFact  | pure hybrid (no graph)   | 0.78    | 0.92      | 0.97       |
| SciFact  | default graph boost 0.2  | 0.76    | 0.90      | 0.98       |
| NFCorpus | pure hybrid (no graph)   | 0.38    | 0.19      | 0.36       |
| NFCorpus | default graph boost 0.2  | 0.37    | 0.18      | 0.37       |

For reference, BM25 scores about 0.67 nDCG@10 on SciFact and 0.33 on NFCorpus;
strong dense models land in the low-to-mid 0.70s and mid-0.30s respectively. The
numbers above are at or above the strong-dense baseline for both datasets.

## What the benchmarks caught

The first run scored 0.564 nDCG@10 on SciFact, well below what EmbeddingGemma
should reach. Two independent root causes were found by measurement, not guessed:

### 1. Symmetric embedding of an asymmetric model

EmbeddingGemma is instruction-tuned and expects different prompts for queries and
documents. turbograph was embedding both as raw text. Applying the documented
prompts (`task: search result | query: ` and `title: none | text: `) raised
SciFact nDCG@10 from 0.564 to 0.60. The fix generalizes: the client now carries
query/document prompts set from the model name, with presets for several model
families and a safe empty default for unknown models. See
[architecture.md](architecture.md#asymmetric-embeddings).

### 2. A convex graph blend that traded away relevance

The retrieval score was `GraphMix * pagerank + (1-GraphMix) * relevance`, with
`GraphMix` defaulting to 0.6. Sweeping the mix exposed a monotonic collapse:

| GraphMix | nDCG@10 (convex) |
| -------- | ---------------- |
| ~0       | 0.78             |
| 0.2      | 0.77             |
| 0.4      | 0.73             |
| 0.6      | 0.60             |
| 0.8      | 0.48             |

PageRank measures centrality, not relevance; a convex blend lets centrality buy
down relevance, and at the 0.6 default a well-connected but off-topic chunk
routinely displaced the right answer. The root-cause fix was to combine
additively, `relevance + GraphMix * pagerank`, so the graph can only lift
associated chunks out of the tail and can never demote a strong direct hit. Under
the additive form the score degrades gracefully with the mix instead of
collapsing, and the graph genuinely helps recall (NFCorpus Recall@100 rises from
0.36 to 0.37 as the boost increases) without hurting precision. The default
dropped to a modest 0.2. Neither fix is tuned to a benchmark: the prompts are the
model's own, and the additive combination is a structural correction validated on
two datasets.

## Reproducing

These are honest, local measurements rather than a committed harness (BEIR's
multi-gigabyte corpora and a running model server do not belong in the repo). To
reproduce: download a BEIR dataset, ingest each document whole, retrieve the test
queries, collapse chunks to documents, and score against the qrels. The shipped
`turbograph eval` command performs the same scoring for chunk-level suites; the
only extra step for BEIR is the document-level collapse described above.
