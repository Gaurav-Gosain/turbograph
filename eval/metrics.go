// Package eval provides deterministic, dependency-free retrieval-quality
// metrics for evaluating a RAG engine's retriever.
//
// All metrics operate on a ranked list of retrieved document/chunk ids (best
// first) and a set of ids considered relevant for the query. The metrics are
// intentionally LLM-free so that scores are reproducible: the same inputs
// always produce the same outputs, which makes them usable in CI as a
// regression gate on retrieval quality.
//
// Duplicate handling: a retrieved list may contain the same id more than once
// (for example when chunks from the same document are returned). For the
// set-based metrics (Recall@k, Precision@k) duplicates are collapsed so a
// document cannot be double-counted toward a hit; the original ranked order is
// preserved and the first occurrence of an id wins. The rank-sensitive metrics
// (MRR, NDCG@k, ContextPrecision@k) likewise count a relevant id only at its
// first appearance, so repeating a relevant id later does not inflate the
// score. See the per-function comments for specifics.
package eval

import "math"

// dedupeTopK returns the first k entries of retrieved with duplicate ids
// removed, keeping the earliest (highest ranked) occurrence of each id. A
// non-positive k yields an empty slice. This is the shared notion of "top k"
// used by the set-based metrics so that duplicates never count twice.
func dedupeTopK(retrieved []string, k int) []string {
	if k <= 0 {
		return nil
	}
	seen := make(map[string]struct{}, k)
	out := make([]string, 0, k)
	for _, id := range retrieved {
		if len(out) >= k {
			break
		}
		if _, dup := seen[id]; dup {
			// Skip the duplicate but do not consume a top-k slot: we want k
			// distinct documents considered, matching how a user would page
			// through unique results.
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

// RecallAtK returns |topk ∩ relevant| / |relevant|, the fraction of relevant
// documents that appear in the top k retrieved results. It returns 0 when there
// are no relevant documents (the metric is undefined, and 0 is the safe value
// for averaging). Duplicates in retrieved are collapsed before counting.
func RecallAtK(retrieved []string, relevant map[string]struct{}, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	hits := 0
	for _, id := range dedupeTopK(retrieved, k) {
		if _, ok := relevant[id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(relevant))
}

// PrecisionAtK returns |topk ∩ relevant| / k, the fraction of the top k
// retrieved results that are relevant. The denominator is k (the requested cut)
// rather than the number actually retrieved, which is the standard definition:
// failing to return k results is penalized. It returns 0 when k <= 0.
// Duplicates in retrieved are collapsed before counting.
func PrecisionAtK(retrieved []string, relevant map[string]struct{}, k int) float64 {
	if k <= 0 {
		return 0
	}
	hits := 0
	for _, id := range dedupeTopK(retrieved, k) {
		if _, ok := relevant[id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

// MRR returns the reciprocal rank of the first relevant document, 1/rank where
// rank is 1-based, or 0 if no retrieved document is relevant. Duplicate ids are
// collapsed so the rank reflects distinct documents: a repeated id does not
// occupy a separate rank.
func MRR(retrieved []string, relevant map[string]struct{}) float64 {
	seen := make(map[string]struct{}, len(retrieved))
	rank := 0
	for _, id := range retrieved {
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		rank++
		if _, ok := relevant[id]; ok {
			return 1 / float64(rank)
		}
	}
	return 0
}

// NDCGAtK returns the normalized discounted cumulative gain at k using binary
// relevance gains (1 for relevant, 0 otherwise). DCG = sum over positions i of
// gain_i / log2(i+1) for 1-based position i. The ideal DCG (IDCG) ranks all
// relevant documents first, so for binary gains IDCG is the sum of the first
// min(k, |relevant|) discount terms. NDCG = DCG / IDCG, or 0 when IDCG is 0
// (no relevant documents within reach). Duplicate ids are collapsed so a
// document contributes gain at most once, at its first position.
func NDCGAtK(retrieved []string, relevant map[string]struct{}, k int) float64 {
	if k <= 0 || len(relevant) == 0 {
		return 0
	}
	dcg := 0.0
	pos := 0
	seen := make(map[string]struct{}, k)
	for _, id := range retrieved {
		if pos >= k {
			break
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		pos++ // 1-based rank of this distinct document
		if _, ok := relevant[id]; ok {
			dcg += 1 / math.Log2(float64(pos)+1)
		}
	}

	// IDCG: the best possible arrangement puts min(k, |relevant|) relevant
	// documents in the first positions.
	ideal := len(relevant)
	if ideal > k {
		ideal = k
	}
	idcg := 0.0
	for i := 1; i <= ideal; i++ {
		idcg += 1 / math.Log2(float64(i)+1)
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

// ContextPrecisionAtK implements the RAGAS-style context precision metric for
// binary relevance.
//
// Formula: walk the distinct top-k retrieved ids in rank order. At each rank i
// (1-based) where retrieved[i] is relevant, compute Precision@i (the number of
// relevant ids in the first i positions divided by i). Sum those per-hit
// precisions and divide by min(k, |relevant|):
//
//	ContextPrecision@k = ( sum_{i: hit at i} Precision@i ) / min(k, |relevant|)
//
// The normalizer is the number of relevant items reachable within k, which
// rewards placing relevant items early and yields 1.0 only when every reachable
// relevant item is ranked at the very top with no irrelevant items above it. It
// returns 0 when there are no relevant ids, when k <= 0, or when nothing
// relevant appears in the top k. Duplicate ids are collapsed.
func ContextPrecisionAtK(retrieved []string, relevant map[string]struct{}, k int) float64 {
	if k <= 0 || len(relevant) == 0 {
		return 0
	}
	relevantSoFar := 0
	sum := 0.0
	pos := 0
	seen := make(map[string]struct{}, k)
	for _, id := range retrieved {
		if pos >= k {
			break
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		pos++ // 1-based rank
		if _, ok := relevant[id]; ok {
			relevantSoFar++
			// Precision@pos at this hit.
			sum += float64(relevantSoFar) / float64(pos)
		}
	}
	if relevantSoFar == 0 {
		return 0
	}
	denom := len(relevant)
	if denom > k {
		denom = k
	}
	return sum / float64(denom)
}
