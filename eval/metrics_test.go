package eval

import (
	"math"
	"testing"
)

// set is a test helper building a relevant-id set.
func set(ids ...string) map[string]struct{} {
	m := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		m[id] = struct{}{}
	}
	return m
}

// approx asserts two floats are equal within a small tolerance.
func approx(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

func TestRecallAtK(t *testing.T) {
	// Ranking: [a b c d e]; relevant = {a, c, x}. Top 3 = [a b c], hits a,c.
	// Recall@3 = 2/3.
	retrieved := []string{"a", "b", "c", "d", "e"}
	rel := set("a", "c", "x")
	approx(t, "Recall@3", RecallAtK(retrieved, rel, 3), 2.0/3.0)
	// Top 5 includes a and c only (x never retrieved): 2/3 still.
	approx(t, "Recall@5", RecallAtK(retrieved, rel, 5), 2.0/3.0)
	// Top 1 = [a]: 1/3.
	approx(t, "Recall@1", RecallAtK(retrieved, rel, 1), 1.0/3.0)
}

func TestRecallEmptyRelevant(t *testing.T) {
	approx(t, "Recall empty relevant", RecallAtK([]string{"a"}, set(), 3), 0)
}

func TestPrecisionAtK(t *testing.T) {
	// Top 3 = [a b c]; relevant = {a, c, x}: 2 hits / 3 = 2/3.
	retrieved := []string{"a", "b", "c", "d", "e"}
	rel := set("a", "c", "x")
	approx(t, "Precision@3", PrecisionAtK(retrieved, rel, 3), 2.0/3.0)
	// Top 1 = [a]: 1 hit / 1 = 1.
	approx(t, "Precision@1", PrecisionAtK(retrieved, rel, 1), 1.0)
	// k larger than retrieved: denominator is k=10, 2 hits => 0.2.
	approx(t, "Precision@10", PrecisionAtK(retrieved, rel, 10), 0.2)
}

func TestPrecisionKZero(t *testing.T) {
	approx(t, "Precision@0", PrecisionAtK([]string{"a"}, set("a"), 0), 0)
}

func TestMRR(t *testing.T) {
	// First relevant at rank 3.
	retrieved := []string{"x", "y", "rel", "z"}
	approx(t, "MRR rank 3", MRR(retrieved, set("rel")), 1.0/3.0)
	// First relevant at rank 1.
	approx(t, "MRR rank 1", MRR([]string{"rel", "y"}, set("rel")), 1.0)
	// None relevant.
	approx(t, "MRR none", MRR(retrieved, set("nope")), 0)
	// Empty retrieved.
	approx(t, "MRR empty", MRR(nil, set("rel")), 0)
}

func TestNDCGAtK(t *testing.T) {
	// Suboptimal order: relevant at ranks 2 and 4 of [a b c d], relevant={b,d}.
	// DCG = 1/log2(3) + 1/log2(5).
	// IDCG (2 relevant, k=4) = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3).
	retrieved := []string{"a", "b", "c", "d"}
	rel := set("b", "d")
	dcg := 1/math.Log2(3) + 1/math.Log2(5)
	idcg := 1/math.Log2(2) + 1/math.Log2(3)
	approx(t, "NDCG@4 suboptimal", NDCGAtK(retrieved, rel, 4), dcg/idcg)

	// Perfect order: relevant first => NDCG = 1.
	perfect := []string{"b", "d", "a", "c"}
	approx(t, "NDCG@4 perfect", NDCGAtK(perfect, rel, 4), 1.0)

	// No relevant => 0.
	approx(t, "NDCG no relevant", NDCGAtK(retrieved, set("z"), 4), 0)
	// k=0 => 0.
	approx(t, "NDCG k0", NDCGAtK(retrieved, rel, 0), 0)
}

func TestContextPrecisionAtK(t *testing.T) {
	// Worked example: relevant items at ranks 1 and 3 of 4.
	// retrieved = [r, x, r2, y], relevant = {r, r2}, k = 4.
	// Rank 1 (hit): Precision@1 = 1/1 = 1.
	// Rank 3 (hit): 2 relevant in first 3 => Precision@3 = 2/3.
	// sum = 1 + 2/3 = 5/3.
	// denom = min(k=4, |relevant|=2) = 2.
	// ContextPrecision@4 = (5/3) / 2 = 5/6 ≈ 0.8333333.
	retrieved := []string{"r", "x", "r2", "y"}
	rel := set("r", "r2")
	approx(t, "ContextPrecision@4 worked", ContextPrecisionAtK(retrieved, rel, 4), 5.0/6.0)

	// Perfect ordering: both relevant first.
	// Rank 1: 1/1 = 1. Rank 2: 2/2 = 1. sum=2, denom=2 => 1.0.
	approx(t, "ContextPrecision perfect", ContextPrecisionAtK([]string{"r", "r2", "x", "y"}, rel, 4), 1.0)

	// None relevant in top k.
	approx(t, "ContextPrecision none", ContextPrecisionAtK([]string{"x", "y"}, rel, 2), 0)
	// k=0 => 0.
	approx(t, "ContextPrecision k0", ContextPrecisionAtK(retrieved, rel, 0), 0)
	// Empty relevant => 0.
	approx(t, "ContextPrecision empty rel", ContextPrecisionAtK(retrieved, set(), 4), 0)
}

func TestEdgeCasesEmptyRetrieved(t *testing.T) {
	rel := set("a")
	approx(t, "Recall empty retrieved", RecallAtK(nil, rel, 3), 0)
	approx(t, "Precision empty retrieved", PrecisionAtK(nil, rel, 3), 0)
	approx(t, "NDCG empty retrieved", NDCGAtK(nil, rel, 3), 0)
	approx(t, "ContextPrecision empty retrieved", ContextPrecisionAtK(nil, rel, 3), 0)
}

func TestKLargerThanRetrieved(t *testing.T) {
	// retrieved has 2 items, k=5. relevant={a}. a is at rank 1.
	retrieved := []string{"a", "b"}
	rel := set("a")
	approx(t, "Recall@5 short", RecallAtK(retrieved, rel, 5), 1.0)       // 1 of 1 relevant found
	approx(t, "Precision@5 short", PrecisionAtK(retrieved, rel, 5), 0.2) // 1 hit / k=5
	// NDCG: DCG = 1/log2(2) = 1. IDCG (1 relevant) = 1. => 1.
	approx(t, "NDCG@5 short", NDCGAtK(retrieved, rel, 5), 1.0)
}

func TestDuplicateDedup(t *testing.T) {
	// Duplicates must not double-count. retrieved = [a a b], relevant = {a, b}.
	// Distinct top-2 = [a b]: both relevant.
	dup := []string{"a", "a", "b"}
	rel := set("a", "b")
	// Recall@2 over distinct: hits a,b => 2/2 = 1.
	approx(t, "Recall@2 dup", RecallAtK(dup, rel, 2), 1.0)
	// Precision@2: distinct top-2 = [a b], 2 hits / 2 = 1.
	approx(t, "Precision@2 dup", PrecisionAtK(dup, rel, 2), 1.0)

	// MRR with leading duplicate of irrelevant id: [x x rel].
	// Distinct ranks: x=1, rel=2 => 1/2.
	approx(t, "MRR dup", MRR([]string{"x", "x", "rel"}, set("rel")), 0.5)

	// NDCG with duplicate relevant: [a a b], relevant={a,b}.
	// Distinct ranks: a=1, b=2. DCG = 1/log2(2) + 1/log2(3).
	// IDCG (2 relevant) = same = 1.0.
	approx(t, "NDCG dup", NDCGAtK(dup, rel, 3), 1.0)

	// ContextPrecision with duplicate: [a a b], relevant={a,b}, k=3.
	// Distinct: rank1 a (hit) 1/1; rank2 b (hit) 2/2. sum=2, denom=2 => 1.
	approx(t, "ContextPrecision dup", ContextPrecisionAtK(dup, rel, 3), 1.0)
}

func TestNegativeK(t *testing.T) {
	rel := set("a")
	approx(t, "Recall negative k", RecallAtK([]string{"a"}, rel, -1), 0)
	approx(t, "Precision negative k", PrecisionAtK([]string{"a"}, rel, -1), 0)
	approx(t, "NDCG negative k", NDCGAtK([]string{"a"}, rel, -1), 0)
	approx(t, "ContextPrecision negative k", ContextPrecisionAtK([]string{"a"}, rel, -1), 0)
}
