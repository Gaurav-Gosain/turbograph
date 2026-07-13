package rag

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

type fakeGen struct {
	out string
	err error
}

func (f fakeGen) Generate(_ context.Context, _, _ string) (string, error) { return f.out, f.err }

func mkRes(ids ...string) []Retrieved {
	out := make([]Retrieved, len(ids))
	for i, id := range ids {
		out[i] = Retrieved{Chunk: Chunk{ID: id, Text: id + " body text"}, Score: float32(len(ids) - i), Similarity: 0.9}
	}
	return out
}

func TestRerankReorders(t *testing.T) {
	res := mkRes("a", "b", "c") // base order a,b,c
	// Model says c is best, then b, then a.
	gen := fakeGen{out: "here you go: [{\"i\":0,\"score\":1},{\"i\":1,\"score\":5},{\"i\":2,\"score\":9}]"}
	got := Rerank(context.Background(), gen, "q", res, 3)
	if got[0].Chunk.ID != "c" {
		t.Errorf("expected c first after rerank, got %s (%v)", got[0].Chunk.ID, []string{got[0].Chunk.ID, got[1].Chunk.ID, got[2].Chunk.ID})
	}
}

func TestRerankFailOpen(t *testing.T) {
	res := mkRes("a", "b", "c")
	// Error from the model: must return the base order, truncated.
	got := Rerank(context.Background(), fakeGen{err: errors.New("boom")}, "q", res, 2)
	if len(got) != 2 || got[0].Chunk.ID != "a" {
		t.Errorf("fail-open should keep base order: %v", got)
	}
	// Garbage output: same fail-open behavior.
	got = Rerank(context.Background(), fakeGen{out: "I cannot do that"}, "q", res, 2)
	if len(got) != 2 || got[0].Chunk.ID != "a" {
		t.Errorf("unparseable should keep base order: %v", got)
	}
}

func TestShouldAbstain(t *testing.T) {
	if !ShouldAbstain(nil, 0.5) {
		t.Error("empty results should abstain")
	}
	if ShouldAbstain(mkRes("a"), 0) {
		t.Error("threshold 0 should never abstain")
	}
	low := []Retrieved{{Chunk: Chunk{ID: "a"}, Similarity: 0.2}}
	if !ShouldAbstain(low, 0.5) {
		t.Error("below threshold should abstain")
	}
	hi := []Retrieved{{Chunk: Chunk{ID: "a"}, Similarity: 0.8}}
	if ShouldAbstain(hi, 0.5) {
		t.Error("above threshold should not abstain")
	}
}

// scoredRes builds a candidate pool in retrieval-rank order with the given
// retrieval scores (descending), as the store would hand to Rerank.
func scoredRes(scores ...float32) []Retrieved {
	out := make([]Retrieved, len(scores))
	for i, s := range scores {
		id := fmt.Sprintf("c%d", i)
		out[i] = Retrieved{Chunk: Chunk{ID: id, Text: id + " body"}, Score: s, Similarity: 0.9}
	}
	return out
}

// modelScores builds the reranker's JSON reply from a score per candidate.
func modelScores(scores ...float32) string {
	parts := make([]string, len(scores))
	for i, s := range scores {
		parts[i] = fmt.Sprintf(`{"i":%d,"score":%g}`, i, s)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// TestRerankResistsTailHijack is the defect this blend exists to fix. A pool of 20
// where the retriever is confident about c0, and the model mildly prefers a weak
// tail candidate. Under the old fixed 0.7 model weight, c15 would take the top
// slot on the model's word alone despite weak retrieval support. Noise must not
// override a strong hit.
func TestRerankResistsTailHijack(t *testing.T) {
	base := make([]float32, 20)
	for i := range base {
		base[i] = float32(20-i) / 20 // 1.00 down to 0.05
	}
	res := scoredRes(base...)

	m := make([]float32, 20)
	m[0] = 6 // model is lukewarm about the strong top hit
	m[15] = 10
	got := Rerank(context.Background(), fakeGen{out: modelScores(m...)}, "q", res, 5)

	if got[0].Chunk.ID != "c0" {
		t.Fatalf("a lukewarm model judgement hijacked the top hit: got %s first", got[0].Chunk.ID)
	}
	// Sanity: with the old fixed 0.7 model weight this exact fixture inverts, which
	// is why the position-aware blend exists. The model scores normalize to 0.6 for
	// the top hit and 1.0 for the tail candidate; retrieval to 1.00 and 0.25.
	oldTop := 0.7*0.6 + 0.3*1.00  // 0.720
	oldTail := 0.7*1.0 + 0.3*0.25 // 0.775
	if oldTop >= oldTail {
		t.Fatal("test no longer reproduces the old pathology; recheck the fixture")
	}
}

// TestRerankStillOverturnsOnDecisiveJudgment guards the other side: the blend must
// protect the head from noise without neutering the reranker. When the model
// decisively rejects the top hit and backs a well-retrieved candidate, it wins.
func TestRerankStillOverturnsOnDecisiveJudgment(t *testing.T) {
	res := scoredRes(1.0, 0.95, 0.9, 0.85, 0.8)
	m := []float32{0, 1, 1, 2, 10} // top is judged irrelevant; c4 is judged perfect
	got := Rerank(context.Background(), fakeGen{out: modelScores(m...)}, "q", res, 3)
	if got[0].Chunk.ID != "c4" {
		t.Fatalf("a decisive model judgement should overturn a rejected top hit, got %s", got[0].Chunk.ID)
	}
}

// TestBlendWeight pins the shape: bounded, monotonically increasing with depth,
// and normalized to the pool so the same rank means different things in pools of
// different sizes.
func TestBlendWeight(t *testing.T) {
	const head, tail = float32(0.35), float32(0.65)
	if w := blendWeight(0, 20); w != head {
		t.Errorf("head weight = %v, want %v", w, head)
	}
	if w := blendWeight(19, 20); w != tail {
		t.Errorf("tail weight = %v, want %v", w, tail)
	}
	if w := blendWeight(0, 1); w != head {
		t.Errorf("degenerate pool should use the head weight, got %v", w)
	}
	// Monotonic in depth.
	prev := float32(-1)
	for i := range 20 {
		w := blendWeight(i, 20)
		if w < prev {
			t.Fatalf("weight decreased at rank %d: %v < %v", i, w, prev)
		}
		if w < head || w > tail {
			t.Fatalf("weight out of bounds at rank %d: %v", i, w)
		}
		prev = w
	}
	// Normalized to the pool: rank 2 is the tail of a 3-pool but the head of a 30-pool.
	if blendWeight(2, 3) <= blendWeight(2, 30) {
		t.Error("weight must be normalized to pool size, not absolute rank")
	}
}
