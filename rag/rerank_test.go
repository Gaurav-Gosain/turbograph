package rag

import (
	"context"
	"errors"
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
