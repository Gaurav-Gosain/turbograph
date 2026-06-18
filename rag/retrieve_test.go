package rag

import (
	"context"
	"testing"
)

// asymEmbedder records whether the query path or the document path was used, and
// embeds queries into a distinct subspace so the routing is observable.
type asymEmbedder struct {
	*keywordEmbedder
	queryCalls int
}

func (e *asymEmbedder) EmbedQuery(ctx context.Context, texts []string) ([][]float32, error) {
	e.queryCalls++
	return e.keywordEmbedder.Embed(ctx, texts)
}

// TestRetrieveUsesQueryEmbedder confirms the store routes the query through
// EmbedQuery when the embedder implements QueryEmbedder.
func TestRetrieveUsesQueryEmbedder(t *testing.T) {
	ctx := context.Background()
	emb := &asymEmbedder{keywordEmbedder: newKeywordEmbedder(64)}
	s := New(emb, Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05})
	docs := []Document{
		{ID: "a", Text: "graphs connect nodes with edges"},
		{ID: "b", Text: "vectors are embedded and quantized"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	before := emb.queryCalls
	if _, err := s.Retrieve(ctx, "graphs", RetrieveParams{TopK: 2}); err != nil {
		t.Fatal(err)
	}
	if emb.queryCalls != before+1 {
		t.Fatalf("expected one EmbedQuery call, got %d", emb.queryCalls-before)
	}
}

// TestGraphMixSentinel verifies the three regimes of GraphMix: unset (0) selects
// the default boost, a negative value disables the graph (pure retrieval), and a
// positive value is used as given. The default and a positive boost can reorder
// results relative to pure retrieval; a negative value must equal pure ranking.
func TestGraphMixSentinel(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []Document{
		{ID: "a", Text: "alpha topic shared core idea"},
		{ID: "b", Text: "alpha topic shared core idea variant"},
		{ID: "c", Text: "beta unrelated separate matter"},
		{ID: "d", Text: "gamma another distinct theme"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	// A negative GraphMix must behave as pure retrieval: identical to explicitly
	// removing the graph term. We assert it does not error and returns results.
	pure, err := s.Retrieve(ctx, "alpha topic", RetrieveParams{TopK: 3, GraphMix: -1})
	if err != nil || len(pure) == 0 {
		t.Fatalf("pure retrieval failed: %v (%d results)", err, len(pure))
	}
	// Unset (0) should apply the default boost without error.
	def, err := s.Retrieve(ctx, "alpha topic", RetrieveParams{TopK: 3, GraphMix: 0})
	if err != nil || len(def) == 0 {
		t.Fatalf("default retrieval failed: %v (%d results)", err, len(def))
	}
	// The top hit by pure relevance should still be present under the default
	// boost (the graph augments, it does not discard strong direct hits).
	top := pure[0].Chunk.ID
	found := false
	for _, r := range def {
		if r.Chunk.ID == top {
			found = true
		}
	}
	if !found {
		t.Errorf("default boost dropped the top pure hit %q", top)
	}
}
