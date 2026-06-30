package rag

import (
	"context"
	"testing"
)

// TestScoreComponentsSumToScore verifies the explainability breakdown is honest:
// the four additive signals reconstruct the blended score, across the fast hybrid
// path and the graph/entity path.
func TestScoreComponentsSumToScore(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.02})
	docs := []Document{
		{ID: "a", Text: "graphs connect nodes with edges and vertices"},
		{ID: "b", Text: "vectors are embedded and quantized for search"},
		{ID: "c", Text: "edges link related nodes in a similarity graph"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	check := func(name string, p RetrieveParams) {
		res, err := s.Retrieve(ctx, "nodes and edges in a graph", p)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(res) == 0 {
			t.Fatalf("%s: no results", name)
		}
		for _, r := range res {
			c := r.Components
			sum := c.Dense + c.Lexical + c.Graph + c.Entity
			if diff := sum - r.Score; diff > 1e-4 || diff < -1e-4 {
				t.Errorf("%s: components %v sum to %.5f, score is %.5f (chunk %s)", name, c, sum, r.Score, r.Chunk.ID)
			}
		}
	}
	check("hybrid", RetrieveParams{TopK: 3})
	check("graph", RetrieveParams{TopK: 3, GraphMix: 0.4})
}
