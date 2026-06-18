package rag

import (
	"context"
	"strings"
	"testing"
)

func TestIncrementalAdd(t *testing.T) {
	emb := newKeywordEmbedder(96)
	st := New(emb, Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
	ctx := context.Background()
	if err := st.Build(ctx, bridgeCorpus()[:3]); err != nil {
		t.Fatal(err)
	}
	before := st.Len()
	if err := st.AddDocuments(ctx, bridgeCorpus()[3:]); err != nil {
		t.Fatal(err)
	}
	if st.Len() <= before {
		t.Fatalf("AddDocuments did not grow store: %d -> %d", before, st.Len())
	}
	// The newly added omega chunks must now be retrievable.
	res, err := st.Retrieve(ctx, "omega downstream application", RetrieveParams{TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range res {
		if strings.HasPrefix(r.Chunk.DocID, "o") {
			found = true
		}
	}
	if !found {
		t.Error("incrementally added omega chunk not retrieved")
	}
}

func TestIncrementalMatchesBatch(t *testing.T) {
	ctx := context.Background()
	cfg := Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.1, Chunk: ChunkConfig{TargetWords: 200}}

	batch := New(newKeywordEmbedder(96), cfg)
	if err := batch.Build(ctx, bridgeCorpus()); err != nil {
		t.Fatal(err)
	}
	incr := New(newKeywordEmbedder(96), cfg)
	if err := incr.Build(ctx, bridgeCorpus()[:4]); err != nil {
		t.Fatal(err)
	}
	if err := incr.AddDocuments(ctx, bridgeCorpus()[4:]); err != nil {
		t.Fatal(err)
	}
	if batch.Len() != incr.Len() {
		t.Fatalf("len mismatch: batch=%d incr=%d", batch.Len(), incr.Len())
	}
	// Both should retrieve the same top result for an alpha query.
	q := "alpha core principle"
	rb, _ := batch.Retrieve(ctx, q, RetrieveParams{TopK: 1})
	ri, _ := incr.Retrieve(ctx, q, RetrieveParams{TopK: 1})
	if rb[0].Chunk.DocID != ri[0].Chunk.DocID {
		t.Errorf("batch vs incremental top result differs: %s vs %s", rb[0].Chunk.DocID, ri[0].Chunk.DocID)
	}
}

// TestHybridFindsExactTerm checks the value of lexical fusion: a rare exact token
// is retrieved even when the dense embedding of a short query smears it.
func TestHybridFindsExactTerm(t *testing.T) {
	ctx := context.Background()
	docs := []Document{
		{ID: "d1", Text: "the configuration uses parameter zylinder for tuning the system"},
		{ID: "d2", Text: "general notes about systems and tuning and configuration options"},
		{ID: "d3", Text: "unrelated discussion of cooking recipes and kitchen tools"},
	}
	build := func(disable bool) *Store {
		s := New(newKeywordEmbedder(128), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
			DisableLexical: disable, Chunk: ChunkConfig{TargetWords: 200}})
		if err := s.Build(ctx, docs); err != nil {
			t.Fatal(err)
		}
		return s
	}
	hybrid := build(false)
	res, err := hybrid.Retrieve(ctx, "zylinder", RetrieveParams{TopK: 1, GraphMix: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if res[0].Chunk.DocID != "d1" {
		t.Errorf("hybrid failed to surface the exact-term doc: got %s", res[0].Chunk.DocID)
	}
}

// TestMMRReducesRedundancy verifies MMR pulls in a more diverse chunk than pure
// relevance ranking would.
func TestMMRReducesRedundancy(t *testing.T) {
	ctx := context.Background()
	docs := []Document{
		{ID: "x1", Text: "alpha alpha alpha alpha topic one detail"},
		{ID: "x2", Text: "alpha alpha alpha alpha topic one detail copy"},
		{ID: "x3", Text: "alpha alpha alpha alpha topic one detail clone"},
		{ID: "y1", Text: "alpha beta gamma different facet of the subject"},
	}
	s := New(newKeywordEmbedder(128), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
		Chunk: ChunkConfig{TargetWords: 200}})
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	plain, _ := s.Retrieve(ctx, "alpha topic", RetrieveParams{TopK: 2, GraphMix: 0.2})
	diverse, _ := s.Retrieve(ctx, "alpha topic", RetrieveParams{TopK: 2, GraphMix: 0.2, MMRLambda: 0.5})

	hasY := func(rs []Retrieved) bool {
		for _, r := range rs {
			if strings.HasPrefix(r.Chunk.DocID, "y") {
				return true
			}
		}
		return false
	}
	if hasY(plain) {
		t.Log("plain retrieval already diverse; MMR still must not error")
	}
	if !hasY(diverse) {
		t.Error("MMR did not surface the diverse facet chunk y1")
	}
}

func TestRetrieveFilter(t *testing.T) {
	st := newTestStore(t)
	ctx := context.Background()
	res, err := st.Retrieve(ctx, "alpha core principle", RetrieveParams{
		TopK:   5,
		Filter: func(c Chunk) bool { return c.DocID == "bridge" },
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.Chunk.DocID != "bridge" {
			t.Errorf("filter violated: %s", r.Chunk.DocID)
		}
	}
}

func TestCommunitiesDetected(t *testing.T) {
	st := newTestStore(t)
	c := st.Communities()
	if c == nil {
		t.Fatal("no communities")
	}
	if c.NumCommunities() < 1 {
		t.Errorf("expected at least one community, got %d", c.NumCommunities())
	}
}
