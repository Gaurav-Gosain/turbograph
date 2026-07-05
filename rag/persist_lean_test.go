package rag

import (
	"bytes"
	"context"
	"testing"
)

// TestVectorModesRoundTrip checks the lean save modes deterministically (no model):
// VectorsNone must be lossless because the keyword embedder is deterministic, so
// re-embedding reproduces the exact vectors; VectorsCodes must round-trip to a
// working index that still ranks the right chunk first.
func TestVectorModesRoundTrip(t *testing.T) {
	ctx := context.Background()
	docs := []Document{
		{ID: "space", Text: "rockets reach orbit by burning propellant for thrust in microgravity"},
		{ID: "cook", Text: "sourdough needs a ripe starter long fermentation and a hot oven to bake"},
		{ID: "money", Text: "central banks raise interest rates to curb inflation and bond yields rise"},
		{ID: "code", Text: "a hash map stores key value pairs with average constant time lookup"},
	}
	build := func() *Store {
		s := New(newKeywordEmbedder(128), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.02})
		if err := s.Build(ctx, docs); err != nil {
			t.Fatal(err)
		}
		return s
	}
	top1 := func(s *Store, q string) string {
		res, err := s.Retrieve(ctx, q, RetrieveParams{TopK: 1})
		if err != nil || len(res) == 0 {
			t.Fatalf("retrieve %q: %v", q, err)
		}
		return res[0].Chunk.DocID
	}
	base := build()
	q := "hash map constant time lookup key value"
	want := top1(base, q)

	for _, m := range []struct {
		name string
		mode VectorMode
	}{{"exact", VectorsExact}, {"codes", VectorsCodes}, {"none", VectorsNone}} {
		var buf bytes.Buffer
		if err := base.SaveLean(&buf, m.mode); err != nil {
			t.Fatalf("save %s: %v", m.name, err)
		}
		st, err := Load(newKeywordEmbedder(128), &buf)
		if err != nil {
			t.Fatalf("load %s: %v", m.name, err)
		}
		if st.Len() != base.Len() {
			t.Fatalf("%s: chunk count %d != %d", m.name, st.Len(), base.Len())
		}
		if got := top1(st, q); got != want {
			t.Errorf("%s: top-1 %q, want %q", m.name, got, want)
		}
		// VectorsNone re-embeds deterministically, so vectors must be byte-identical.
		if m.mode == VectorsNone {
			st.mu.RLock()
			base.mu.RLock()
			for i := range st.embeds {
				for j := range st.embeds[i] {
					if st.embeds[i][j] != base.embeds[i][j] {
						t.Fatalf("none mode not lossless at chunk %d dim %d", i, j)
					}
				}
			}
			base.mu.RUnlock()
			st.mu.RUnlock()
		}
	}
}
