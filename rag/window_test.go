package rag

import (
	"context"
	"strings"
	"testing"
)

// TestExpandWindow checks small-to-big neighbor expansion: a retrieved chunk's
// text grows to include adjacent chunks of the same document in positional order,
// stays bounded by the window, and never crosses document boundaries.
func TestExpandWindow(t *testing.T) {
	ctx := context.Background()
	// Two docs, each splitting into several short chunks (small target).
	s := New(newKeywordEmbedder(64), Config{Seed: 1, MinSimilarity: 0.01,
		Chunk: ChunkConfig{Strategy: StrategySentence, TargetWords: 4, OverlapWords: 0}})
	docs := []Document{
		{ID: "d1", Text: "Alpha one two. Beta three four. Gamma five six. Delta seven eight."},
		{ID: "d2", Text: "Other apple pie. Other banana split."},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	// Find a middle chunk of d1.
	var mid Chunk
	for _, c := range s.chunks {
		if c.DocID == "d1" && strings.Contains(c.Text, "Gamma") {
			mid = c
		}
	}
	if mid.ID == "" {
		t.Fatal("did not find the Gamma chunk")
	}
	// window 0 -> just the chunk.
	if got := s.ExpandWindow(mid.ID, 0); !strings.Contains(got, "Gamma") || strings.Contains(got, "Beta") {
		t.Fatalf("window 0 should be the chunk alone, got %q", got)
	}
	// window 1 -> includes the immediate neighbors (Beta before, Delta after).
	w1 := s.ExpandWindow(mid.ID, 1)
	if !strings.Contains(w1, "Beta") || !strings.Contains(w1, "Gamma") || !strings.Contains(w1, "Delta") {
		t.Fatalf("window 1 should include Beta+Gamma+Delta, got %q", w1)
	}
	if strings.Contains(w1, "Alpha") {
		t.Fatalf("window 1 should not reach Alpha (two away), got %q", w1)
	}
	// Never crosses into d2.
	if strings.Contains(s.ExpandWindow(mid.ID, 10), "apple") {
		t.Fatal("expansion crossed a document boundary")
	}
	// Positional order preserved: Beta before Gamma before Delta.
	if !(strings.Index(w1, "Beta") < strings.Index(w1, "Gamma") && strings.Index(w1, "Gamma") < strings.Index(w1, "Delta")) {
		t.Fatalf("neighbors out of order: %q", w1)
	}
}
