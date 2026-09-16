package rag

import (
	"context"
	"testing"
)

// TestIncrementalCommunitiesPreserveLabels covers the store-level contract of
// incremental community detection: adding documents extends the partition in
// place, so a community the new documents did not disturb keeps its id and its
// members instead of a from-scratch recompute renumbering it.
//
// The corpus is three documents that share vocabulary (one community) plus an
// isolated document (its own community). The appended document is similar enough
// to both to bridge them, so a from-scratch recompute collapses everything into
// one community and renumbers it; the incremental path keeps the clique's
// original id (and its now-empty sibling slot) intact.
func TestIncrementalCommunitiesPreserveLabels(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(256), Config{Seed: 7, GraphKNN: 4, MinSimilarity: 0.5})
	docs := []Document{
		{ID: "solo", Text: "solo alpha beta unique"},
		{ID: "r", Text: "clique share common bridge red"},
		{ID: "b", Text: "clique share common bridge blue"},
		{ID: "g", Text: "clique share common bridge green"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	prev := s.Communities()
	if prev == nil || prev.NodeCount() != 4 {
		t.Fatalf("build partition: %+v", prev)
	}
	if prev.Label(1) != prev.Label(2) || prev.Label(1) != prev.Label(3) {
		t.Fatalf("clique docs not unified at build: %d %d %d",
			prev.Label(1), prev.Label(2), prev.Label(3))
	}
	if prev.Label(0) == prev.Label(1) {
		t.Fatalf("solo doc merged into the clique at build")
	}

	// The new document shares the clique vocabulary and enough of the solo
	// document's to attach to each, joining the two clusters.
	if err := s.AddDocuments(ctx, []Document{{ID: "n", Text: "clique share common bridge solo alpha beta"}}); err != nil {
		t.Fatal(err)
	}
	cur := s.Communities()
	if cur.NodeCount() != 5 {
		t.Fatalf("NodeCount after add = %d, want 5", cur.NodeCount())
	}
	for _, node := range []int{1, 2, 3} {
		if cur.Label(node) != prev.Label(node) {
			t.Fatalf("clique node %d changed community id: %d -> %d",
				node, prev.Label(node), cur.Label(node))
		}
	}
	if cur.Label(1) != prev.Label(1) {
		t.Fatalf("clique community id drifted: %d -> %d", prev.Label(1), cur.Label(1))
	}
	// Preserving the clique's id means its now-empty sibling slot is preserved
	// too: a from-scratch recompute would have collapsed the two into one
	// community and numbered it 0.
	if cur.NumCommunities() != prev.NumCommunities() {
		t.Fatalf("NumCommunities = %d, want %d (preserved ids)", cur.NumCommunities(), prev.NumCommunities())
	}
}
