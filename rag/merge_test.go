package rag

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func mergeStore(t *testing.T, docs ...Document) *Store {
	t.Helper()
	s := New(newKeywordEmbedder(32), Config{Seed: 1})
	if err := s.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return s
}

// TestMergeCombinesCorpora: the point of a shareable .tg. Two stores built
// separately become one, and the merged store retrieves from both.
func TestMergeCombinesCorpora(t *testing.T) {
	a := mergeStore(t, Document{ID: "a1", Text: "the caldera reactor was built in northgate"})
	b := mergeStore(t, Document{ID: "b1", Text: "project helios is led by mira tan at verdant labs"})

	st, err := Merge(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if st.Documents != 1 || st.Chunks < 1 {
		t.Fatalf("merge stats: %+v", st)
	}
	if a.DocCount() != 2 {
		t.Fatalf("merged store has %d documents, want 2", a.DocCount())
	}
	// Both corpora are retrievable from the merged store.
	for _, q := range []string{"caldera reactor northgate", "helios mira tan"} {
		res, err := a.Retrieve(context.Background(), q, RetrieveParams{TopK: 2})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 {
			t.Errorf("no results for %q after merge", q)
		}
	}
}

// TestMergeIsIdempotent: merging the same store twice must not duplicate it. An
// agent that re-runs a merge should not end up with the corpus twice over.
func TestMergeIsIdempotent(t *testing.T) {
	a := mergeStore(t, Document{ID: "a1", Text: "alpha content here"})
	b := mergeStore(t, Document{ID: "b1", Text: "beta content here"})

	if _, err := Merge(a, b); err != nil {
		t.Fatal(err)
	}
	docs, chunks := a.DocCount(), a.Len()

	st, err := Merge(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if st.Documents != 0 || st.Skipped != 1 {
		t.Errorf("second merge should add nothing and skip 1, got %+v", st)
	}
	if a.DocCount() != docs || a.Len() != chunks {
		t.Errorf("second merge duplicated content: %d docs / %d chunks, want %d / %d",
			a.DocCount(), a.Len(), docs, chunks)
	}
}

// TestMergeSkipsSameContentUnderDifferentID: two people indexing the same source
// file under different names must not double-count it.
func TestMergeSkipsSameContentUnderDifferentID(t *testing.T) {
	same := "the same document text, indexed twice under different names"
	a := mergeStore(t, Document{ID: "mine.md", Text: same})
	b := mergeStore(t, Document{ID: "theirs.md", Text: same})

	st, err := Merge(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if st.Documents != 0 || st.Skipped != 1 {
		t.Errorf("identical content under a different id should be skipped, got %+v", st)
	}
	if a.DocCount() != 1 {
		t.Errorf("merged store has %d documents, want 1", a.DocCount())
	}
}

// TestMergeRejectsMismatchedDim: two stores built with different embedding models
// cannot be merged, and must say so rather than produce a corrupt index.
func TestMergeRejectsMismatchedDim(t *testing.T) {
	a := mergeStore(t, Document{ID: "a", Text: "alpha"})
	b := New(newKeywordEmbedder(64), Config{Seed: 1})
	if err := b.Build(context.Background(), []Document{{ID: "b", Text: "beta"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := Merge(a, b); err == nil {
		t.Fatal("merging stores of different dimension should fail")
	}
}

// TestMergeCarriesExtractionCache: the merged store must not have to re-read every
// chunk with the model to rebuild its entity graph.
func TestMergeCarriesExtractionCache(t *testing.T) {
	a := mergeStore(t, Document{ID: "a1", Text: "alpha"})
	b := mergeStore(t, Document{ID: "b1", Text: "beta"})
	ex := &countingExtractor{}
	opt := EntityBuildOptions{Model: "m1"}
	for _, s := range []*Store{a, b} {
		if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
			t.Fatal(err)
		}
	}
	before := ex.calls.Load()

	st, err := Merge(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if st.Cached == 0 {
		t.Error("merge carried no extraction cache")
	}
	// Rebuilding the merged store's entity graph must cost nothing.
	if err := a.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if extra := ex.calls.Load() - before; extra != 0 {
		t.Errorf("rebuilding the merged graph made %d model calls; the merged cache should cover it", extra)
	}
	if a.EntityCount() < 2 {
		t.Errorf("merged entity graph has %d entities, want both stores' entities", a.EntityCount())
	}
}

// TestMergeSurvivesRoundTrip: a merged store saves and loads like any other.
func TestMergeSurvivesRoundTrip(t *testing.T) {
	a := mergeStore(t, Document{ID: "a1", Text: "alpha content"})
	b := mergeStore(t, Document{ID: "b1", Text: "beta content"})
	if _, err := Merge(a, b); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := a.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(newKeywordEmbedder(32), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.DocCount() != 2 || loaded.Len() != a.Len() {
		t.Fatalf("merged store did not round-trip: %d docs / %d chunks", loaded.DocCount(), loaded.Len())
	}
}

// fpEmbedder is a keyword embedder that also names its vector space, like the real
// clients do.
type fpEmbedder struct {
	*keywordEmbedder
	name string
}

func (e fpEmbedder) Fingerprint() string { return e.name }

// TestMergeRefusesDifferentEmbedders is the case a dimension check cannot catch, and
// the one that matters: two models at the SAME dimension do not agree on which
// dimensions. Merging them produces an index whose distances are meaningless. Nothing
// errors, retrieval just quietly gets worse and stays that way, in a file you shared.
func TestMergeRefusesDifferentEmbedders(t *testing.T) {
	mk := func(model, id, text string) *Store {
		s := New(fpEmbedder{newKeywordEmbedder(32), model}, Config{Seed: 1})
		if err := s.Build(context.Background(), []Document{{ID: id, Text: text}}); err != nil {
			t.Fatal(err)
		}
		return s
	}
	a := mk("ollama/nomic-embed-text/dim=0", "a", "alpha content")
	b := mk("ollama/bge-base/dim=0", "b", "beta content") // same dimension, different space

	if a.Encoder() == "" || b.Encoder() == "" {
		t.Fatal("the store did not record its encoder")
	}
	_, err := Merge(a, b)
	if err == nil {
		t.Fatal("merging stores built with different embedders must fail; it silently corrupts the vector space")
	}
	if !strings.Contains(err.Error(), "different embedders") {
		t.Errorf("unhelpful error: %v", err)
	}
	// The same embedder still merges.
	c := mk("ollama/nomic-embed-text/dim=0", "c", "gamma content")
	if _, err := Merge(a, c); err != nil {
		t.Errorf("stores built with the same embedder must merge: %v", err)
	}
}

// TestMergeAllowsUnfingerprintedStores: a .tg written before fingerprints has no
// encoder recorded. It must still merge, with only the dimension check, rather than
// becoming unusable.
func TestMergeAllowsUnfingerprintedStores(t *testing.T) {
	a := mergeStore(t, Document{ID: "a", Text: "alpha"}) // keywordEmbedder: no Fingerprint
	b := mergeStore(t, Document{ID: "b", Text: "beta"})
	if a.Encoder() != "" {
		t.Fatalf("expected no encoder, got %q", a.Encoder())
	}
	if _, err := Merge(a, b); err != nil {
		t.Errorf("stores without a fingerprint must still merge: %v", err)
	}
}
