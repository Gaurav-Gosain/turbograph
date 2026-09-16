package rag

import (
	"context"
	"testing"
)

// metaTestStore builds a small store whose three documents differ only by
// metadata, so the retrieval predicate is the only thing that can tell them apart.
func metaTestStore(t *testing.T) *Store {
	t.Helper()
	s := New(newKeywordEmbedder(32), Config{Seed: 1, MinSimilarity: 0.01})
	docs := []Document{
		{ID: "slk", Text: "needle alpha slack channel", Meta: map[string]any{"source": "slack", "year": 2024}},
		{ID: "eml", Text: "needle beta email thread", Meta: map[string]any{"source": "email", "year": 2024}},
		{ID: "non", Text: "needle gamma no metadata"},
	}
	if err := s.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return s
}

// TestMetaFilterRestrictsToMatchingDocuments verifies that a document-level
// metadata predicate keeps only chunks whose document matches. This failed to
// compile before MetaFilter existed, which makes it a regression test for the
// new retrieval parameter itself.
func TestMetaFilterRestrictsToMatchingDocuments(t *testing.T) {
	s := metaTestStore(t)
	f, err := CompileMetaFilter([]string{"source=slack"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Retrieve(context.Background(), "needle", RetrieveParams{TopK: 10, MetaFilter: f})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("expected results for the matching document")
	}
	for _, r := range res {
		if r.Chunk.DocID != "slk" {
			t.Fatalf("got doc %q, want only the slack document", r.Chunk.DocID)
		}
	}
}

// TestMetaFilterAndSemantics verifies pairs compose with AND and that an
// unsatisfiable conjunction yields no results at all.
func TestMetaFilterAndSemantics(t *testing.T) {
	s := metaTestStore(t)
	f, err := CompileMetaFilter([]string{"source=slack", "year=2025"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Retrieve(context.Background(), "needle", RetrieveParams{TopK: 10, MetaFilter: f})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("expected no results with a conflicting pair, got %d", len(res))
	}
}

// TestMetaFilterNumeric compares numbers by JSON text and excludes documents
// without metadata.
func TestMetaFilterNumeric(t *testing.T) {
	s := metaTestStore(t)
	f, err := CompileMetaFilter([]string{"year=2024"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Retrieve(context.Background(), "needle", RetrieveParams{TopK: 10, MetaFilter: f})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("expected results for the metadata-bearing documents")
	}
	seen := map[string]bool{}
	for _, r := range res {
		seen[r.Chunk.DocID] = true
		if r.Chunk.DocID == "non" {
			t.Fatal("document without metadata must not match a meta filter")
		}
	}
	if !seen["slk"] || !seen["eml"] {
		t.Fatalf("expected both slack and email documents, got %v", seen)
	}
}

// TestCompileMetaFilterErrors covers the parser contract: malformed pairs are
// rejected and an empty list is a no-op (nil predicate).
func TestCompileMetaFilterErrors(t *testing.T) {
	if _, err := CompileMetaFilter([]string{"noseparator"}); err == nil {
		t.Fatal("want an error for a pair without '='")
	}
	if _, err := CompileMetaFilter([]string{"=v"}); err == nil {
		t.Fatal("want an error for an empty key")
	}
	if f, err := CompileMetaFilter(nil); err != nil || f != nil {
		t.Fatalf("empty input must yield a nil predicate (f=%v err=%v)", f != nil, err)
	}
}
