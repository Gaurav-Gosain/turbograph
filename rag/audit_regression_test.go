package rag

import (
	"bytes"
	"context"
	"fmt"
	"testing"
)

// These pin the defects the July 2026 Fable audit found. Each failed before its fix.

// A store loaded from disk, searched once, then mutated must not leave the vector and
// lexical indexes pointing at chunk ordinals that no longer exist. The two-tier lazy
// deferral let reindexLocked early-return on deferGraph even when a delete set
// needsRebuild, so the next search read past the end of the arrays and panicked.
func TestReindexAfterMutationOnLoadedStore(t *testing.T) {
	build := func() *Store {
		s := New(newKeywordEmbedder(64), Config{Seed: 1})
		docs := make([]Document, 8)
		for i := range docs {
			docs[i] = Document{ID: fmt.Sprintf("d%d", i), Text: fmt.Sprintf("document %d alpha beta gamma", i)}
		}
		if err := s.Build(context.Background(), docs); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		s.Save(&buf)
		l, err := Load(newKeywordEmbedder(64), &buf)
		if err != nil {
			t.Fatal(err)
		}
		l.Retrieve(context.Background(), "alpha beta", RetrieveParams{TopK: 3}) // build search index, leave graph deferred
		return l
	}
	t.Run("delete", func(t *testing.T) {
		l := build()
		l.DeleteDocument("d7")
		res, err := l.Retrieve(context.Background(), "alpha beta", RetrieveParams{TopK: 8})
		if err != nil {
			t.Fatalf("retrieve after delete: %v", err)
		}
		for _, r := range res {
			if r.Chunk.DocID == "d7" {
				t.Error("deleted document still retrievable")
			}
		}
	})
	t.Run("update", func(t *testing.T) {
		l := build()
		if err := l.AddDocuments(context.Background(), []Document{{ID: "d0", Text: "d0 replaced with wholly different words entirely"}}); err != nil {
			t.Fatal(err)
		}
		if _, err := l.Retrieve(context.Background(), "alpha beta", RetrieveParams{TopK: 8}); err != nil {
			t.Fatalf("retrieve after update: %v", err)
		}
	})
}

// The same store and query must return the same ranking every time, even when
// candidates tie on score. The fast path collected them by map iteration and sorted on
// score alone, so a tie at the top-k boundary was broken by Go's randomized map order.
func TestRetrieveIsDeterministicOnTies(t *testing.T) {
	s := New(newKeywordEmbedder(64), Config{Seed: 1})
	docs := make([]Document, 6)
	for i := range docs {
		docs[i] = Document{ID: fmt.Sprintf("d%d", i), Text: "identical tied text tokens"}
	}
	if err := s.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	first, _ := s.Retrieve(context.Background(), "identical tied text", RetrieveParams{TopK: 2})
	for run := 0; run < 40; run++ {
		got, _ := s.Retrieve(context.Background(), "identical tied text", RetrieveParams{TopK: 2})
		for i := range first {
			if got[i].Chunk.DocID != first[i].Chunk.DocID {
				t.Fatalf("run %d rank %d: %s vs first %s", run, i, got[i].Chunk.DocID, first[i].Chunk.DocID)
			}
		}
	}
}

// Re-saving a store that was loaded but not searched must not drop its persisted HNSW
// links; a `turbograph add` of an unchanged document does exactly this, and dropping the
// links forces the next open to reconstruct the whole graph.
func TestReSaveKeepsPersistedHNSW(t *testing.T) {
	s := New(newKeywordEmbedder(64), Config{Seed: 1})
	docs := make([]Document, 150)
	for i := range docs {
		docs[i] = Document{ID: fmt.Sprintf("d%03d", i), Text: fmt.Sprintf("document number %d indexed here", i)}
	}
	if err := s.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	var buf1 bytes.Buffer
	s.Save(&buf1)
	snap1, _ := readSnapshot(bytes.NewReader(buf1.Bytes()))
	if snap1.HNSW == nil {
		t.Fatal("setup: original save has no HNSW block")
	}
	loaded, _ := Load(newKeywordEmbedder(64), &buf1)
	var buf2 bytes.Buffer
	loaded.Save(&buf2) // no search
	snap2, _ := readSnapshot(bytes.NewReader(buf2.Bytes()))
	if snap2.HNSW == nil {
		t.Error("re-save of a loaded, unsearched store dropped the persisted HNSW links")
	}
	l2, _ := Load(newKeywordEmbedder(64), &buf2)
	if res, err := l2.Retrieve(context.Background(), "document number 99", RetrieveParams{TopK: 3}); err != nil || len(res) == 0 {
		t.Fatalf("twice-saved store does not retrieve: %v %v", res, err)
	}
}
