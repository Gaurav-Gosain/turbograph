package rag

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestManagerCreateGetList(t *testing.T) {
	m, err := NewManager("", newKeywordEmbedder(64), Config{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create("alpha"); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create("alpha"); err == nil {
		t.Error("creating a duplicate bucket should error")
	}
	if _, ok := m.Get("alpha"); !ok {
		t.Error("alpha should exist")
	}
	if _, ok := m.Get("missing"); ok {
		t.Error("missing should not exist")
	}
	m.GetOrCreate("beta")
	got := m.List()
	if len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
		t.Errorf("List = %v, want [alpha beta] sorted", got)
	}
}

func TestManagerBucketNameValidation(t *testing.T) {
	m, _ := NewManager("", newKeywordEmbedder(64), Config{})
	for _, bad := range []string{"", "../escape", "has/slash", "has space", "x" + string(make([]byte, 100))} {
		if _, err := m.Create(bad); err == nil {
			t.Errorf("expected %q to be rejected", bad)
		}
	}
	for _, ok := range []string{"default", "my-corpus", "v1.2_data", "A"} {
		if !ValidBucketName(ok) {
			t.Errorf("expected %q to be valid", ok)
		}
	}
}

func TestManagerBucketsAreIsolated(t *testing.T) {
	ctx := context.Background()
	m, _ := NewManager("", newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
	a, _ := m.Create("a")
	b, _ := m.Create("b")
	a.AddDocuments(ctx, []Document{{ID: "x", Text: "alpha content about cats"}})
	b.AddDocuments(ctx, []Document{{ID: "y", Text: "omega content about ships"}})

	if a.DocCount() != 1 || b.DocCount() != 1 {
		t.Fatalf("each bucket should hold one document: a=%d b=%d", a.DocCount(), b.DocCount())
	}
	// A document in one bucket must not be visible in the other.
	if a.HasDoc("y") || b.HasDoc("x") {
		t.Error("buckets are not isolated")
	}
}

func TestManagerPersistAndReload(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	m, err := NewManager(dir, newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
	if err != nil {
		t.Fatal(err)
	}
	st, _ := m.Create("research")
	st.AddDocuments(ctx, []Document{
		{ID: "d1", Text: "graphs connect nodes with edges"},
		{ID: "d2", Text: "vectors are quantized for compact storage"},
	})
	if err := m.Save("research"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "research.tg")); err != nil {
		t.Fatalf("bucket file not written: %v", err)
	}

	// A fresh manager over the same directory must load the bucket.
	m2, err := NewManager(dir, newKeywordEmbedder(96), Config{})
	if err != nil {
		t.Fatal(err)
	}
	st2, ok := m2.Get("research")
	if !ok {
		t.Fatal("reloaded manager missing the research bucket")
	}
	if st2.DocCount() != 2 {
		t.Errorf("reloaded bucket has %d docs, want 2", st2.DocCount())
	}

	// Delete removes the bucket and its file.
	if err := m2.Delete("research"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "research.tg")); err == nil {
		t.Error("bucket file should be removed after delete")
	}
}
