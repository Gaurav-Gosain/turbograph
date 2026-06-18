package rag

import (
	"context"
	"path/filepath"
	"testing"
)

func TestContentHashDedupCrossID(t *testing.T) {
	ctx := context.Background()
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
	body := "alpha beta gamma identical content shared across two uploads"
	if err := st.AddDocuments(ctx, []Document{{ID: "first", Text: body}}); err != nil {
		t.Fatal(err)
	}
	chunks := st.Len()
	// Same content under a different id must be treated as a duplicate.
	if err := st.AddDocuments(ctx, []Document{{ID: "second", Text: body}}); err != nil {
		t.Fatal(err)
	}
	if st.Len() != chunks {
		t.Errorf("content dedup failed: chunks grew %d -> %d", chunks, st.Len())
	}
	if owner, ok := st.ContentOwner(contentHash(body)); !ok || owner != "first" {
		t.Errorf("ContentOwner = %q,%v want first,true", owner, ok)
	}
}

func TestContentHashDedupWithinBatch(t *testing.T) {
	ctx := context.Background()
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
	docs := []Document{
		{ID: "a", Text: "duplicate body one two three"},
		{ID: "b", Text: "duplicate body one two three"}, // same content
		{ID: "c", Text: "distinct body four five six"},
	}
	if err := st.AddDocuments(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if st.DocCount() != 2 {
		t.Errorf("expected 2 distinct documents after within-batch dedup, got %d", st.DocCount())
	}
}

func TestContentHashDedupSurvivesReload(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "s.tg")
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
	body := "persisted content for dedup across reload alpha beta"
	st.AddDocuments(ctx, []Document{{ID: "x", Text: body}})
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	// After reload, the same content under a new id is still a duplicate.
	chunks := st2.Len()
	st2.AddDocuments(ctx, []Document{{ID: "y", Text: body}})
	if st2.Len() != chunks {
		t.Errorf("dedup did not survive reload: chunks grew %d -> %d", chunks, st2.Len())
	}
}

func TestIngestContentDedup(t *testing.T) {
	ctx := context.Background()
	st := newIngestStore()
	docs := []Document{
		{ID: "p", Text: "alpha repeated content for ingestion"},
		{ID: "q", Text: "alpha repeated content for ingestion"}, // dup content, different id
		{ID: "r", Text: "unique content for ingestion path"},
	}
	prog, err := st.Ingest(ctx, feed(docs), len(docs), IngestOptions{Workers: 4})
	if err != nil {
		t.Fatal(err)
	}
	if st.DocCount() != 2 {
		t.Errorf("ingest content dedup: want 2 docs, got %d (%+v)", st.DocCount(), prog)
	}
	if prog.Done+prog.Skipped != 3 {
		t.Errorf("all docs should be accounted for: %+v", prog)
	}
}
