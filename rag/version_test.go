package rag

import (
	"context"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// countingEmbedder counts how many texts it embeds, to prove an update only
// re-embeds the chunks that actually changed.
type countingEmbedder struct {
	inner    *keywordEmbedder
	embedded int64
}

func (e *countingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	atomic.AddInt64(&e.embedded, int64(len(texts)))
	return e.inner.Embed(ctx, texts)
}

func versionStore(emb Embedder) *Store {
	return New(emb, Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 6, OverlapWords: 0}})
}

func TestUpdateReembedsOnlyChangedChunks(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: newKeywordEmbedder(96)}
	st := versionStore(emb)
	// Three short paragraphs become three chunks (TargetWords 6).
	v1 := "alpha alpha alpha alpha alpha one beta beta beta beta beta two gamma gamma gamma gamma gamma three"
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: v1}}); err != nil {
		t.Fatal(err)
	}
	first := atomic.LoadInt64(&emb.embedded)
	if first == 0 {
		t.Fatal("nothing embedded on first add")
	}
	chunksV1 := st.Len()

	// Change only the middle chunk; the first and third are byte-identical.
	v2 := "alpha alpha alpha alpha alpha one CHANGED beta words here entirely two gamma gamma gamma gamma gamma three"
	atomicReset := atomic.LoadInt64(&emb.embedded)
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: v2}}); err != nil {
		t.Fatal(err)
	}
	delta := atomic.LoadInt64(&emb.embedded) - atomicReset
	if delta == 0 {
		t.Error("update embedded nothing; the change was not applied")
	}
	if int(delta) >= chunksV1 {
		t.Errorf("update re-embedded %d chunks; expected fewer than the %d total (reuse failed)", delta, chunksV1)
	}
	// The store still holds exactly one document.
	if st.DocCount() != 1 {
		t.Errorf("doc count = %d, want 1", st.DocCount())
	}
}

func TestUpdateReplacesContent(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "d", Text: "the original zebra content about deserts and dunes"}})
	st.AddDocuments(ctx, []Document{{ID: "d", Text: "the replacement penguin content about ice and snow"}})

	if st.DocCount() != 1 {
		t.Fatalf("doc count = %d, want 1", st.DocCount())
	}
	// Old content must be gone; new content must be retrievable.
	res, _ := st.Retrieve(ctx, "penguin ice snow", RetrieveParams{TopK: 3})
	newFound := false
	for _, r := range res {
		if strings.Contains(r.Chunk.Text, "zebra") || strings.Contains(r.Chunk.Text, "desert") {
			t.Error("stale content from the old version is still indexed")
		}
		if strings.Contains(r.Chunk.Text, "penguin") || strings.Contains(r.Chunk.Text, "snow") {
			newFound = true
		}
	}
	if !newFound {
		t.Errorf("updated content not retrievable: %+v", res)
	}
}

func TestUpdateShrinksDocument(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	long := "alpha alpha alpha one beta beta beta two gamma gamma gamma three delta delta delta four"
	st.AddDocuments(ctx, []Document{{ID: "d", Text: long}})
	big := st.Len()
	st.AddDocuments(ctx, []Document{{ID: "d", Text: "alpha alpha alpha one"}})
	if st.Len() >= big {
		t.Errorf("shrinking a document did not remove chunks: %d -> %d", big, st.Len())
	}
	if st.DocCount() != 1 {
		t.Errorf("doc count = %d, want 1", st.DocCount())
	}
}

func TestUpdateSurvivesReload(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "s.tg")
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "d", Text: "first version about volcanoes and lava"}})
	st.AddDocuments(ctx, []Document{{ID: "d", Text: "second version about glaciers and frost"}})
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	if st2.DocCount() != 1 {
		t.Fatalf("reloaded doc count = %d, want 1", st2.DocCount())
	}
	res, _ := st2.Retrieve(ctx, "glaciers frost", RetrieveParams{TopK: 2})
	if len(res) == 0 || strings.Contains(res[0].Chunk.Text, "volcanoes") {
		t.Errorf("reloaded store did not preserve the update: %+v", res)
	}
}

func TestUpdateViaIngestEngine(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "d", Text: "engine first content about rivers"}})
	// Re-ingest the same id with new content through the streaming engine.
	prog, err := st.Ingest(ctx, feed([]Document{{ID: "d", Text: "engine updated content about mountains"}}), 1, IngestOptions{Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	if prog.Done != 1 {
		t.Errorf("expected the update to count as done, got %+v", prog)
	}
	if st.DocCount() != 1 {
		t.Errorf("doc count = %d, want 1", st.DocCount())
	}
	res, _ := st.Retrieve(ctx, "mountains", RetrieveParams{TopK: 2})
	if len(res) == 0 || strings.Contains(res[0].Chunk.Text, "rivers") {
		t.Errorf("engine update did not replace content: %+v", res)
	}
}

func TestVersionHistoryTracksUpdates(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta gamma"}}); err != nil {
		t.Fatal(err)
	}
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta gamma delta"}}); err != nil {
		t.Fatal(err)
	}
	// Re-adding identical content must not append a version.
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta gamma delta"}}); err != nil {
		t.Fatal(err)
	}
	vs := st.DocVersions("doc")
	if len(vs) != 2 {
		t.Fatalf("got %d versions, want 2", len(vs))
	}
	if vs[0].N != 1 || vs[1].N != 2 {
		t.Errorf("version numbers = %d,%d; want 1,2", vs[0].N, vs[1].N)
	}
	if vs[0].Current || !vs[1].Current {
		t.Errorf("current flags wrong: %v,%v", vs[0].Current, vs[1].Current)
	}
	if txt, ok := st.DocVersionText("doc", 1); !ok || txt != "alpha beta gamma" {
		t.Errorf("version 1 text = %q (ok=%v)", txt, ok)
	}
}

func TestRestoreReingestsOldVersion(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta gamma"}})
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "totally different words here"}})
	// Restore is a re-ingest of version 1's text through the normal path.
	old, _ := st.DocVersionText("doc", 1)
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: old}}); err != nil {
		t.Fatal(err)
	}
	vs := st.DocVersions("doc")
	if len(vs) != 3 {
		t.Fatalf("got %d versions after restore, want 3", len(vs))
	}
	if vs[2].Hash != vs[0].Hash {
		t.Errorf("restored version hash %q != original %q", vs[2].Hash, vs[0].Hash)
	}
}

func TestVersionsSurviveReload(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta gamma"}})
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta gamma delta"}})
	path := filepath.Join(t.TempDir(), "snap.tg")
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	if vs := st2.DocVersions("doc"); len(vs) != 2 {
		t.Fatalf("after reload got %d versions, want 2", len(vs))
	}
}
