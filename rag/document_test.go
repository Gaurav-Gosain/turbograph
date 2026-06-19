package rag

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestLocateSpanWhitespaceInsensitive(t *testing.T) {
	text := []rune("The quick brown fox\njumps over   the lazy dog.")
	// A piece with collapsed whitespace still locates against the original.
	start, end := locateSpan(text, "brown fox jumps over the lazy", 0)
	if start < 0 {
		t.Fatal("expected a match")
	}
	got := string(text[start:end])
	if !strings.HasPrefix(got, "brown fox") || !strings.HasSuffix(got, "lazy") {
		t.Fatalf("span = %q", got)
	}
}

func TestLocateSpanForwardCursor(t *testing.T) {
	text := []rune("alpha beta gamma alpha beta delta")
	// Searching from 0 finds the first "alpha beta"; from past it finds the second.
	s1, _ := locateSpan(text, "alpha beta", 0)
	s2, _ := locateSpan(text, "alpha beta", s1+1)
	if s1 == s2 || s2 < s1 {
		t.Fatalf("forward search failed: %d then %d", s1, s2)
	}
}

func TestChunkOffsetsMapBackToDocument(t *testing.T) {
	st := versionStore(newKeywordEmbedder(96))
	text := "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
	if err := st.AddDocuments(context.Background(), []Document{{ID: "doc", Text: text}}); err != nil {
		t.Fatal(err)
	}
	view, ok := st.DocumentView("doc")
	if !ok {
		t.Fatal("no view")
	}
	if view.Text != text {
		t.Fatalf("view text mismatch")
	}
	if len(view.Spans) < 2 {
		t.Fatalf("expected multiple chunk spans, got %d", len(view.Spans))
	}
	runes := []rune(text)
	for _, sp := range view.Spans {
		if sp.Start < 0 || sp.End > len(runes) || sp.Start >= sp.End {
			t.Fatalf("invalid span %+v (text has %d runes)", sp, len(runes))
		}
		// The highlighted slice must be real document text.
		if strings.TrimSpace(string(runes[sp.Start:sp.End])) == "" {
			t.Fatalf("empty span %+v", sp)
		}
	}
}

func TestDocumentMetadataRoundTrips(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	meta := map[string]any{"author": "ada", "year": float64(1843), "tags": []any{"math", "notes"}}
	if err := st.AddDocuments(ctx, []Document{{ID: "doc", Text: "analytical engine notes", Meta: meta}}); err != nil {
		t.Fatal(err)
	}
	// Metadata comes back with retrieval.
	res, err := st.Retrieve(ctx, "engine", RetrieveParams{TopK: 3})
	if err != nil || len(res) == 0 {
		t.Fatalf("retrieve: %v (%d results)", err, len(res))
	}
	var got map[string]any
	if err := json.Unmarshal(res[0].Meta, &got); err != nil {
		t.Fatalf("meta unmarshal: %v (raw %s)", err, res[0].Meta)
	}
	if got["author"] != "ada" || got["year"].(float64) != 1843 {
		t.Fatalf("metadata = %v", got)
	}
}

func TestMetadataPreservedAcrossContentUpdate(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "first text", Meta: map[string]any{"k": "v"}}})
	// Update the content without supplying metadata; it must survive.
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "second different text"}})
	if m := st.DocMeta("doc"); !strings.Contains(string(m), `"k":"v"`) {
		t.Fatalf("metadata lost on content update: %s", m)
	}
	// SetDocMeta updates independently of content.
	st.SetDocMeta("doc", map[string]any{"k": "v2"})
	if m := st.DocMeta("doc"); !strings.Contains(string(m), `"v2"`) {
		t.Fatalf("SetDocMeta failed: %s", m)
	}
	st.SetDocMeta("doc", nil)
	if m := st.DocMeta("doc"); m != nil {
		t.Fatalf("clear failed: %s", m)
	}
}

func TestDeleteDocument(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{
		{ID: "keep", Text: "volcanoes erupt with lava and ash"},
		{ID: "drop", Text: "glaciers carve valleys over millennia"},
	})
	before := st.Len()
	removed := st.DeleteDocument("drop")
	if removed == 0 {
		t.Fatal("expected chunks removed")
	}
	if st.Len() != before-removed {
		t.Fatalf("len %d, want %d", st.Len(), before-removed)
	}
	if st.HasDoc("drop") {
		t.Fatal("document still present")
	}
	if _, ok := st.DocumentView("drop"); ok {
		t.Fatal("view should be gone")
	}
	// The remaining document still retrieves.
	res, _ := st.Retrieve(ctx, "lava ash", RetrieveParams{TopK: 2})
	if len(res) == 0 || res[0].Chunk.DocID != "keep" {
		t.Fatalf("kept document not retrievable: %+v", res)
	}
	// Deleting a missing document is a no-op.
	if st.DeleteDocument("nope") != 0 {
		t.Fatal("deleting a missing document should remove nothing")
	}
}

func TestDocMetaSurvivesReload(t *testing.T) {
	ctx := context.Background()
	st := versionStore(newKeywordEmbedder(96))
	st.AddDocuments(ctx, []Document{{ID: "doc", Text: "alpha beta", Meta: map[string]any{"src": "test"}}})
	path := t.TempDir() + "/s.tg"
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	if m := st2.DocMeta("doc"); !strings.Contains(string(m), `"src":"test"`) {
		t.Fatalf("metadata not persisted: %s", m)
	}
	if v, ok := st2.DocumentView("doc"); !ok || v.Text != "alpha beta" {
		t.Fatalf("view after reload: %+v ok=%v", v, ok)
	}
}

func BenchmarkLocateSpan(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 400; i++ {
		sb.WriteString("the quick brown fox jumps over the lazy dog ")
	}
	text := []rune(sb.String())
	needle := "quick brown fox jumps over the lazy dog the quick brown"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		locateSpan(text, needle, 0)
	}
}
