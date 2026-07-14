package rag

import (
	"bytes"
	"context"
	"encoding/gob"
	"testing"
)

// TestLegacyEmbedsStillLoad: a .tg written before the flat vector layout must still
// open and still search. The layout changed for speed; a store someone already has is
// not a thing they should have to rebuild.
func TestLegacyEmbedsStillLoad(t *testing.T) {
	s := New(newKeywordEmbedder(64), Config{Seed: 1})
	if err := s.Build(context.Background(), []Document{
		{ID: "a", Text: "the caldera reactor was built in northgate"},
		{ID: "b", Text: "project helios is led by mira tan"},
	}); err != nil {
		t.Fatal(err)
	}
	// Hand-write a snapshot in the OLD per-chunk layout.
	snap := snapshot{Cfg: s.cfg, Dim: s.dim, Chunks: s.chunks, Embeds: s.embeds,
		Hashes: s.idHash, Versions: s.versions, DocMeta: s.docMeta}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(&snap); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(newKeywordEmbedder(64), &buf)
	if err != nil {
		t.Fatalf("a legacy snapshot failed to load: %v", err)
	}
	if loaded.Len() != s.Len() {
		t.Fatalf("legacy load lost chunks: %d want %d", loaded.Len(), s.Len())
	}
	res, err := loaded.Retrieve(context.Background(), "caldera reactor northgate", RetrieveParams{TopK: 1})
	if err != nil || len(res) == 0 {
		t.Fatalf("legacy store does not retrieve: %v %v", res, err)
	}
	if res[0].Chunk.DocID != "a" {
		t.Errorf("legacy store retrieved %q, want a", res[0].Chunk.DocID)
	}
}
