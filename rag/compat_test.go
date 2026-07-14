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

// TestFlatF32SnapshotStillLoads: the interim layout, a gob []float32 block, shipped in
// one commit before the raw-byte block replaced it. Anyone who built a store from it
// must still be able to open it.
func TestFlatF32SnapshotStillLoads(t *testing.T) {
	s := New(newKeywordEmbedder(64), Config{Seed: 1})
	if err := s.Build(context.Background(), []Document{
		{ID: "a", Text: "the caldera reactor was built in northgate"},
		{ID: "b", Text: "project helios is led by mira tan"},
	}); err != nil {
		t.Fatal(err)
	}
	flat := make([]float32, 0, len(s.embeds)*s.dim)
	for _, v := range s.embeds {
		flat = append(flat, v...)
	}
	snap := snapshot{Cfg: s.cfg, Dim: s.dim, Chunks: s.chunks, FlatF32: flat,
		Hashes: s.idHash, Versions: s.versions, DocMeta: s.docMeta}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(&snap); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(newKeywordEmbedder(64), &buf)
	if err != nil {
		t.Fatalf("a FlatF32 snapshot failed to load: %v", err)
	}
	res, err := loaded.Retrieve(context.Background(), "caldera reactor northgate", RetrieveParams{TopK: 1})
	if err != nil || len(res) == 0 || res[0].Chunk.DocID != "a" {
		t.Fatalf("FlatF32 store does not retrieve correctly: %+v %v", res, err)
	}
}

// TestVectorsSurviveTheRawBlock: the raw little-endian block must round-trip the exact
// float bits. A silently lossy vector format would degrade retrieval and never say so.
func TestVectorsSurviveTheRawBlock(t *testing.T) {
	s := New(newKeywordEmbedder(64), Config{Seed: 1})
	if err := s.Build(context.Background(), []Document{
		{ID: "a", Text: "the caldera reactor was built in northgate"},
		{ID: "b", Text: "project helios is led by mira tan at verdant labs"},
	}); err != nil {
		t.Fatal(err)
	}
	want := make([][]float32, len(s.embeds))
	for i, v := range s.embeds {
		want[i] = append([]float32(nil), v...)
	}
	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(newKeywordEmbedder(64), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.embeds) != len(want) {
		t.Fatalf("got %d vectors, want %d", len(loaded.embeds), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if loaded.embeds[i][j] != want[i][j] {
				t.Fatalf("vector %d coord %d changed: %v -> %v", i, j, want[i][j], loaded.embeds[i][j])
			}
		}
	}
}
