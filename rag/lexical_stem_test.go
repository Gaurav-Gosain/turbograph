package rag

import (
	"context"
	"testing"
)

func TestLexicalStemConfiguredFromStore(t *testing.T) {
	ctx := context.Background()
	docs := []Document{{ID: "ru", Text: "машины и автомобили едут по дороге"}}

	base := Config{
		Bits: 5, ResidualDims: 32, Seed: 1,
		GraphKNN: 4, MinSimilarity: 0.15, SequentialWeight: 0.0,
		Chunk: ChunkConfig{TargetWords: 200, OverlapWords: 0},
	}

	stemmed := New(newKeywordEmbedder(96), func() Config { c := base; c.LexicalStem = true; return c }())
	if err := stemmed.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	res := stemmed.bm25.Search("машина", 5)
	if len(res) == 0 || res[0].ID != "ru#0" {
		t.Fatalf("stemmed store must retrieve the inflected form, got %+v", res)
	}

	plain := New(newKeywordEmbedder(96), base)
	if err := plain.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if res := plain.bm25.Search("машина", 5); len(res) != 0 {
		t.Fatalf("default store must not stem, got %+v", res)
	}
}
