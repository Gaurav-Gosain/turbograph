package rag

import (
	"context"
	"testing"
)

// TestSetStemRetokenizesLive covers the runtime stem parameter: SetStem must
// re-tokenize the corpus immediately, honor the language selection (so a
// language that is not named is left untouched), reject unknown language names
// without mutating the store, and be reversible. The language-specific half of
// the asserts would be impossible without per-language stemming.
func TestSetStemRetokenizesLive(t *testing.T) {
	ctx := context.Background()
	base := Config{
		Bits: 5, ResidualDims: 32, Seed: 1,
		GraphKNN: 4, MinSimilarity: 0.15, SequentialWeight: 0.0,
		Chunk: ChunkConfig{TargetWords: 200, OverlapWords: 0},
	}
	s := New(newKeywordEmbedder(96), base)
	if err := s.Build(ctx, []Document{{ID: "ru", Text: "машины и автомобили едут по дороге"}}); err != nil {
		t.Fatal(err)
	}
	// Stemming off: the inflected form does not match the base form.
	if res := s.bm25.Search("машина", 5); len(res) != 0 {
		t.Fatalf("stem off must not match inflections, got %+v", res)
	}

	// English-only selection must not start stemming Russian.
	if err := s.SetStem(true, []string{"en"}); err != nil {
		t.Fatal(err)
	}
	if res := s.bm25.Search("машина", 5); len(res) != 0 {
		t.Fatalf("en-only stemming must leave Russian untouched, got %+v", res)
	}

	// Selecting Russian re-tokenizes and the inflected form now matches.
	if err := s.SetStem(true, []string{"ru"}); err != nil {
		t.Fatal(err)
	}
	if res := s.bm25.Search("машина", 5); len(res) == 0 || res[0].ID != "ru#0" {
		t.Fatalf("ru stemming must retrieve the inflected form, got %+v", res)
	}

	// An unknown language is rejected and the store keeps its current setup.
	if err := s.SetStem(true, []string{"zz"}); err == nil {
		t.Fatal("expected an error for an unknown language")
	}
	if res := s.bm25.Search("машина", 5); len(res) == 0 {
		t.Fatal("failed SetStem must not disable the working stemmer")
	}

	// And switching it off again reverts to exact matching.
	if err := s.SetStem(false, nil); err != nil {
		t.Fatal(err)
	}
	if res := s.bm25.Search("машина", 5); len(res) != 0 {
		t.Fatalf("stem off after SetStem must match nothing, got %+v", res)
	}
}

// TestSetStemNoopWhenUnchanged ensures an idempotent call skips the (expensive)
// re-tokenization and stays valid.
func TestSetStemNoopWhenUnchanged(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(96), Config{Seed: 1})
	if err := s.Build(ctx, []Document{{ID: "a", Text: "running and runs"}}); err != nil {
		t.Fatal(err)
	}
	if err := s.SetStem(true, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.SetStem(true, nil); err != nil {
		t.Fatal(err)
	}
	if res := s.bm25.Search("run", 5); len(res) == 0 || res[0].ID != "a#0" {
		t.Fatalf("stemmed search after idempotent SetStem = %+v", res)
	}
}
