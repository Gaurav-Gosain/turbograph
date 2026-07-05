package server

import (
	"context"
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// TestExpandPassages verifies the server glue for small-to-big retrieval:
// expandPassages widens each passage's text to a neighbour window while leaving
// the originals (the cited sources) untouched.
func TestExpandPassages(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, MinSimilarity: 0.01,
		Chunk: rag.ChunkConfig{Strategy: rag.StrategySentence, TargetWords: 4, OverlapWords: 0}})
	err := store.Build(context.Background(), []rag.Document{
		{ID: "d", Text: "Alpha one two. Beta three four. Gamma five six. Delta seven eight."},
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := store.Retrieve(context.Background(), "Gamma five six", rag.RetrieveParams{TopK: 1})
	if err != nil || len(res) == 0 {
		t.Fatalf("retrieve: %v", err)
	}
	orig := res[0].Chunk.Text

	exp := expandPassages(store, res, 1)
	if len(exp) != len(res) {
		t.Fatalf("length changed: %d", len(exp))
	}
	if len(exp[0].Chunk.Text) <= len(orig) {
		t.Fatalf("window did not expand: %q -> %q", orig, exp[0].Chunk.Text)
	}
	if !strings.Contains(exp[0].Chunk.Text, "Gamma") {
		t.Fatalf("expanded text lost the original chunk: %q", exp[0].Chunk.Text)
	}
	if res[0].Chunk.Text != orig {
		t.Fatal("original result was mutated; the cited source must stay the small chunk")
	}

	// window 0 is a passthrough (same texts as the input).
	if got := expandPassages(store, res, 0); got[0].Chunk.Text != orig {
		t.Fatalf("window 0 should be a no-op, got %q", got[0].Chunk.Text)
	}
}
