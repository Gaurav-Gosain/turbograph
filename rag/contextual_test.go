package rag

import (
	"context"
	"strings"
	"testing"
)

// keyedContextualizer situates a chunk by a keyword found in its body, so the
// generated context differs per document, deterministically, without a model.
type keyedContextualizer struct {
	match  string // substring identifying the target chunk's body (in the prompt)
	prefix string // context to attach when match is present
}

func (f keyedContextualizer) Generate(_ context.Context, _, prompt string) (string, error) {
	if strings.Contains(prompt, f.match) {
		return f.prefix, nil
	}
	return "This passage concerns an unrelated topic.", nil
}

func TestContextualRetrieval(t *testing.T) {
	ctx := context.Background()
	// The body never mentions "Borealis"; the context does. Without contextual
	// retrieval a "Borealis" query cannot match this chunk lexically at all.
	s := New(newKeywordEmbedder(128), Config{Seed: 1, MinSimilarity: 0.01})
	s.SetContextualizer(keyedContextualizer{match: "cycle life", prefix: "This passage is about Project Borealis and its battery chemistry."})
	docs := []Document{
		{ID: "d1", Text: "The cells offer high cycle life at low cost and avoid rare-earth metals."},
		{ID: "d2", Text: "Unrelated text about weather patterns and ocean currents over the pacific."},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}

	// The stored body must be untouched (the prefix is index-only).
	var d1 *Chunk
	for i := range s.chunks {
		if s.chunks[i].DocID == "d1" {
			d1 = &s.chunks[i]
			break
		}
	}
	if d1 == nil {
		t.Fatal("d1 chunk missing")
	}
	if strings.Contains(d1.Text, "Borealis") {
		t.Fatalf("context leaked into the body: %q", d1.Text)
	}
	if d1.Context == "" || !strings.Contains(d1.IndexText(), "Borealis") {
		t.Fatalf("IndexText should carry the context prefix, got %q", d1.IndexText())
	}

	// A query that only matches the context must now retrieve the chunk.
	res, err := s.Retrieve(ctx, "Borealis", RetrieveParams{TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].Chunk.DocID != "d1" {
		t.Fatalf("contextual retrieval failed to surface d1: %+v", res)
	}
	// And what is handed back to the caller is the clean body, not the prefix.
	if strings.Contains(res[0].Chunk.Text, "Borealis") {
		t.Fatalf("retrieved chunk body should be prefix-free: %q", res[0].Chunk.Text)
	}
}

func TestContextualDisabledIsUnchanged(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(64), Config{Seed: 1, MinSimilarity: 0.01})
	if err := s.Build(ctx, []Document{{ID: "d", Text: "plain body text"}}); err != nil {
		t.Fatal(err)
	}
	for i := range s.chunks {
		if s.chunks[i].Context != "" {
			t.Fatalf("no contextualizer set, Context must be empty, got %q", s.chunks[i].Context)
		}
		if s.chunks[i].IndexText() != s.chunks[i].Text {
			t.Fatal("IndexText must equal Text on the default path")
		}
	}
}
