package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmbedPromptsKnownModels(t *testing.T) {
	cases := []struct {
		model, wantQ, wantD string
	}{
		{"embeddinggemma", "task: search result | query: ", "title: none | text: "},
		{"embeddinggemma:latest", "task: search result | query: ", "title: none | text: "},
		{"nomic-embed-text", "search_query: ", "search_document: "},
		{"multilingual-e5-large", "query: ", "passage: "},
		{"bge-large", "Represent this sentence for searching relevant passages: ", ""},
		{"some-unknown-model", "", ""},
	}
	for _, c := range cases {
		q, d := EmbedPrompts(c.model)
		if q != c.wantQ || d != c.wantD {
			t.Errorf("%s: got (%q,%q), want (%q,%q)", c.model, q, d, c.wantQ, c.wantD)
		}
	}
}

func TestSetEmbedModelAppliesPrompts(t *testing.T) {
	c := New() // defaults to embeddinggemma
	if c.QueryPrefix == "" || c.DocPrefix == "" {
		t.Fatal("default client should carry embeddinggemma prompts")
	}
	c.SetEmbedModel("plain-model")
	if c.QueryPrefix != "" || c.DocPrefix != "" {
		t.Fatalf("unknown model should clear prompts, got (%q,%q)", c.QueryPrefix, c.DocPrefix)
	}
}

// TestEmbedAppliesPrefix verifies that Embed and EmbedQuery prepend the document
// and query prompts respectively to what is actually sent to the server.
func TestEmbedAppliesPrefix(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		got = req.Input
		// echo one trivial vector per input
		out := embedResponse{Embeddings: make([][]float32, len(req.Input))}
		for i := range out.Embeddings {
			out.Embeddings[i] = []float32{1, 0}
		}
		json.NewEncoder(w).Encode(out)
	}))
	defer srv.Close()

	c := New()
	c.BaseURL = srv.URL
	c.QueryPrefix = "Q: "
	c.DocPrefix = "D: "

	ctx := context.Background()
	if _, err := c.Embed(ctx, []string{"hello"}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !strings.HasPrefix(got[0], "D: ") {
		t.Fatalf("Embed should apply DocPrefix, sent %v", got)
	}
	if _, err := c.EmbedQuery(ctx, []string{"hello"}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !strings.HasPrefix(got[0], "Q: ") {
		t.Fatalf("EmbedQuery should apply QueryPrefix, sent %v", got)
	}
}
