package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
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

func TestTruncateNormalize(t *testing.T) {
	v := []float32{3, 4, 100, 100} // first two have norm 5
	out := truncateNormalize(v, 2)
	if len(out) != 2 {
		t.Fatalf("want dim 2, got %d", len(out))
	}
	var n float64
	for _, x := range out {
		n += float64(x) * float64(x)
	}
	if d := n - 1; d > 1e-6 || d < -1e-6 {
		t.Fatalf("truncated vector not unit length: norm^2=%v", n)
	}
	// direction preserved: 3/5, 4/5
	if d := out[0] - 0.6; d > 1e-5 || d < -1e-5 {
		t.Fatalf("want 0.6, got %v", out[0])
	}
	// dim >= len is a no-op
	if got := truncateNormalize(v, 8); len(got) != 4 {
		t.Fatalf("oversized dim should be a no-op, got len %d", len(got))
	}
}

// TestEmbedDimTruncates checks the client truncates server embeddings to EmbedDim.
func TestEmbedDimTruncates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		out := embedResponse{Embeddings: [][]float32{{1, 1, 1, 1, 1, 1, 1, 1}}}
		_ = req
		json.NewEncoder(w).Encode(out)
	}))
	defer srv.Close()
	c := New()
	c.BaseURL = srv.URL
	c.DocPrefix = ""
	c.EmbedDim = 4
	v, err := c.Embed(context.Background(), []string{"x"})
	if err != nil {
		t.Fatal(err)
	}
	if len(v[0]) != 4 {
		t.Fatalf("want truncated dim 4, got %d", len(v[0]))
	}
}

// TestEmbedSplitsLargeRequests verifies that Embed never sends more than
// MaxEmbedBatch inputs in one request and returns the vectors in input order
// across the requests.
func TestEmbedSplitsLargeRequests(t *testing.T) {
	var sizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		json.NewDecoder(r.Body).Decode(&req)
		sizes = append(sizes, len(req.Input))
		out := embedResponse{Embeddings: make([][]float32, len(req.Input))}
		for i, in := range req.Input {
			n, _ := strconv.Atoi(in)
			out.Embeddings[i] = []float32{float32(n), 1}
		}
		json.NewEncoder(w).Encode(out)
	}))
	defer srv.Close()

	c := New()
	c.BaseURL = srv.URL
	c.DocPrefix = ""
	texts := make([]string, 2*MaxEmbedBatch+22)
	for i := range texts {
		texts[i] = strconv.Itoa(i)
	}
	vecs, err := c.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(sizes) != 3 {
		t.Fatalf("want 3 requests for %d inputs, got %v", len(texts), sizes)
	}
	for _, n := range sizes {
		if n > MaxEmbedBatch {
			t.Fatalf("request carried %d inputs, cap is %d", n, MaxEmbedBatch)
		}
	}
	if len(vecs) != len(texts) {
		t.Fatalf("want %d vectors, got %d", len(texts), len(vecs))
	}
	for i, v := range vecs {
		if v[0] != float32(i) {
			t.Fatalf("vector %d out of order: first coord %v", i, v[0])
		}
	}
}
