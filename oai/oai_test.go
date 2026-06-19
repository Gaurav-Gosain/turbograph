package oai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeOAI is a minimal OpenAI-compatible server for exercising the client.
func fakeOAI(t *testing.T) (*Client, *httptest.Server) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "gpt-4o-mini"}, {"id": "text-embedding-3-small"}}})
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		// Return embeddings deliberately OUT of order to test index handling.
		data := make([]map[string]any, len(req.Input))
		for i := range req.Input {
			idx := len(req.Input) - 1 - i
			data[i] = map[string]any{"embedding": []float32{float32(idx), 1, 0}, "index": idx}
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["stream"] == true {
			for _, tok := range []string{"hi", " there"} {
				b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]string{"content": tok}}}})
				w.Write([]byte("data: "))
				w.Write(b)
				w.Write([]byte("\n\n"))
			}
			w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": "hi there"}}}})
	})
	srv := httptest.NewServer(mux)
	return New(srv.URL, "test-key", "text-embedding-3-small"), srv
}

func TestOAIEmbedOrdersByIndex(t *testing.T) {
	c, srv := fakeOAI(t)
	defer srv.Close()
	vecs, err := c.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 3 {
		t.Fatalf("want 3 vectors, got %d", len(vecs))
	}
	// Each vector's first coordinate equals its index (the fake encodes that), so
	// correct index handling means vecs[i][0] == i despite out-of-order delivery.
	for i := range vecs {
		if vecs[i][0] != float32(i) {
			t.Errorf("vector %d misplaced: first coord %v", i, vecs[i][0])
		}
	}
}

func TestOAIGenerate(t *testing.T) {
	c, srv := fakeOAI(t)
	defer srv.Close()
	out, err := c.Generate(context.Background(), "gpt-4o-mini", "sys", "prompt")
	if err != nil {
		t.Fatal(err)
	}
	if out != "hi there" {
		t.Fatalf("got %q", out)
	}
}

func TestOAIGenerateStream(t *testing.T) {
	c, srv := fakeOAI(t)
	defer srv.Close()
	var got strings.Builder
	err := c.GenerateStream(context.Background(), "gpt-4o-mini", "sys", "prompt", func(tok string) error {
		got.WriteString(tok)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "hi there" {
		t.Fatalf("streamed %q", got.String())
	}
}

func TestOAIListModelsAndAuth(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "m1"}}})
	}))
	defer srv.Close()
	c := New(srv.URL, "secret", "m1")
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0] != "m1" {
		t.Fatalf("unexpected models: %v", models)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("API key not sent as bearer, got %q", gotAuth)
	}
}

func TestOAIEmbedDimTruncates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"embedding": []float32{1, 1, 1, 1, 1, 1, 1, 1}, "index": 0}}})
	}))
	defer srv.Close()
	c := New(srv.URL, "", "m")
	c.EmbedDim = 4
	v, err := c.Embed(context.Background(), []string{"x"})
	if err != nil {
		t.Fatal(err)
	}
	if len(v[0]) != 4 {
		t.Fatalf("want truncated dim 4, got %d", len(v[0]))
	}
}
