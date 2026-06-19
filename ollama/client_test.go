package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeOllama is a minimal stand-in for the Ollama HTTP API, enough to exercise
// the client without a running model server.
func fakeOllama(t *testing.T) (*Client, *httptest.Server) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]string{{"name": "embeddinggemma:latest"}, {"name": "qwen3.5:2b"}},
		})
	})
	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["stream"] == true {
			for _, tok := range []string{"hello", " ", "world"} {
				json.NewEncoder(w).Encode(map[string]any{"response": tok, "done": false})
			}
			json.NewEncoder(w).Encode(map[string]any{"response": "", "done": true})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"response": "hello world", "done": true})
	})
	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(PullProgress{Status: "pulling", Total: 100, Completed: 50})
		json.NewEncoder(w).Encode(PullProgress{Status: "success", Total: 100, Completed: 100})
	})
	srv := httptest.NewServer(mux)
	c := New()
	c.BaseURL = srv.URL
	return c, srv
}

func TestListModels(t *testing.T) {
	c, srv := fakeOllama(t)
	defer srv.Close()
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0] != "embeddinggemma:latest" {
		t.Fatalf("unexpected models: %v", models)
	}
}

func TestGenerate(t *testing.T) {
	c, srv := fakeOllama(t)
	defer srv.Close()
	out, err := c.Generate(context.Background(), "m", "sys", "prompt")
	if err != nil {
		t.Fatal(err)
	}
	if out != "hello world" {
		t.Fatalf("got %q", out)
	}
}

func TestGenerateStream(t *testing.T) {
	c, srv := fakeOllama(t)
	defer srv.Close()
	var got strings.Builder
	err := c.GenerateStream(context.Background(), "m", "sys", "prompt", func(tok string) error {
		got.WriteString(tok)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "hello world" {
		t.Fatalf("streamed %q", got.String())
	}
}

func TestGenerateStreamStopsOnError(t *testing.T) {
	c, srv := fakeOllama(t)
	defer srv.Close()
	n := 0
	stop := fmt.Errorf("stop")
	err := c.GenerateStream(context.Background(), "m", "s", "p", func(string) error {
		n++
		return stop
	})
	if err != stop {
		t.Fatalf("expected the callback error to propagate, got %v", err)
	}
	if n != 1 {
		t.Fatalf("stream should stop after the first token, got %d", n)
	}
}

func TestPull(t *testing.T) {
	c, srv := fakeOllama(t)
	defer srv.Close()
	var last PullProgress
	err := c.Pull(context.Background(), "m", func(p PullProgress) error {
		last = p
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if last.Completed != 100 {
		t.Fatalf("expected final progress, got %+v", last)
	}
}

func TestPing(t *testing.T) {
	c, srv := fakeOllama(t)
	defer srv.Close()
	if err := c.Ping(context.Background()); err != nil {
		t.Fatalf("ping should succeed against a live tags endpoint: %v", err)
	}
}

func TestGenerateServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	c := New()
	c.BaseURL = srv.URL
	if _, err := c.Generate(context.Background(), "m", "s", "p"); err == nil {
		t.Fatal("expected an error on 500")
	}
}
