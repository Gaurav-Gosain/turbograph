package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// fakeOllamaServer is a minimal Ollama API stand-in for exercising the server's
// model-backed handlers (chat streaming, model listing, pull) without a real
// model server.
func fakeOllamaServer() *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"models": []map[string]string{{"name": "qwen3.5:2b"}}})
	})
	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		for _, t := range []string{"answer", " from [1]"} {
			json.NewEncoder(w).Encode(map[string]any{"response": t, "done": false})
		}
		json.NewEncoder(w).Encode(map[string]any{"response": "", "done": true})
	})
	mux.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"status": "pulling", "total": 10, "completed": 5})
		json.NewEncoder(w).Encode(map[string]any{"status": "success", "total": 10, "completed": 10})
	})
	return httptest.NewServer(mux)
}

func newGenServer(t *testing.T) (*httptest.Server, *httptest.Server) {
	t.Helper()
	oll := fakeOllamaServer()
	client := ollama.New()
	client.BaseURL = oll.URL
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []rag.Document{
		{ID: "a", Text: "graphs connect nodes with edges"},
		{ID: "b", Text: "vectors are embedded and quantized"},
	}
	if err := store.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	s := New(store)
	s.SetGenerator(client, "qwen3.5:2b", "embeddinggemma")
	return httptest.NewServer(s.Handler()), oll
}

func TestModelsEndpoint(t *testing.T) {
	ts, oll := newGenServer(t)
	defer ts.Close()
	defer oll.Close()
	resp, err := http.Get(ts.URL + "/api/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Models  []string `json:"models"`
		Default string   `json:"default"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Models) != 1 || out.Default != "qwen3.5:2b" {
		t.Fatalf("unexpected models response: %+v", out)
	}
}

func TestChatStreaming(t *testing.T) {
	ts, oll := newGenServer(t)
	defer ts.Close()
	defer oll.Close()
	body := `{"query":"graphs","top_k":2,"model":"qwen3.5:2b"}`
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	out := buf.String()
	for _, want := range []string{"event: sources", "event: token", "event: done"} {
		if !strings.Contains(out, want) {
			t.Fatalf("chat stream missing %q in:\n%s", want, out)
		}
	}
}

func TestChatCompletionsNonStream(t *testing.T) {
	ts, oll := newGenServer(t)
	defer ts.Close()
	defer oll.Close()
	body := `{"messages":[{"role":"user","content":"graphs"}],"top_k":2}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Choices) == 0 || out.Choices[0].Message.Content == "" {
		t.Fatalf("empty completion: %+v", out)
	}
}

func TestPullEndpoint(t *testing.T) {
	ts, oll := newGenServer(t)
	defer ts.Close()
	defer oll.Close()
	resp, err := http.Post(ts.URL+"/api/pull?model=qwen3.5:2b", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	if !strings.Contains(buf.String(), "event: done") {
		t.Fatalf("pull stream missing done event:\n%s", buf.String())
	}
}
