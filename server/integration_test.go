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

func TestDocumentsEndpoint(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []rag.Document{{ID: "readme.md", Text: "alpha beta gamma delta epsilon"}, {ID: "guide.md", Text: "one two three"}}
	if err := store.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(New(store).Handler())
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/api/documents")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Documents []struct {
			ID     string `json:"id"`
			Chunks int    `json:"chunks"`
			Bytes  int    `json:"bytes"`
		} `json:"documents"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Documents) != 2 {
		t.Fatalf("expected 2 documents, got %d", len(out.Documents))
	}
	if out.Documents[0].ID != "readme.md" || out.Documents[0].Chunks < 1 || out.Documents[0].Bytes == 0 {
		t.Fatalf("unexpected first document: %+v", out.Documents[0])
	}
}

func TestVersionEndpoints(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	ctx := context.Background()
	if err := store.Build(ctx, []rag.Document{{ID: "doc", Text: "alpha beta gamma"}}); err != nil {
		t.Fatal(err)
	}
	if err := store.AddDocuments(ctx, []rag.Document{{ID: "doc", Text: "alpha beta gamma delta"}}); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(New(store).Handler())
	defer ts.Close()

	// list
	var list struct {
		Versions []struct {
			N       int  `json:"n"`
			Current bool `json:"current"`
		} `json:"versions"`
	}
	getJSON(t, ts.URL+"/api/versions?doc=doc", &list)
	if len(list.Versions) != 2 || !list.Versions[1].Current {
		t.Fatalf("versions = %+v", list.Versions)
	}

	// text of version 1
	var v1 struct {
		Text string `json:"text"`
	}
	getJSON(t, ts.URL+"/api/version?doc=doc&n=1", &v1)
	if v1.Text != "alpha beta gamma" {
		t.Fatalf("v1 text = %q", v1.Text)
	}

	// restore version 1 -> appends a third version equal to v1
	resp, err := http.Post(ts.URL+"/api/restore?doc=doc&n=1", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("restore status %d", resp.StatusCode)
	}
	getJSON(t, ts.URL+"/api/versions?doc=doc", &list)
	if len(list.Versions) != 3 {
		t.Fatalf("after restore got %d versions, want 3", len(list.Versions))
	}
}

func getJSON(t *testing.T, url string, dst any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s -> %d", url, resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentViewAndDelete(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	ctx := context.Background()
	if err := store.Build(ctx, []rag.Document{
		{ID: "a.md", Text: "alpha beta gamma delta epsilon zeta", Meta: map[string]any{"source": "unit", "page": float64(3)}},
		{ID: "b.md", Text: "one two three four five six"},
	}); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(New(store).Handler())
	defer ts.Close()

	// GET full document view: text, meta, and chunk spans.
	var view struct {
		ID    string          `json:"id"`
		Text  string          `json:"text"`
		Meta  json.RawMessage `json:"meta"`
		Spans []struct {
			Start int `json:"start"`
			End   int `json:"end"`
		} `json:"spans"`
	}
	getJSON(t, ts.URL+"/api/document?doc=a.md", &view)
	if view.Text == "" || len(view.Spans) == 0 {
		t.Fatalf("empty view: %+v", view)
	}
	if !strings.Contains(string(view.Meta), `"source":"unit"`) {
		t.Fatalf("meta missing: %s", view.Meta)
	}
	for _, sp := range view.Spans {
		if sp.Start < 0 || sp.End > len([]rune(view.Text)) || sp.Start >= sp.End {
			t.Fatalf("bad span %+v", sp)
		}
	}

	// DELETE removes it.
	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/api/document?doc=b.md", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete status %d", resp.StatusCode)
	}
	var docs struct {
		Documents []struct {
			ID string `json:"id"`
		} `json:"documents"`
	}
	getJSON(t, ts.URL+"/api/documents", &docs)
	if len(docs.Documents) != 1 || docs.Documents[0].ID != "a.md" {
		t.Fatalf("after delete: %+v", docs.Documents)
	}
}
