package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// hashEmbedder is a tiny deterministic embedder: it hashes tokens into a fixed
// dimension so shared vocabulary yields similar vectors, enough to drive the API
// end to end without a model.
type hashEmbedder struct{ dim int }

func (e hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, e.dim)
		for _, w := range strings.Fields(strings.ToLower(t)) {
			var h uint32 = 2166136261
			for _, c := range w {
				h = (h ^ uint32(c)) * 16777619
			}
			v[int(h)%e.dim] += 1
		}
		out[i] = v
	}
	return out, nil
}

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	store := rag.New(hashEmbedder{dim: 128}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []rag.Document{
		{ID: "a", Text: "graphs connect nodes with edges and weights"},
		{ID: "b", Text: "vector search finds nearest neighbors quickly"},
		{ID: "c", Text: "quantization compresses vectors to save memory"},
	}
	if err := store.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return httptest.NewServer(New(store).Handler())
}

func TestHealthAndStats(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("health failed: %v %v", err, resp.StatusCode)
	}

	resp, _ = http.Get(ts.URL + "/stats")
	var stats map[string]any
	json.NewDecoder(resp.Body).Decode(&stats)
	resp.Body.Close()
	if stats["chunks"].(float64) < 3 {
		t.Errorf("expected at least 3 chunks, got %v", stats["chunks"])
	}
}

func TestQueryEndpoint(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body, _ := json.Marshal(queryRequest{Query: "nearest neighbor vector search", TopK: 2})
	resp, err := http.Post(ts.URL+"/query", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out struct {
		Results []queryResult `json:"results"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Results) == 0 {
		t.Fatal("no results")
	}
	if out.Results[0].DocID != "b" {
		t.Errorf("expected doc b first for a vector-search query, got %s", out.Results[0].DocID)
	}
}

func TestIngestIncremental(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	req := ingestRequest{Documents: []rag.Document{{ID: "d", Text: "personalized pagerank propagates relevance over a graph"}}}
	body, _ := json.Marshal(req)
	resp, err := http.Post(ts.URL+"/ingest", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if out["chunks"].(float64) < 4 {
		t.Errorf("expected chunk count to grow after ingest, got %v", out["chunks"])
	}
}

func TestQueryValidation(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()
	resp, _ := http.Post(ts.URL+"/query", "application/json", strings.NewReader(`{"query":""}`))
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("empty query should be 400, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestUIAndGraphEndpoints(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	// UI is served at root.
	resp, err := http.Get(ts.URL + "/")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("ui failed: %v %v", err, resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/html") {
		t.Errorf("ui content-type = %s", ct)
	}
	resp.Body.Close()

	// Graph export has nodes and at least one edge for the test corpus.
	resp, _ = http.Get(ts.URL + "/api/graph")
	var gv struct {
		Nodes []map[string]any `json:"nodes"`
		Edges []map[string]any `json:"edges"`
	}
	json.NewDecoder(resp.Body).Decode(&gv)
	resp.Body.Close()
	if len(gv.Nodes) < 3 {
		t.Errorf("expected nodes in graph view, got %d", len(gv.Nodes))
	}

	// Models endpoint responds even without an Ollama client attached.
	resp, _ = http.Get(ts.URL + "/api/models")
	if resp.StatusCode != 200 {
		t.Errorf("models status %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Chat without a generator should still stream sources then an error event.
	body, _ := json.Marshal(chatRequest{Query: "vector search", TopK: 2})
	resp, _ = http.Post(ts.URL+"/api/chat", "application/json", bytes.NewReader(body))
	if resp.StatusCode != 200 {
		t.Errorf("chat status %d", resp.StatusCode)
	}
	buf := make([]byte, 4096)
	n, _ := resp.Body.Read(buf)
	resp.Body.Close()
	if !strings.Contains(string(buf[:n]), "event: sources") {
		t.Errorf("chat did not emit a sources event: %s", buf[:n])
	}
}

func TestBucketsIsolation(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	// Create a second bucket and ingest a distinctive document into it.
	resp, _ := http.Post(ts.URL+"/api/buckets?bucket=other", "application/json", nil)
	if resp.StatusCode != 200 {
		t.Fatalf("create bucket status %d", resp.StatusCode)
	}
	resp.Body.Close()

	body, _ := json.Marshal(ingestRequest{Documents: []rag.Document{
		{ID: "z", Text: "zebra distinctive content only in the other bucket"}}})
	resp, _ = http.Post(ts.URL+"/ingest?bucket=other", "application/json", bytes.NewReader(body))
	resp.Body.Close()

	// The buckets list should include both, and the default bucket must not see
	// the other bucket's document.
	resp, _ = http.Get(ts.URL + "/api/buckets")
	var bl struct {
		Buckets []map[string]any `json:"buckets"`
	}
	json.NewDecoder(resp.Body).Decode(&bl)
	resp.Body.Close()
	if len(bl.Buckets) < 2 {
		t.Errorf("expected at least 2 buckets, got %d", len(bl.Buckets))
	}

	q, _ := json.Marshal(queryRequest{Query: "zebra distinctive", TopK: 3})
	resp, _ = http.Post(ts.URL+"/query", "application/json", bytes.NewReader(q)) // default bucket
	var out struct {
		Results []queryResult `json:"results"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	for _, r := range out.Results {
		if r.DocID == "z" {
			t.Error("default bucket leaked a document from the other bucket")
		}
	}

	// The other bucket must find it.
	resp, _ = http.Post(ts.URL+"/query?bucket=other", "application/json", bytes.NewReader(q))
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if len(out.Results) == 0 || out.Results[0].DocID != "z" {
		t.Errorf("other bucket did not return its document: %+v", out.Results)
	}
}
