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

// metaTestServer builds a server whose two documents share vocabulary but carry
// different metadata, so meta_filter is the only way to tell them apart.
func metaTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []rag.Document{
		{ID: "slk", Text: "needle alpha slack channel", Meta: map[string]any{"source": "slack"}},
		{ID: "eml", Text: "needle beta email thread", Meta: map[string]any{"source": "email"}},
	}
	if err := store.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return httptest.NewServer(New(store).Handler())
}

func TestQueryMetaFilterRestrictsResults(t *testing.T) {
	ts := metaTestServer(t)
	defer ts.Close()

	post := func(filter []string) (*http.Response, []queryResult) {
		body, _ := json.Marshal(queryRequest{Query: "needle", TopK: 5, MetaFilter: filter})
		resp, err := http.Post(ts.URL+"/query", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out struct {
			Results []queryResult `json:"results"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		return resp, out.Results
	}

	resp, res := post([]string{"source=email"})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if len(res) == 0 {
		t.Fatal("expected results")
	}
	for _, r := range res {
		if r.DocID != "eml" {
			t.Fatalf("got doc %q, want only the email document", r.DocID)
		}
	}

	_, res = post([]string{"source=slack", "source=email"})
	if len(res) != 0 {
		t.Fatalf("conflicting AND pair must yield nothing, got %d", len(res))
	}

	body, _ := json.Marshal(queryRequest{Query: "needle", TopK: 5, MetaFilter: []string{"noequals"}})
	resp, err := http.Post(ts.URL+"/query", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("malformed meta_filter must be rejected with 400, got %d", resp.StatusCode)
	}
}

func TestChatMetaFilterRestrictsSources(t *testing.T) {
	ts, oll := metaGenServer(t)
	defer ts.Close()
	defer oll.Close()

	body := `{"query":"needle","top_k":2,"meta_filter":["source=slack"],"model":"qwen3.5:2b"}`
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	out := buf.String()

	idx := strings.Index(out, "event: sources")
	if idx < 0 {
		t.Fatalf("missing sources event in:\n%s", out)
	}
	data := out[idx+len("event: sources"):]
	i := strings.Index(data, "data: ")
	if i < 0 {
		t.Fatalf("sources event without data in:\n%s", out)
	}
	line := data[i+len("data: "):]
	if nl := strings.IndexByte(line, '\n'); nl >= 0 {
		line = line[:nl]
	}
	var ev struct {
		Sources []queryResult `json:"sources"`
	}
	if err := json.Unmarshal([]byte(line), &ev); err != nil {
		t.Fatalf("parse sources event: %v\n%s", err, out)
	}
	if len(ev.Sources) == 0 {
		t.Fatal("expected sources")
	}
	for _, s := range ev.Sources {
		if s.DocID != "slk" {
			t.Fatalf("chat source doc %q, want only the slack document", s.DocID)
		}
	}
}

// metaGenServer is metaTestServer plus a fake generation backend so /api/chat
// runs end to end.
func metaGenServer(t *testing.T) (*httptest.Server, *httptest.Server) {
	t.Helper()
	oll := fakeOllamaServer()
	client := ollama.New()
	client.BaseURL = oll.URL
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []rag.Document{
		{ID: "slk", Text: "needle alpha slack channel", Meta: map[string]any{"source": "slack"}},
		{ID: "eml", Text: "needle beta email thread", Meta: map[string]any{"source": "email"}},
	}
	if err := store.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	s := New(store)
	s.SetGenerator(pullClient{client}, "qwen3.5:2b", "embeddinggemma")
	return httptest.NewServer(s.Handler()), oll
}
