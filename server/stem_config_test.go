package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// TestConfigStemAppliesLive verifies the /api/config stem toggle reaches the open
// bucket's lexical index immediately: a Russian document indexed with stemming off
// does not match an inflected query, and after POSTing stem=true with the ru
// language it does.
func TestConfigStemAppliesLive(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, MinSimilarity: 0.05})
	if err := store.Build(context.Background(), []rag.Document{{ID: "car", Text: "машина едет"}}); err != nil {
		t.Fatal(err)
	}
	s := New(store)
	path := filepath.Join(t.TempDir(), "config.json")
	s.EnableConfig(RuntimeConfig{}, path, Factories{})
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	query := func() []string {
		body, _ := json.Marshal(queryRequest{Query: "машины", TopK: 5})
		resp, err := http.Post(ts.URL+"/query", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out struct {
			Results []queryResult `json:"results"`
		}
		json.NewDecoder(resp.Body).Decode(&out)
		var ids []string
		for _, r := range out.Results {
			ids = append(ids, r.DocID)
		}
		return ids
	}

	if got := query(); len(got) != 0 {
		t.Fatalf("stemming off: inflected query must not match, got %v", got)
	}

	// GET exposes the setting and the registered languages.
	resp, err := http.Get(ts.URL + "/api/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var cfg map[string]any
	json.NewDecoder(resp.Body).Decode(&cfg)
	if langs, _ := cfg["languages"].([]any); len(langs) == 0 {
		t.Fatalf("config GET must list registered stemmer languages, got %v", cfg["languages"])
	}

	post := map[string]any{"stem": true, "stem_langs": []string{"ru"}}
	body, _ := json.Marshal(post)
	resp, err = http.Post(ts.URL+"/api/config", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("post stem config status %d", resp.StatusCode)
	}
	if got := query(); len(got) != 1 || got[0] != "car" {
		t.Fatalf("stemming on: inflected query must match the document, got %v", got)
	}

	// An unknown language must be rejected without touching the live setting.
	bad := map[string]any{"stem": true, "stem_langs": []string{"zz"}}
	body, _ = json.Marshal(bad)
	resp, err = http.Post(ts.URL+"/api/config", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("unknown stemmer language must be rejected with 400, got %d", resp.StatusCode)
	}
	if got := query(); len(got) != 1 || got[0] != "car" {
		t.Fatalf("rejected update must not disable stemming, got %v", got)
	}
}
