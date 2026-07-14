package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// stubBackend is an inert Backend for config tests. It records the endpoint it was
// built from so tests can assert what the server resolved.
type stubBackend struct {
	model string
	ep    Endpoint
}

func (stubBackend) Generate(context.Context, string, string, string) (string, error) { return "", nil }
func (stubBackend) GenerateStream(context.Context, string, string, string, func(string) error) error {
	return nil
}
func (b stubBackend) ListModels(context.Context) ([]string, error) {
	return []string{"stub-" + b.ep.BaseURL}, nil
}
func (stubBackend) Ping(context.Context) error { return nil }

func newConfigServer(t *testing.T) (*Server, string) {
	t.Helper()
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1})
	if err := store.Build(context.Background(), []rag.Document{{ID: "a", Text: "hello world"}}); err != nil {
		t.Fatal(err)
	}
	s := New(store)
	path := filepath.Join(t.TempDir(), "config.json")
	s.EnableConfig(RuntimeConfig{
		GenAPI: "ollama", GenModel: "m1", EmbedModel: "e1", ChunkStrategy: rag.StrategyRecursive,
	}, path, Factories{
		Backend:  func(ep Endpoint) Backend { return stubBackend{model: ep.API, ep: ep} },
		Embedder: func(Endpoint, string, int) rag.Embedder { return hashEmbedder{dim: 64} },
	})
	return s, path
}

func TestConfigGetRedactsSecrets(t *testing.T) {
	s, _ := newConfigServer(t)
	s.cfg.GenKey = "supersecret"
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, _ := http.Get(ts.URL + "/api/config")
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	if _, leaked := out["gen_key"]; leaked {
		t.Fatal("config GET leaked the raw API key")
	}
	if out["gen_key_set"] != true {
		t.Fatalf("gen_key_set should be true, got %v", out["gen_key_set"])
	}
	if out["gen_model"] != "m1" || out["editable"] != true {
		t.Fatalf("unexpected config: %+v", out)
	}
}

func TestConfigPostPersistsAndApplies(t *testing.T) {
	s, path := newConfigServer(t)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"gen_api":"ollama","gen_model":"m2","embed_model":"e2","chunk_strategy":"markdown","chunk_words":200}`
	resp, err := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("post config status %d", resp.StatusCode)
	}
	// Applied live: the manager's new-bucket chunking changed.
	if got := s.mgr.Config().Chunk.Strategy; got != "markdown" {
		t.Errorf("chunk strategy not applied to manager: %q", got)
	}
	if s.genModel != "m2" {
		t.Errorf("gen model not applied: %q", s.genModel)
	}
	// Persisted to disk.
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("config file not written: %v", err)
	}
	var saved RuntimeConfig
	json.Unmarshal(b, &saved)
	if saved.GenModel != "m2" || saved.ChunkStrategy != "markdown" {
		t.Errorf("config not persisted correctly: %+v", saved)
	}
}

func TestConfigPostKeepsExistingSecret(t *testing.T) {
	s, _ := newConfigServer(t)
	s.cfg.GenKey = "keepme"
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	// A post without gen_key must not wipe the stored one.
	http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(`{"gen_api":"ollama","gen_model":"x"}`))
	if s.cfg.GenKey != "keepme" {
		t.Fatalf("blank key in POST wiped the stored secret: %q", s.cfg.GenKey)
	}
}

func TestConfigPostValidates(t *testing.T) {
	s, _ := newConfigServer(t)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	// openai backend without a URL is rejected.
	resp, _ := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(`{"embed_api":"openai"}`))
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("invalid config should be 400, got %d", resp.StatusCode)
	}
}

func TestStatusAggregates(t *testing.T) {
	s, _ := newConfigServer(t)
	s.SetGenerator(stubBackend{}, "m1", "e1")
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	if out["storage"] == nil || out["generation"] == nil || out["stats"] == nil {
		t.Fatalf("status missing sections: %+v", out)
	}
	gen := out["generation"].(map[string]any)
	if gen["reachable"] != true { // stubBackend.Ping returns nil
		t.Errorf("generation should be reachable, got %v", gen["reachable"])
	}
	st := out["stats"].(map[string]any)
	if st["chunk_strategy"] == nil {
		t.Errorf("stats should report the chunk strategy: %+v", st)
	}
}

func TestConfigDisabledWhenNotEnabled(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1})
	store.Build(context.Background(), []rag.Document{{ID: "a", Text: "x"}})
	ts := httptest.NewServer(New(store).Handler()) // no EnableConfig
	defer ts.Close()
	resp, _ := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(`{}`))
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("config edit should be 403 when disabled, got %d", resp.StatusCode)
	}
}

func TestValidateBackendURLBlocksMetadataSSRF(t *testing.T) {
	ok := []string{"", "http://127.0.0.1:11434", "https://api.openai.com", "http://localhost:8080", "http://10.0.0.5:1234"}
	for _, u := range ok {
		if err := validateBackendURL(u); err != nil {
			t.Errorf("expected %q allowed, got %v", u, err)
		}
	}
	bad := []string{"http://169.254.169.254/latest/meta-data/", "https://metadata.google.internal/", "file:///etc/passwd", "gopher://x", "http://"}
	for _, u := range bad {
		if err := validateBackendURL(u); err == nil {
			t.Errorf("expected %q rejected", u)
		}
	}
}
