package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestProviderHeadersReachTheEndpoint is the point of the whole feature: a named
// provider's extra headers must be on the wire, not merely stored.
func TestProviderHeadersReachTheEndpoint(t *testing.T) {
	s, _ := newConfigServer(t)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"gen_api":"openrouter","gen_model":"m","providers":[{"name":"openrouter",
	  "base_url":"https://openrouter.ai/api","api_key":"sk-test",
	  "headers":{"HTTP-Referer":"https://turbograph.dev","X-Title":"turbograph"}}]}`
	resp, err := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("post status %d", resp.StatusCode)
	}
	got, ok := s.gen.(stubBackend)
	if !ok {
		t.Fatalf("backend was not rebuilt: %T", s.gen)
	}
	if got.ep.API != "openai" || got.ep.BaseURL != "https://openrouter.ai/api" {
		t.Fatalf("provider not resolved: %+v", got.ep)
	}
	if got.ep.APIKey != "sk-test" {
		t.Errorf("api key not passed through: %q", got.ep.APIKey)
	}
	if got.ep.Headers["HTTP-Referer"] != "https://turbograph.dev" || got.ep.Headers["X-Title"] != "turbograph" {
		t.Errorf("headers did not reach the endpoint: %+v", got.ep.Headers)
	}
}

// TestProviderKeyIsNeverReadBack: the key is write-only, like every other secret.
func TestProviderKeyIsNeverReadBack(t *testing.T) {
	s, _ := newConfigServer(t)
	s.cfg.Providers = []Provider{{Name: "p", BaseURL: "https://x.test", APIKey: "sk-secret"}}
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, _ := http.Get(ts.URL + "/api/config")
	raw, _ := json.Marshal(mustDecode(t, resp))
	if bytes.Contains(raw, []byte("sk-secret")) {
		t.Fatalf("config GET leaked a provider key: %s", raw)
	}
	var out struct {
		Providers []struct {
			Name   string `json:"name"`
			KeySet bool   `json:"key_set"`
		} `json:"providers"`
	}
	json.Unmarshal(raw, &out)
	if len(out.Providers) != 1 || !out.Providers[0].KeySet {
		t.Fatalf("provider key_set not reported: %+v", out.Providers)
	}
}

// TestProviderBlankKeyKeepsStoredSecret: the UI never receives the key, so it
// sends a blank back; that must not wipe it.
func TestProviderBlankKeyKeepsStoredSecret(t *testing.T) {
	s, _ := newConfigServer(t)
	s.cfg.Providers = []Provider{{Name: "p", BaseURL: "https://x.test", APIKey: "keepme"}}
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"gen_api":"ollama","providers":[{"name":"p","base_url":"https://x.test"}]}`
	http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(body))
	p, ok := s.cfg.provider("p")
	if !ok || p.APIKey != "keepme" {
		t.Fatalf("blank key wiped the stored provider secret: %+v", p)
	}
}

func TestProviderValidation(t *testing.T) {
	s, _ := newConfigServer(t)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	cases := map[string]string{
		"no name":             `{"providers":[{"base_url":"https://x.test"}]}`,
		"no url":              `{"providers":[{"name":"p"}]}`,
		"reserved name":       `{"providers":[{"name":"ollama","base_url":"https://x.test"}]}`,
		"duplicate name":      `{"providers":[{"name":"p","base_url":"https://x.test"},{"name":"p","base_url":"https://y.test"}]}`,
		"header injection":    `{"providers":[{"name":"p","base_url":"https://x.test","headers":{"X-A":"a\r\nX-Evil: b"}}]}`,
		"bad header name":     `{"providers":[{"name":"p","base_url":"https://x.test","headers":{"X A":"b"}}]}`,
		"metadata ssrf":       `{"providers":[{"name":"p","base_url":"http://169.254.169.254"}]}`,
		"unknown gen_api":     `{"gen_api":"nope"}`,
		"gen_api no provider": `{"gen_api":"ghost","providers":[{"name":"p","base_url":"https://x.test"}]}`,
	}
	for name, body := range cases {
		resp, err := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(body))
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: expected 400, got %d", name, resp.StatusCode)
		}
	}
}

// TestProviderKeyFromEnv: a key written as an env reference is resolved at use, so
// the secret itself never lands in the config file.
func TestProviderKeyFromEnv(t *testing.T) {
	t.Setenv("TG_TEST_PROVIDER_KEY", "sk-from-env")
	p := Provider{Name: "p", BaseURL: "https://x.test", APIKey: "${TG_TEST_PROVIDER_KEY}"}
	if got := p.endpoint().APIKey; got != "sk-from-env" {
		t.Errorf("env-referenced key not resolved: %q", got)
	}
	// A literal key containing a dollar sign stays literal.
	lit := Provider{Name: "p", BaseURL: "https://x.test", APIKey: "sk-a$b"}
	if got := lit.endpoint().APIKey; got != "sk-a$b" {
		t.Errorf("literal key was mangled: %q", got)
	}
	// An unset variable is left alone rather than silently becoming an empty key.
	unset := Provider{Name: "p", BaseURL: "https://x.test", APIKey: "$TG_TEST_NOT_SET"}
	if got := unset.endpoint().APIKey; got != "$TG_TEST_NOT_SET" {
		t.Errorf("unset env reference should stay verbatim, got %q", got)
	}
}

// TestProviderTestEndpoint probes a provider without saving it.
func TestProviderTestEndpoint(t *testing.T) {
	s, _ := newConfigServer(t)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	body := `{"name":"new","base_url":"https://x.test","headers":{"X-Ok":"1"}}`
	resp, err := http.Post(ts.URL+"/api/provider/test", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	out := mustDecode(t, resp)
	if out["ok"] != true {
		t.Fatalf("probe failed: %+v", out)
	}
	if len(s.cfg.Providers) != 0 {
		t.Errorf("probing must not save the provider, got %+v", s.cfg.Providers)
	}
	// A bad URL is refused before any request is made.
	bad, _ := http.Post(ts.URL+"/api/provider/test", "application/json",
		bytes.NewBufferString(`{"name":"n","base_url":"http://169.254.169.254"}`))
	if bad.StatusCode != http.StatusBadRequest {
		t.Errorf("probing the metadata endpoint should be 400, got %d", bad.StatusCode)
	}
}

func mustDecode(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestConfigMergePreservesProvidersOnPartialPost pins issue #15: a partial POST that
// omits "providers" must not destroy the stored providers and their API keys.
func TestConfigMergePreservesProvidersOnPartialPost(t *testing.T) {
	s, _ := newConfigServer(t)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Save a provider with a key, selected for generation.
	full := `{"providers":[{"name":"p1","base_url":"https://x.test","api_key":"sk-secret"}],
	  "gen_api":"p1","gen_model":"m1","embed_api":"ollama","embed_model":"e1"}`
	if r, _ := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(full)); r.StatusCode != 200 {
		t.Fatalf("save status %d", r.StatusCode)
	}
	// Change one unrelated field, omitting everything else.
	if r, _ := http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(`{"gen_model":"m2"}`)); r.StatusCode != 200 {
		t.Fatalf("partial status %d", r.StatusCode)
	}
	// The provider, its key, and the other fields must survive.
	p, ok := s.cfg.provider("p1")
	if !ok {
		t.Fatal("the stored provider was destroyed by a partial POST")
	}
	if p.APIKey != "sk-secret" {
		t.Errorf("the provider's API key was lost: %q", p.APIKey)
	}
	if s.cfg.GenAPI != "p1" || s.cfg.GenModel != "m2" || s.cfg.EmbedModel != "e1" {
		t.Errorf("merge dropped fields: gen_api=%q gen_model=%q embed_model=%q",
			s.cfg.GenAPI, s.cfg.GenModel, s.cfg.EmbedModel)
	}
	// An explicit empty array still clears providers (intentional, not an omission), as
	// long as nothing still references one -- so switch generation back to ollama too.
	http.Post(ts.URL+"/api/config", "application/json", bytes.NewBufferString(`{"providers":[],"gen_api":"ollama"}`))
	if len(s.cfg.Providers) != 0 {
		t.Errorf("an explicit empty providers array should clear them, got %d", len(s.cfg.Providers))
	}
}
