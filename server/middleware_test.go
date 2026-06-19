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

func newOptServer(t *testing.T, opt Options) *httptest.Server {
	t.Helper()
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	if err := store.Build(context.Background(), []rag.Document{{ID: "a", Text: "graphs and vectors"}}); err != nil {
		t.Fatal(err)
	}
	return httptest.NewServer(New(store).HandlerWithOptions(opt))
}

func TestRecoverPanic(t *testing.T) {
	// A handler that panics must yield a 500, not crash the test process.
	h := recoverPanic(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom")
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/x", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("panic should produce 500, got %d", rec.Code)
	}
}

func TestBodyLimit(t *testing.T) {
	ts := newOptServer(t, Options{MaxBodyBytes: 16})
	defer ts.Close()
	big := bytes.NewReader(make([]byte, 1024))
	resp, err := http.Post(ts.URL+"/query", "application/json", big)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("oversized body should be rejected, got %d", resp.StatusCode)
	}
}

func TestAPIKeyAuth(t *testing.T) {
	ts := newOptServer(t, Options{APIKey: "secret"})
	defer ts.Close()

	// No key: rejected.
	resp, _ := http.Get(ts.URL + "/stats")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("missing key should be 401, got %d", resp.StatusCode)
	}
	// Health is always open.
	resp, _ = http.Get(ts.URL + "/healthz")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("health should be open, got %d", resp.StatusCode)
	}
	// Valid key via each accepted channel.
	for _, mk := range []func(*http.Request){
		func(r *http.Request) { r.Header.Set("X-API-Key", "secret") },
		func(r *http.Request) { r.Header.Set("Authorization", "Bearer secret") },
	} {
		req, _ := http.NewRequest("GET", ts.URL+"/stats", nil)
		mk(req)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("valid key should pass, got %d", resp.StatusCode)
		}
	}
	// Valid key via query parameter.
	resp, _ = http.Get(ts.URL + "/stats?api_key=secret")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("query-param key should pass, got %d", resp.StatusCode)
	}
	// Wrong key: rejected.
	req, _ := http.NewRequest("GET", ts.URL+"/stats", nil)
	req.Header.Set("X-API-Key", "wrong")
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong key should be 401, got %d", resp.StatusCode)
	}
}

func TestCORSPreflight(t *testing.T) {
	ts := newOptServer(t, Options{CORSOrigin: "*"})
	defer ts.Close()
	req, _ := http.NewRequest(http.MethodOptions, ts.URL+"/query", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("preflight should be 204, got %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("missing CORS origin, got %q", got)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	ts := newOptServer(t, Options{Metrics: true})
	defer ts.Close()
	http.Get(ts.URL + "/healthz") // generate a request to count
	resp, err := http.Get(ts.URL + "/debug/vars")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body := new(bytes.Buffer)
	body.ReadFrom(resp.Body)
	if !strings.Contains(body.String(), "turbograph_requests_total") {
		t.Fatalf("expvar metrics missing turbograph counters")
	}
}

func TestHealthReportsVersion(t *testing.T) {
	ts := newOptServer(t, Options{Version: "v9.9.9"})
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]string
	json.NewDecoder(resp.Body).Decode(&out)
	if out["version"] != "v9.9.9" {
		t.Fatalf("health should report the configured version, got %q", out["version"])
	}
}

func TestReadyz(t *testing.T) {
	// No generator configured: readiness is immediately ok.
	ts := newOptServer(t, Options{})
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/readyz")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("readyz without generator should be 200, got %d", resp.StatusCode)
	}
}
