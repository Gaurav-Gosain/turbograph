package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
	"github.com/Gaurav-Gosain/turbograph/script"
)

// scriptServer stands up a server whose operator registered the given scripts.
func scriptServer(t *testing.T, files map[string]string) *httptest.Server {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	reg, err := script.Load(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	s := New(rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, MinSimilarity: 0.01}))
	s.SetScripts(reg)
	return httptest.NewServer(s.Handler())
}

func postJSON(t *testing.T, url, body string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Post(url, "application/json", bytes.NewReader([]byte(body)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

// TestIngestTransformPipeline drives the whole feature over HTTP: a script that
// rewrites text and sets metadata, one that drops a document, and one that fails,
// all in a single ingest. Failures and drops are isolated per document.
func TestIngestTransformPipeline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell scripts are not executable on windows")
	}
	ts := scriptServer(t, map[string]string{
		// Tags every document, keeping its text.
		"tag.sh": `#!/bin/sh
read -r line
text=$(printf '%s' "$line" | sed 's/.*"text":"\([^"]*\)".*/\1/')
printf '{"text":"%s","meta":{"tagged":true}}' "$text"
`,
		// Drops anything whose text mentions "junk".
		"dropjunk.sh": `#!/bin/sh
read -r line
if printf '%s' "$line" | grep -q junk; then printf '{"drop":true}'; else
  text=$(printf '%s' "$line" | sed 's/.*"text":"\([^"]*\)".*/\1/')
  printf '{"text":"%s"}' "$text"
fi
`,
		"boom.sh": "#!/bin/sh\ncat >/dev/null\necho 'script exploded' >&2\nexit 1\n",
	})
	defer ts.Close()

	// The UI can discover what the operator registered.
	resp, err := http.Get(ts.URL + "/api/scripts")
	if err != nil {
		t.Fatal(err)
	}
	var list struct {
		Scripts []string `json:"scripts"`
	}
	json.NewDecoder(resp.Body).Decode(&list)
	resp.Body.Close()
	if len(list.Scripts) != 3 {
		t.Fatalf("expected 3 registered scripts, got %v", list.Scripts)
	}

	// A chained transform: tag, then drop junk. One good doc, one junk doc.
	code, out := postJSON(t, ts.URL+"/api/ingest", `{"documents":[
		{"id":"good","text":"a real document about vectors"},
		{"id":"bad","text":"this is junk"}
	],"transform":["tag.sh","dropjunk.sh"]}`)
	if code != http.StatusOK {
		t.Fatalf("ingest status %d: %v", code, out)
	}
	dropped, _ := out["dropped"].([]any)
	if len(dropped) != 1 || dropped[0] != "bad" {
		t.Errorf("expected the junk document to be dropped, got %v", out["dropped"])
	}
	if n, _ := out["chunks"].(float64); n < 1 {
		t.Errorf("the surviving document should have been indexed, chunks=%v", out["chunks"])
	}

	// A failing script isolates to its document and reports why.
	code, out = postJSON(t, ts.URL+"/api/ingest", `{"documents":[{"id":"x","text":"hello"}],"transform":["boom.sh"]}`)
	if code != http.StatusOK {
		t.Fatalf("a failing script should not fail the request, got %d", code)
	}
	failed, _ := out["failed"].([]any)
	if len(failed) != 1 {
		t.Fatalf("expected 1 failed document, got %v", out["failed"])
	}
	if f, ok := failed[0].(map[string]any); !ok || f["id"] != "x" {
		t.Errorf("failure not attributed to the document: %v", failed[0])
	}
}

// TestIngestRejectsUnknownScript is the security boundary at the HTTP edge: a
// caller cannot name anything the operator did not register, and cannot smuggle a
// command in through the name.
func TestIngestRejectsUnknownScript(t *testing.T) {
	ts := scriptServer(t, map[string]string{})
	defer ts.Close()

	for _, name := range []string{"/bin/sh", "../../bin/sh", "rm -rf /", "whatever.sh"} {
		body, _ := json.Marshal(map[string]any{
			"documents": []map[string]string{{"id": "d", "text": "hi"}},
			"transform": []string{name},
		})
		code, _ := postJSON(t, ts.URL+"/api/ingest", string(body))
		if code != http.StatusBadRequest {
			t.Errorf("transform %q should be rejected with 400, got %d", name, code)
		}
	}
}

// TestIngestWithoutTransformsIsUnchanged: the feature is inert unless asked for.
func TestIngestWithoutTransformsIsUnchanged(t *testing.T) {
	ts := scriptServer(t, map[string]string{})
	defer ts.Close()
	code, out := postJSON(t, ts.URL+"/api/ingest", `{"documents":[{"id":"d","text":"plain ingest still works"}]}`)
	if code != http.StatusOK {
		t.Fatalf("plain ingest broke: %d %v", code, out)
	}
	if n, _ := out["chunks"].(float64); n < 1 {
		t.Errorf("expected the document to be indexed, got %v", out)
	}
}
