package main

import (
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeOpenAI serves /v1/embeddings and /v1/chat/completions the way any
// OpenAI-compatible provider does, and records each chat call it receives.
type fakeOpenAI struct {
	mu    sync.Mutex
	chats []fakeChat
}

type fakeChat struct{ auth, model string }

func (f *fakeOpenAI) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/embeddings":
			var req struct {
				Input []string `json:"input"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			type item struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}
			out := struct {
				Data []item `json:"data"`
			}{}
			for i, s := range req.Input {
				v := make([]float32, 8)
				for j, c := range s {
					v[j%8] += float32(c%7) + 1
				}
				out.Data = append(out.Data, item{Index: i, Embedding: v})
			}
			json.NewEncoder(w).Encode(out)
		case "/v1/chat/completions":
			var req struct {
				Model string `json:"model"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			f.mu.Lock()
			f.chats = append(f.chats, fakeChat{auth: r.Header.Get("Authorization"), model: req.Model})
			f.mu.Unlock()
			content := "entity|Pandan Factory|project|an OCR model factory\n" +
				"entity|dots.mocr|product|a document OCR model\n" +
				"relation|Pandan Factory|dots.mocr|fine-tunes it\n"
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": content}}},
				"usage":   map[string]any{"prompt_tokens": 100, "completion_tokens": 20},
			})
		default:
			t.Errorf("unexpected request %s", r.URL.Path)
			http.NotFound(w, r)
		}
	})
}

// entities and ask reach a hosted OpenAI-compatible model through --llm-api,
// with the key sent as a bearer token and no Ollama involved.
func TestEntitiesAndAskUseOpenAIBackend(t *testing.T) {
	fake := &fakeOpenAI{}
	ts := httptest.NewServer(fake.handler(t))
	defer ts.Close()

	store := filepath.Join(t.TempDir(), "kb.tg")
	embed := []string{"--embed-api", "openai", "--embed-url", ts.URL, "--embed-model", "fake-embed"}
	llm := []string{"--llm-api", "openai", "--llm-url", ts.URL, "--llm-key", "sk-test", "--gen-model", "deepseek-chat"}

	text := "Pandan Factory fine-tunes dots.mocr to read classical Arabic books with their vowel marks."
	if err := cmdAdd(append([]string{"--store", store, "--id", "note", "--text", text}, embed...)); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := cmdEntities(append(append([]string{"--store", store, "--batch", "1", "--json"}, embed...), llm...)); err != nil {
		t.Fatalf("entities: %v", err)
	}
	if err := cmdAsk(append(append([]string{"--store", store, "--q", "what does Pandan Factory fine-tune?", "--json"}, embed...), llm...)); err != nil {
		t.Fatalf("ask: %v", err)
	}

	// token usage is totalled for cost reporting
	be := buildBackend(cliEndpoint("openai", ts.URL, "sk-test"))
	if _, err := be.Generate(t.Context(), "deepseek-chat", "", "hi"); err != nil {
		t.Fatal(err)
	}
	res := map[string]any{}
	if note := addUsage(res, be); note == "" || res["usage"].(map[string]int64)["prompt_tokens"] != 100 {
		t.Errorf("usage not reported: %q %v", note, res)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.chats) < 2 {
		t.Fatalf("want chat calls from entities and ask, got %d", len(fake.chats))
	}
	for _, c := range fake.chats {
		if c.auth != "Bearer sk-test" || c.model != "deepseek-chat" {
			t.Errorf("chat call: auth %q model %q", c.auth, c.model)
		}
	}

	// the graph the hosted model produced was saved with the store
	s, err := readStore(store, embedFlagsFor(t, embed).embedder())
	if err != nil {
		t.Fatal(err)
	}
	if s.EntityCount() == 0 {
		t.Errorf("entity graph is empty after extraction")
	}
}

// embedFlagsFor parses embedding flags the way a command would.
func embedFlagsFor(t *testing.T, args []string) embedOpts {
	t.Helper()
	fs := newFlagSet()
	eo := embedFlags(fs)
	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	return eo
}

// the key never has to appear on the command line
func TestLLMKeyFromEnvironment(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-env")
	fs := newFlagSet()
	g := genFlags(fs)
	fs.Parse([]string{"--llm-api", "openai", "--llm-url", "https://api.example.com"})
	ep := cliEndpoint(*g.api, orStr(*g.url, *g.ollamaURL), *g.key)
	if ep.API != "openai" || ep.APIKey != "sk-env" || !strings.HasPrefix(ep.BaseURL, "https://api.example.com") {
		t.Errorf("endpoint %+v", ep)
	}
}

func newFlagSet() *flag.FlagSet { return flag.NewFlagSet("test", flag.ContinueOnError) }

func TestDocFilter(t *testing.T) {
	f, err := docFilter([]string{"sessions/**", "projects/*/_card.md", "projects/**/README.md"})
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]bool{
		"sessions/2026-10-26-x":    true,
		"projects/kcd/_card.md":    true,
		"projects/a/b/_card.md":    false,
		"projects/a/b/README.md":   true,
		"projects/a/README.md.bak": false,
		"research/x/README.md":     false,
		"projects/a+b/README.md":   true,
	}
	for id, want := range cases {
		if got := f(id); got != want {
			t.Errorf("%s: got %v want %v", id, got, want)
		}
	}
	if f, _ := docFilter(nil); f != nil {
		t.Error("no patterns should mean no filter")
	}
}
