package rag

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Gaurav-Gosain/turbograph/ollama"
)

// TestLeanStorageModes measures the storage / load-time / recall trade-off of the
// three VectorModes on a real corpus with a real embedder, so the decision to
// integrate (or not) is grounded rather than asserted. Skipped unless TG_LEAN=1
// and Ollama is reachable; never runs in CI.
//
//	TG_LEAN=1 go test ./rag/ -run TestLeanStorageModes -v -timeout 20m
//
// Recall is measured against the exact-mode ranking (standard ANN recall-vs-exact):
// the query embedding is identical across modes, so any difference is purely how
// much the decoded/recomputed document vectors perturb the neighbour ordering.
func TestLeanStorageModes(t *testing.T) {
	if os.Getenv("TG_LEAN") == "" {
		t.Skip("set TG_LEAN=1 (and have Ollama running) to run the lean-storage benchmark")
	}
	embedModel := envOrLean("TG_LEAN_EMBED", "nomic-embed-text")
	client := ollama.New()
	if url := os.Getenv("TG_LEAN_URL"); url != "" {
		client.BaseURL = url
	}
	client.SetEmbedModel(embedModel)
	ctx := context.Background()

	docs := repoProseDocs(t)
	if len(docs) < 5 {
		t.Fatalf("need a few prose docs, got %d", len(docs))
	}
	t.Logf("corpus: %d docs from repo prose, embed=%s", len(docs), embedModel)

	cfg := Config{Seed: 1, GraphKNN: 6, MinSimilarity: 0.05, Chunk: ChunkConfig{TargetWords: 120}}
	if b := os.Getenv("TG_LEAN_BITS"); b != "" {
		if n, err := strconv.Atoi(b); err == nil {
			cfg.Bits = n
		}
	}
	t.Logf("quantizer bits=%d (codes-mode fidelity; storage is bits-independent)", cfg.Bits)
	base := New(client, cfg)
	if err := base.Build(ctx, docs); err != nil {
		t.Fatalf("build: %v", err)
	}
	nChunks := base.Len()

	// Queries: the text of a spread of chunks, so each has a true home in the index
	// and meaningful neighbours. The recall metric is codes/text top-k vs exact top-k.
	var queries []string
	base.mu.RLock()
	for i := 0; i < len(base.chunks); i += maxIntLean(1, len(base.chunks)/40) {
		queries = append(queries, firstNWords(base.chunks[i].Text, 20))
	}
	base.mu.RUnlock()

	const k = 10
	type modeInfo struct {
		name string
		mode VectorMode
	}
	modes := []modeInfo{{"exact", VectorsExact}, {"codes", VectorsCodes}, {"text", VectorsNone}}

	// Ground truth: exact-mode top-k per query.
	gold := map[string][]string{}
	var goldBytes int
	for _, m := range modes {
		var buf bytes.Buffer
		if err := base.SaveLean(&buf, m.mode); err != nil {
			t.Fatalf("save %s: %v", m.name, err)
		}
		size := buf.Len()
		t0 := time.Now()
		st, err := Load(client, &buf)
		if err != nil {
			t.Fatalf("load %s: %v", m.name, err)
		}
		loadMS := float64(time.Since(t0).Microseconds()) / 1000

		var recallSum float64
		for _, q := range queries {
			res, err := st.Retrieve(ctx, q, RetrieveParams{TopK: k})
			if err != nil {
				t.Fatalf("retrieve %s: %v", m.name, err)
			}
			ids := topIDs(res, k)
			if m.mode == VectorsExact {
				gold[q] = ids
				recallSum += 1
				continue
			}
			recallSum += overlap(ids, gold[q]) / float64(len(gold[q]))
		}
		recall := recallSum / float64(len(queries))
		if m.mode == VectorsExact {
			goldBytes = size
		}
		t.Logf("%-6s  size=%7d B (%5.1f%% of exact, %4.0f B/chunk)  load=%7.1f ms  recall@%d-vs-exact=%.3f",
			m.name, size, 100*float64(size)/float64(goldBytes), float64(size)/float64(nChunks), loadMS, k, recall)
	}
	t.Logf("chunks=%d  (recall is top-%d overlap with the exact ranking; text mode re-embeds on load)", nChunks, k)
}

func envOrLean(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func maxIntLean(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func firstNWords(s string, n int) string {
	f := strings.Fields(s)
	if len(f) > n {
		f = f[:n]
	}
	return strings.Join(f, " ")
}

func topIDs(res []Retrieved, k int) []string {
	if len(res) > k {
		res = res[:k]
	}
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.Chunk.ID
	}
	return out
}

func overlap(a, b []string) float64 {
	set := make(map[string]struct{}, len(b))
	for _, x := range b {
		set[x] = struct{}{}
	}
	var n float64
	for _, x := range a {
		if _, ok := set[x]; ok {
			n++
		}
	}
	return n
}

// repoProseDocs loads the repository's markdown as a real, varied prose corpus.
func repoProseDocs(t *testing.T) []Document {
	t.Helper()
	roots := []string{"../README.md", "../docs", "../ROADMAP.md", "../CHANGELOG.md"}
	var docs []Document
	add := func(path string) {
		b, err := os.ReadFile(path)
		if err != nil || len(b) < 200 {
			return
		}
		docs = append(docs, Document{ID: filepath.Base(path), Text: string(b)})
	}
	for _, r := range roots {
		info, err := os.Stat(r)
		if err != nil {
			continue
		}
		if info.IsDir() {
			entries, _ := os.ReadDir(r)
			for _, e := range entries {
				if strings.HasSuffix(e.Name(), ".md") {
					add(filepath.Join(r, e.Name()))
				}
			}
		} else {
			add(r)
		}
	}
	return docs
}
