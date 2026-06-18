package rag

import (
	"context"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// keywordEmbedder maps each vocabulary word to a fixed random unit vector and
// embeds text as the normalized sum of its word vectors. Cosine similarity then
// reflects shared vocabulary, which is enough to exercise the pipeline and the
// graph deterministically without a live model.
type keywordEmbedder struct {
	mu    sync.RWMutex
	dim   int
	basis map[string][]float32
}

func newKeywordEmbedder(dim int) *keywordEmbedder {
	return &keywordEmbedder{dim: dim, basis: map[string][]float32{}}
}

// vec is safe for concurrent use so the embedder can stand in for a real,
// concurrency-safe client (such as the Ollama HTTP client) under -race.
func (e *keywordEmbedder) vec(word string) []float32 {
	e.mu.RLock()
	v, ok := e.basis[word]
	e.mu.RUnlock()
	if ok {
		return v
	}
	var seed uint64 = 1469598103934665603
	for _, c := range word {
		seed = (seed ^ uint64(c)) * 1099511628211
	}
	rng := quant.NewPCG(seed)
	v = make([]float32, e.dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	e.mu.Lock()
	e.basis[word] = v
	e.mu.Unlock()
	return v
}

func (e *keywordEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		acc := make([]float32, e.dim)
		for _, w := range strings.Fields(strings.ToLower(t)) {
			wv := e.vec(w)
			for j := range acc {
				acc[j] += wv[j]
			}
		}
		var n float64
		for _, x := range acc {
			n += float64(x) * float64(x)
		}
		n = math.Sqrt(n)
		if n == 0 {
			acc[0] = 1
			n = 1
		}
		for j := range acc {
			acc[j] /= float32(n)
		}
		out[i] = acc
	}
	return out, nil
}

func TestChunking(t *testing.T) {
	text := strings.Repeat("word ", 300)
	chunks := chunkDocument("d", text, ChunkConfig{TargetWords: 100, OverlapWords: 20})
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	for i, c := range chunks {
		if c.Pos != i || c.DocID != "d" {
			t.Errorf("chunk %d metadata wrong: %+v", i, c)
		}
		if len(strings.Fields(c.Text)) > 100 {
			t.Errorf("chunk %d too large", i)
		}
	}
}

func bridgeCorpus() []Document {
	return []Document{
		{ID: "a1", Text: "alpha alpha alpha foundational core concept"},
		{ID: "a2", Text: "alpha alpha core principle theory"},
		{ID: "a3", Text: "alpha alpha primary subject matter"},
		{ID: "bridge", Text: "alpha omega bridge linking the two distinct domains"},
		{ID: "o1", Text: "omega omega omega downstream application phase"},
		{ID: "o2", Text: "omega omega resulting outcome stage"},
		{ID: "noise", Text: "zeta zeta unrelated random filler content"},
	}
}

func newTestStore(t *testing.T) *Store {
	t.Helper()
	emb := newKeywordEmbedder(96)
	st := New(emb, Config{
		Bits: 5, ResidualDims: 32, Seed: 1,
		GraphKNN: 4, MinSimilarity: 0.15, SequentialWeight: 0.0,
		Chunk: ChunkConfig{TargetWords: 200, OverlapWords: 0},
	})
	if err := st.Build(context.Background(), bridgeCorpus()); err != nil {
		t.Fatal(err)
	}
	return st
}

func TestBuildPopulates(t *testing.T) {
	st := newTestStore(t)
	if st.Len() != 7 {
		t.Fatalf("expected 7 chunks, got %d", st.Len())
	}
	if st.g == nil || st.g.N() != 7 {
		t.Fatal("graph not built")
	}
}

func TestRetrieveRanksRelevantFirst(t *testing.T) {
	st := newTestStore(t)
	res, err := st.Retrieve(context.Background(), "alpha core principle", RetrieveParams{TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("no results")
	}
	for _, r := range res {
		if strings.HasPrefix(r.Chunk.DocID, "o") || r.Chunk.DocID == "noise" {
			t.Errorf("irrelevant chunk %s ranked in top-3", r.Chunk.DocID)
		}
	}
}

// TestGraphPropagationReachesNonSimilar is the defining graph-RAG property: a
// chunk with near-zero direct similarity to the query is still reached through
// the similarity graph. With pure-similarity scoring it is invisible; with graph
// mixing it gains positive score.
func TestGraphPropagationReachesNonSimilar(t *testing.T) {
	st := newTestStore(t)
	ctx := context.Background()

	// Pure similarity: omega chunks should be essentially unreachable for an
	// alpha query.
	pure, err := st.Retrieve(ctx, "alpha core principle", RetrieveParams{TopK: 7, GraphMix: 0})
	if err != nil {
		t.Fatal(err)
	}
	// Graph propagation: alpha -> bridge -> omega should give omega mass.
	graphed, err := st.Retrieve(ctx, "alpha core principle", RetrieveParams{TopK: 7, GraphMix: 1.0})
	if err != nil {
		t.Fatal(err)
	}

	scoreOf := func(res []Retrieved, doc string) float32 {
		for _, r := range res {
			if r.Chunk.DocID == doc {
				return r.Score
			}
		}
		return 0
	}
	pureO1 := scoreOf(pure, "o1")
	graphO1 := scoreOf(graphed, "o1")
	if graphO1 <= pureO1 {
		t.Errorf("graph propagation did not raise omega score: pure=%.4f graph=%.4f", pureO1, graphO1)
	}
	// The bridge chunk must be reachable and present under graph scoring.
	if scoreOf(graphed, "bridge") <= 0 {
		t.Error("bridge chunk not surfaced by graph scoring")
	}
}

func TestRetrieveDeterministic(t *testing.T) {
	a := newTestStore(t)
	b := newTestStore(t)
	ra, _ := a.Retrieve(context.Background(), "alpha principle", RetrieveParams{TopK: 5})
	rb, _ := b.Retrieve(context.Background(), "alpha principle", RetrieveParams{TopK: 5})
	if len(ra) != len(rb) {
		t.Fatal("length mismatch")
	}
	for i := range ra {
		if ra[i].Chunk.ID != rb[i].Chunk.ID {
			t.Errorf("nondeterministic retrieval at %d: %s vs %s", i, ra[i].Chunk.ID, rb[i].Chunk.ID)
		}
	}
}
