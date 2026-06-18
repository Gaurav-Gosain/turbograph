package rag

import (
	"context"
	"testing"
)

// asymEmbedder records whether the query path or the document path was used, and
// embeds queries into a distinct subspace so the routing is observable.
type asymEmbedder struct {
	*keywordEmbedder
	queryCalls int
}

func (e *asymEmbedder) EmbedQuery(ctx context.Context, texts []string) ([][]float32, error) {
	e.queryCalls++
	return e.keywordEmbedder.Embed(ctx, texts)
}

// TestRetrieveUsesQueryEmbedder confirms the store routes the query through
// EmbedQuery when the embedder implements QueryEmbedder.
func TestRetrieveUsesQueryEmbedder(t *testing.T) {
	ctx := context.Background()
	emb := &asymEmbedder{keywordEmbedder: newKeywordEmbedder(64)}
	s := New(emb, Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05})
	docs := []Document{
		{ID: "a", Text: "graphs connect nodes with edges"},
		{ID: "b", Text: "vectors are embedded and quantized"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	before := emb.queryCalls
	if _, err := s.Retrieve(ctx, "graphs", RetrieveParams{TopK: 2}); err != nil {
		t.Fatal(err)
	}
	if emb.queryCalls != before+1 {
		t.Fatalf("expected one EmbedQuery call, got %d", emb.queryCalls-before)
	}
}

// TestGraphMixDefaultIsOff verifies the graph is opt-in: the zero value and a
// negative value both mean no graph (identical rankings), while a positive
// GraphMix can reorder results. The zero-value default must equal the pure path.
func TestGraphMixDefaultIsOff(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []Document{
		{ID: "a", Text: "alpha topic shared core idea"},
		{ID: "b", Text: "alpha topic shared core idea variant"},
		{ID: "c", Text: "beta unrelated separate matter"},
		{ID: "d", Text: "gamma another distinct theme"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	def, err := s.Retrieve(ctx, "alpha topic", RetrieveParams{TopK: 3, GraphMix: 0})
	if err != nil || len(def) == 0 {
		t.Fatalf("default retrieval failed: %v (%d results)", err, len(def))
	}
	neg, err := s.Retrieve(ctx, "alpha topic", RetrieveParams{TopK: 3, GraphMix: -1})
	if err != nil {
		t.Fatal(err)
	}
	// Zero (default) and negative both mean graph-off, so the rankings must match.
	if len(def) != len(neg) {
		t.Fatalf("default and negative GraphMix differ in length: %d vs %d", len(def), len(neg))
	}
	for i := range def {
		if def[i].Chunk.ID != neg[i].Chunk.ID {
			t.Errorf("default GraphMix is not graph-off: rank %d %q vs %q", i, def[i].Chunk.ID, neg[i].Chunk.ID)
		}
	}
}

// TestPRFRunsAndStaysSane verifies pseudo-relevance feedback executes (the extra
// ANN search and query expansion) and still returns the obviously relevant chunk
// for a clear query, i.e. expansion does not cause runaway drift on easy cases.
func TestPRFRunsAndStaysSane(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	docs := []Document{
		{ID: "graph", Text: "graphs connect nodes with edges and weights"},
		{ID: "vector", Text: "vector search finds nearest neighbors quickly"},
		{ID: "quant", Text: "quantization compresses vectors to save memory"},
		{ID: "bm25", Text: "bm25 ranks documents by term frequency"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	res, err := s.Retrieve(ctx, "nearest neighbor vector search", RetrieveParams{
		TopK: 2, GraphMix: -1, PRF: 3, PRFWeight: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].Chunk.DocID != "vector" {
		t.Fatalf("PRF retrieval should keep the on-topic doc first, got %+v", docids(res))
	}
}

func docids(res []Retrieved) []string {
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.Chunk.DocID
	}
	return out
}

// TestLexicalWeightLiftsExactMatch checks the additive BM25 term: a document that
// contains a rare query term but is not the dense-nearest should rank higher with
// the lexical weight on than with it forced off, without the graph involved.
func TestLexicalWeightLiftsExactMatch(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(128), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
		Chunk: ChunkConfig{TargetWords: 200}})
	docs := []Document{
		{ID: "exact", Text: "the rare token zylinder appears here among other words"},
		{ID: "near", Text: "many other words and tokens and terms and things here"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	// Graph off both times; vary only the lexical weight.
	off, err := s.Retrieve(ctx, "zylinder", RetrieveParams{TopK: 2, GraphMix: -1, LexicalWeight: -1})
	if err != nil {
		t.Fatal(err)
	}
	on, err := s.Retrieve(ctx, "zylinder", RetrieveParams{TopK: 2, GraphMix: -1, LexicalWeight: 1})
	if err != nil {
		t.Fatal(err)
	}
	// With lexical weight on, the exact-term doc must be first.
	if on[0].Chunk.DocID != "exact" {
		t.Errorf("lexical weight failed to surface the exact-term doc: %v", docids(on))
	}
	// The lexical term must matter: turning it off should change the ranking (the
	// dense-only order need not put the exact-term doc first).
	if len(off) > 0 && len(on) > 0 && off[0].Chunk.DocID == on[0].Chunk.DocID &&
		off[0].Score == on[0].Score {
		t.Error("lexical weight had no effect on the score")
	}
}
