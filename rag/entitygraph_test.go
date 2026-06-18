package rag

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/entity"
)

// capExtractor is a deterministic stand-in for an LLM extractor: capitalized words
// are entities and consecutive capitalized words in a chunk are related.
type capExtractor struct{}

func (capExtractor) Extract(_ context.Context, text string) (entity.Extraction, error) {
	var ex entity.Extraction
	var caps []string
	for _, w := range strings.Fields(text) {
		w = strings.Trim(w, ".,;:!?")
		if w != "" && w[0] >= 'A' && w[0] <= 'Z' {
			caps = append(caps, w)
			ex.Entities = append(ex.Entities, entity.ExtractedEntity{Name: w, Type: "thing"})
		}
	}
	for i := 0; i+1 < len(caps); i++ {
		ex.Relations = append(ex.Relations, entity.ExtractedRelation{Source: caps[i], Target: caps[i+1]})
	}
	return ex, nil
}

func entityStore(t *testing.T) *Store {
	t.Helper()
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
		Chunk: ChunkConfig{TargetWords: 200}})
	docs := []Document{
		{ID: "d1", Text: "Alice works at Acme building things"},
		{ID: "d2", Text: "Acme builds Rockets for space travel"},
		{ID: "d3", Text: "Bob enjoys Tennis on weekends"},
	}
	if err := st.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	if err := st.BuildEntityGraph(context.Background(), capExtractor{}, EntityBuildOptions{Workers: 2}); err != nil {
		t.Fatal(err)
	}
	return st
}

func TestEntityGraphBuilds(t *testing.T) {
	st := entityStore(t)
	if !st.HasEntityGraph() {
		t.Fatal("entity graph not built")
	}
	v := st.EntityGraphView()
	if len(v.Nodes) < 5 { // Alice, Acme, Rockets, Bob, Tennis
		t.Errorf("expected several entity nodes, got %d", len(v.Nodes))
	}
	// Acme should connect to both Alice and Rockets.
	if len(v.Edges) < 3 {
		t.Errorf("expected entity relationships, got %d edges", len(v.Edges))
	}
}

// TestEntityRetrievalConnectsViaSharedEntity is the point of an entity graph: a
// chunk that does not contain the query term is still retrieved because it shares
// an entity with one that does.
func TestEntityRetrievalConnectsViaSharedEntity(t *testing.T) {
	st := entityStore(t)
	ctx := context.Background()
	// "Rockets" only appears in d2. Querying "Alice" (in d1) should, through the
	// Alice -> Acme -> Rockets path in the entity graph, surface d2.
	res, err := st.Retrieve(ctx, "Alice", RetrieveParams{TopK: 3, GraphMix: 0.2, EntityMix: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, r := range res {
		got[r.Chunk.DocID] = true
	}
	if !got["d2"] {
		t.Errorf("entity graph did not connect Alice to the Acme/Rockets chunk: %v", docIDs(res))
	}
	// The unrelated tennis chunk should not be pulled in by the entity signal.
	if got["d3"] {
		t.Errorf("unrelated chunk d3 surfaced via entity retrieval: %v", docIDs(res))
	}
}

func TestEntityGraphSurvivesReload(t *testing.T) {
	st := entityStore(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "s.tg")
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	if !st2.HasEntityGraph() {
		t.Fatal("entity graph not restored after reload")
	}
	if len(st2.EntityGraphView().Nodes) != len(st.EntityGraphView().Nodes) {
		t.Error("entity node count changed across reload")
	}
}

func docIDs(res []Retrieved) []string {
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.Chunk.DocID
	}
	return out
}
