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
	// Alice -> Acme -> Rockets path in the entity graph, surface d2. The similarity
	// graph is disabled (GraphMix negative) so this isolates the entity signal: the
	// only way d2 can appear is the shared entity, and an unrelated chunk receives
	// no propagated mass to leak across the threshold.
	res, err := st.Retrieve(ctx, "Alice", RetrieveParams{TopK: 3, GraphMix: -1, EntityMix: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	rank := map[string]int{}
	for i, r := range res {
		rank[r.Chunk.DocID] = i
	}
	d2, ok2 := rank["d2"]
	if !ok2 {
		t.Errorf("entity graph did not connect Alice to the Acme/Rockets chunk: %v", docIDs(res))
	}
	// The entity signal must rank the connected chunk (d2, shared entity) above the
	// unrelated tennis chunk (d3, no shared entity). On this 3-doc corpus every doc
	// is a seed, so the test is about ordering, not membership: the shared entity is
	// the only reason d2 outranks d3.
	if d3, ok3 := rank["d3"]; ok3 && d2 > d3 {
		t.Errorf("entity signal failed to rank the connected chunk d2 above unrelated d3: %v", docIDs(res))
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

func TestEntityDenseSeeding(t *testing.T) {
	ctx := context.Background()
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
		Chunk: ChunkConfig{TargetWords: 30}})
	docs := []Document{
		{ID: "a", Text: "Ada Lovelace wrote the first algorithm for the Analytical Engine designed by Charles Babbage."},
		{ID: "b", Text: "Charles Babbage invented the Difference Engine, a mechanical calculator."},
	}
	if err := st.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if err := st.BuildEntityGraph(ctx, capExtractor{}, EntityBuildOptions{Workers: 2}); err != nil {
		t.Fatal(err)
	}
	// Entity embeddings are populated and counted with the entity list.
	st.mu.RLock()
	gotVec, gotEnt := len(st.entVec), len(st.entList)
	st.mu.RUnlock()
	if gotEnt == 0 {
		t.Fatal("no entities extracted")
	}
	if gotVec != gotEnt {
		t.Fatalf("entVec=%d, want %d (one per entity)", gotVec, gotEnt)
	}

	// Dense seeding produces seeds for a query that shares vocabulary with an
	// entity, and falls back to lexical when embeddings are absent.
	qv, err := embedQuery(ctx, st.embedder, []string{"Babbage Analytical Engine"})
	if err != nil {
		t.Fatal(err)
	}
	st.mu.RLock()
	dense := st.entitySeeds("Babbage Analytical Engine", qv[0])
	saved := st.entVec
	st.entVec = nil
	lexOnly := st.entitySeeds("Babbage Analytical Engine", qv[0])
	st.entVec = saved
	st.mu.RUnlock()
	if len(dense) == 0 {
		t.Fatal("dense seeding produced no seeds")
	}
	if len(lexOnly) == 0 {
		t.Fatal("lexical fallback produced no seeds")
	}

	// Embeddings survive a save/load round trip.
	path := t.TempDir() + "/e.tg"
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	st2.mu.RLock()
	v2 := len(st2.entVec)
	st2.mu.RUnlock()
	if v2 != gotEnt {
		t.Fatalf("entVec after reload=%d, want %d", v2, gotEnt)
	}
}
