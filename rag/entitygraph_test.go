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

// TestRelationContext checks that knowledge-graph triplets are surfaced only for
// the retrieved chunks: facts whose endpoints are both present are returned, and
// facts about entities outside the retrieved set are excluded.
func TestRelationContext(t *testing.T) {
	st := entityStore(t)
	ctx := context.Background()
	// Retrieve enough to pull in d1 (Alice, Acme) and d2 (Acme, Rockets) but not
	// d3 (Bob, Tennis). "Acme" sits in both d1 and d2.
	res, err := st.Retrieve(ctx, "Acme Alice Rockets", RetrieveParams{TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(res))
	for i, r := range res {
		ids[i] = r.Chunk.ID
	}
	facts := st.RelationContext(ids, 12)
	if len(facts) == 0 {
		t.Fatal("expected grounded relationship facts, got none")
	}
	joined := strings.ToLower(strings.Join(facts, "\n"))
	if !strings.Contains(joined, "acme") {
		t.Fatalf("expected an Acme relationship in the facts, got %v", facts)
	}
	// Bob/Tennis live only in d3, which was not retrieved, so no fact may mention
	// them.
	if strings.Contains(joined, "bob") || strings.Contains(joined, "tennis") {
		t.Fatalf("facts leaked an entity from an unretrieved chunk: %v", facts)
	}
	// No entity graph -> no facts, and the cap is honoured.
	if got := st.RelationContext(ids, 0); got != nil {
		t.Errorf("maxRels=0 should return nil, got %v", got)
	}
}

// descExtractor relates capitalized words and describes each edge differently per
// chunk, to prove that every distinct description of a relationship reaches the
// prompt, not just the first one seen.
type descExtractor struct {
	byChunk map[string]string
}

func (e descExtractor) Extract(ctx context.Context, text string) (entity.Extraction, error) {
	var ex entity.Extraction
	var caps []string
	for _, w := range strings.Fields(text) {
		w = strings.Trim(w, ".,;:!?")
		if w != "" && w[0] >= 'A' && w[0] <= 'Z' {
			caps = append(caps, w)
			ex.Entities = append(ex.Entities, entity.ExtractedEntity{Name: w, Type: "thing"})
		}
	}
	desc := e.byChunk[text]
	for i := 0; i+1 < len(caps); i++ {
		ex.Relations = append(ex.Relations, entity.ExtractedRelation{Source: caps[i], Target: caps[i+1], Description: desc})
	}
	return ex, nil
}

func TestRelationContextShowsDistinctDescriptions(t *testing.T) {
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
		Chunk: ChunkConfig{TargetWords: 200}})
	docs := []Document{
		{ID: "d1", Text: "Alice works at Acme"},
		{ID: "d2", Text: "Alice and Acme are partners"},
	}
	ex := descExtractor{byChunk: map[string]string{
		"Alice works at Acme":         "the employer of",
		"Alice and Acme are partners": "co-signer on the deal",
	}}
	if err := st.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	if err := st.BuildEntityGraph(context.Background(), ex, EntityBuildOptions{Workers: 1}); err != nil {
		t.Fatal(err)
	}
	res, err := st.Retrieve(context.Background(), "Alice Acme", RetrieveParams{TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, r := range res {
		ids = append(ids, r.Chunk.ID)
	}
	facts := st.RelationContext(ids, 5)
	if len(facts) == 0 {
		t.Fatal("expected grounded fact lines, got none")
	}
	joined := strings.ToLower(strings.Join(facts, "\n"))
	if !strings.Contains(joined, "the employer of") || !strings.Contains(joined, "co-signer on the deal") {
		t.Fatalf("expected both distinct descriptions of the edge in the facts, got %v", facts)
	}
}

// TestEntityBuildPrefilterSkipsNoisyChunks: a chunk that is mostly not letters, or
// that has too few tokens to hold an entity, must never reach the model, must not
// appear in the graph, and must be reported as skipped_empty in the progress.
func TestEntityBuildPrefilterSkipsNoisyChunks(t *testing.T) {
	st := New(newKeywordEmbedder(32), Config{Seed: 1})
	docs := []Document{
		{ID: "d1", Text: "Acme builds Rockets near the Moon."},
		{ID: "d2", Text: "1f8b0800 4c53553 00ff4473 c3a9d9aa 913b3747 9ff05490 8a6a2"}, // hex dump-ish
		{ID: "d3", Text: "RTÉ"},
	}
	if err := st.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	ex := &countingExtractor{}
	var skipped, total int
	opt := EntityBuildOptions{
		Workers:            1,
		PrefilterRatio:     0.3,
		PrefilterMinTokens: 3,
		OnProgress: func(p EntityProgress) {
			skipped, total = p.SkippedEmpty, p.Total
		},
	}
	if err := st.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if total != 3 {
		t.Fatalf("total progress wrong: %d, want 3", total)
	}
	if skipped != 2 {
		t.Errorf("progress reported %d skipped chunks, want 2 (d2 low letter ratio, d3 too few tokens)", skipped)
	}
	// d2 has letters (the hex chars) but a letter ratio below 0.3; d3 has 2 tokens.
	if got := ex.calls.Load(); got != 1 {
		t.Errorf("prefilter still let %d chunks through to the model, want 1 (d1 only)", got)
	}
	// The noise must not pollute the graph: only d1's entities should be present.
	if st.EntityCount() != 2 { // E-Acme… and Hub from countingExtractor
		t.Errorf("entity count %d, want 2 (only the clean chunk extracted)", st.EntityCount())
	}
	// With the feature off (zero thresholds) nothing is ever skipped: the chunks the
	// prefilter had left unextracted (d2, d3) now go to the model; d1 is already cached.
	ex2 := &countingExtractor{}
	if err := st.BuildEntityGraph(context.Background(), ex2, EntityBuildOptions{Workers: 1}); err != nil {
		t.Fatal(err)
	}
	if got := ex2.calls.Load(); got != 2 {
		t.Errorf("with prefilter off %d chunks reached the model, want 2 (d2, d3; d1 cached)", got)
	}
}
