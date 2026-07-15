package rag

import (
	"bytes"
	"context"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/entity"
)

// factExtractor returns a fixed extraction, so the entity graph can be built in a test
// without a model. It ignores the text and emits the same facts for chunk c0.
type factExtractor struct{ ex entity.Extraction }

func (f factExtractor) Extract(context.Context, string) (entity.Extraction, error) {
	return f.ex, nil
}

// TestFactLinkingSeedsFromRelations checks the mechanism end to end on a hand-built
// graph: a query about a relationship should, under fact-linking, seed PageRank from the
// entities that relationship connects, and surface the chunk that mentions them.
func TestFactLinkingSeedsFromRelations(t *testing.T) {
	ctx := context.Background()
	// keywordEmbedder embeds by shared vocabulary, so a query sharing words with a fact
	// string lands near that fact's vector. That is enough to exercise the linking.
	s := New(newKeywordEmbedder(128), Config{Seed: 1, MinSimilarity: 0.0})
	docs := []Document{
		{ID: "d1", Text: "jane doe is the chief executive officer of acme corporation since 2021"},
		{ID: "d2", Text: "unrelated notes about weather patterns over the pacific ocean and rainfall"},
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	ex := factExtractor{ex: entity.Extraction{
		Entities: []entity.ExtractedEntity{
			{Name: "Jane Doe", Type: "person"},
			{Name: "Acme Corporation", Type: "organization"},
		},
		Relations: []entity.ExtractedRelation{
			{Source: "Jane Doe", Target: "Acme Corporation", Description: "is the chief executive officer of"},
		},
	}}
	if err := s.BuildEntityGraph(ctx, ex, EntityBuildOptions{Model: "test"}); err != nil {
		t.Fatal(err)
	}
	if s.RelationCount() == 0 {
		t.Fatal("no relations extracted")
	}

	// Fact-linking must build the fact index and seed from the relation's endpoints.
	s.ensureFactIndex(ctx)
	if len(s.factVec) == 0 {
		t.Fatal("fact index was not built")
	}
	qv, _ := embedQuery(ctx, s.embedder, []string{"who is the chief executive officer of acme"})
	seeds := s.factSeeds(qv[0])
	if len(seeds) == 0 {
		t.Fatal("fact-linking produced no seeds for a query about the relationship")
	}
	// The seeds must be the entities the fact connects.
	want := map[string]bool{"jane doe": false, "acme corporation": false}
	for node := range seeds {
		if node >= 0 && node < len(s.entList) {
			if _, ok := want[s.entList[node].Name]; ok {
				want[s.entList[node].Name] = true
			}
		}
	}
	for name, seeded := range want {
		if !seeded {
			t.Errorf("fact-linking did not seed %q, the entity the matched fact connects", name)
		}
	}
}

// TestFactAndNodeLinkingBothRun: both modes must be selectable on the same store and
// neither may error, since the A/B runs both against one index.
func TestFactAndNodeLinkingBothRun(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(128), Config{Seed: 1, MinSimilarity: 0.0})
	if err := s.Build(ctx, []Document{
		{ID: "d1", Text: "jane doe leads acme corporation as its chief executive"},
		{ID: "d2", Text: "bob smith founded beta industries in the same city"},
	}); err != nil {
		t.Fatal(err)
	}
	ex := factExtractor{ex: entity.Extraction{
		Entities:  []entity.ExtractedEntity{{Name: "Jane Doe", Type: "person"}, {Name: "Acme Corporation", Type: "organization"}},
		Relations: []entity.ExtractedRelation{{Source: "Jane Doe", Target: "Acme Corporation", Description: "leads"}},
	}}
	if err := s.BuildEntityGraph(ctx, ex, EntityBuildOptions{Model: "test"}); err != nil {
		t.Fatal(err)
	}
	// "" must default to fact-linking; all three must run without error.
	for _, link := range []string{"", "node", "fact"} {
		res, err := s.Retrieve(ctx, "who leads acme corporation",
			RetrieveParams{TopK: 2, EntityMix: 0.5, EntityLink: link})
		if err != nil {
			t.Fatalf("link=%q: %v", link, err)
		}
		if len(res) == 0 {
			t.Errorf("link=%q returned nothing", link)
		}
	}
}

// TestFactIndexSurvivesSaveLoad pins issue #17: the fact index is built when the entity
// graph is built and persisted with the store, so the first entity-linked query after a
// restart does not re-embed every relation inside the request.
func TestFactIndexSurvivesSaveLoad(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(64), Config{Seed: 1, MinSimilarity: 0.0})
	if err := s.Build(ctx, []Document{
		{ID: "d1", Text: "jane doe leads acme corporation as chief executive"},
		{ID: "d2", Text: "bob smith founded beta industries in berlin"},
	}); err != nil {
		t.Fatal(err)
	}
	ex := factExtractor{ex: entity.Extraction{
		Entities: []entity.ExtractedEntity{
			{Name: "Jane Doe", Type: "person"}, {Name: "Acme Corporation", Type: "organization"},
			{Name: "Bob Smith", Type: "person"}, {Name: "Beta Industries", Type: "organization"},
		},
		Relations: []entity.ExtractedRelation{
			{Source: "Jane Doe", Target: "Acme Corporation", Description: "leads"},
			{Source: "Bob Smith", Target: "Beta Industries", Description: "founded"},
		},
	}}
	if err := s.BuildEntityGraph(ctx, ex, EntityBuildOptions{Model: "test"}); err != nil {
		t.Fatal(err)
	}
	// The fact index must have been built during BuildEntityGraph, not lazily later.
	if len(s.factVec) == 0 {
		t.Fatal("fact index was not built when the entity graph was built")
	}
	built := len(s.factVec)

	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(newKeywordEmbedder(64), &buf)
	if err != nil {
		t.Fatal(err)
	}
	// The loaded store must already carry the fact index, remapped to its node indices.
	if len(loaded.factVec) != built {
		t.Fatalf("fact index did not survive save/load: %d entries, want %d", len(loaded.factVec), built)
	}
	for i := range loaded.factSrc {
		if loaded.factSrc[i] < 0 || loaded.factSrc[i] >= len(loaded.entList) ||
			loaded.factTgt[i] < 0 || loaded.factTgt[i] >= len(loaded.entList) {
			t.Fatalf("restored fact %d has an out-of-range endpoint", i)
		}
	}
	// A fact-linked query works immediately, without a rebuild.
	res, err := loaded.Retrieve(ctx, "who leads acme corporation",
		RetrieveParams{TopK: 2, EntityMix: 0.5, EntityLink: "fact"})
	if err != nil || len(res) == 0 {
		t.Fatalf("fact-linked query on the loaded store failed: %v %v", res, err)
	}
}
