package entity

import (
	"context"
	"strings"
	"testing"
)

func TestParseLenient(t *testing.T) {
	out := "```\n" +
		"entity|Ada Lovelace|person|a mathematician\n" +
		"some stray commentary that should be ignored\n" +
		"- relation|Ada Lovelace|Analytical Engine|wrote an algorithm for it\n" +
		"entity|Analytical Engine|product\n" +
		"```"
	ex := Parse(out)
	if len(ex.Entities) != 2 {
		t.Fatalf("entities = %d, want 2: %+v", len(ex.Entities), ex.Entities)
	}
	if len(ex.Relations) != 1 {
		t.Fatalf("relations = %d, want 1", len(ex.Relations))
	}
	if ex.Relations[0].Source != "Ada Lovelace" || ex.Relations[0].Target != "Analytical Engine" {
		t.Errorf("relation endpoints wrong: %+v", ex.Relations[0])
	}
}

func TestGraphMergesAcrossChunks(t *testing.T) {
	g := NewGraph()
	g.Add("c1", Extraction{
		Entities:  []ExtractedEntity{{Name: "Acme", Type: "org", Description: "a company"}},
		Relations: []ExtractedRelation{{Source: "Acme", Target: "Widget", Description: "makes"}},
	})
	g.Add("c2", Extraction{
		Entities: []ExtractedEntity{{Name: "acme", Type: "org", Description: "founded in 1990"}},
	})
	ents := g.Entities()
	// Acme (case-insensitive merge) and Widget (created from the relation).
	if len(ents) != 2 {
		t.Fatalf("entities = %d, want 2: %+v", len(ents), ents)
	}
	var acme *Entity
	for i := range ents {
		if ents[i].Name == "acme" {
			acme = &ents[i]
		}
	}
	if acme == nil {
		t.Fatal("acme entity missing")
	}
	if acme.Mentions != 2 {
		t.Errorf("acme mentions = %d, want 2", acme.Mentions)
	}
	if !strings.Contains(acme.Description, "company") || !strings.Contains(acme.Description, "1990") {
		t.Errorf("descriptions not merged: %q", acme.Description)
	}
	if len(acme.Chunks) != 2 {
		t.Errorf("acme should be mentioned in 2 chunks, got %v", acme.Chunks)
	}
	if len(g.Relations()) != 1 {
		t.Errorf("relations = %d, want 1", len(g.Relations()))
	}
}

func TestGraphUndirectedRelationMerge(t *testing.T) {
	g := NewGraph()
	g.Add("c1", Extraction{Relations: []ExtractedRelation{{Source: "A", Target: "B"}}})
	g.Add("c2", Extraction{Relations: []ExtractedRelation{{Source: "B", Target: "A"}}})
	rels := g.Relations()
	if len(rels) != 1 {
		t.Fatalf("A-B and B-A should merge into one edge, got %d", len(rels))
	}
	if rels[0].Weight != 2 {
		t.Errorf("merged weight = %v, want 2", rels[0].Weight)
	}
}

func TestRestoreRoundTrip(t *testing.T) {
	g := NewGraph()
	g.Add("c1", Extraction{
		Entities:  []ExtractedEntity{{Name: "X", Type: "concept", Description: "d"}},
		Relations: []ExtractedRelation{{Source: "X", Target: "Y"}},
	})
	g2 := Restore(g.Entities(), g.Relations())
	if g2.Len() != g.Len() || len(g2.Relations()) != len(g.Relations()) {
		t.Errorf("restore mismatch: %d/%d vs %d/%d", g2.Len(), len(g2.Relations()), g.Len(), len(g.Relations()))
	}
}

// keywordExtractor is a deterministic stand-in for an LLM: capitalized words are
// entities and words sharing a line are related. It lets the pipeline be tested
// without a model.
type keywordExtractor struct{}

func (keywordExtractor) Extract(_ context.Context, text string) (Extraction, error) {
	var ex Extraction
	var caps []string
	for _, w := range strings.Fields(text) {
		w = strings.Trim(w, ".,;:")
		if w != "" && w[0] >= 'A' && w[0] <= 'Z' {
			caps = append(caps, w)
			ex.Entities = append(ex.Entities, ExtractedEntity{Name: w, Type: "thing"})
		}
	}
	for i := 0; i+1 < len(caps); i++ {
		ex.Relations = append(ex.Relations, ExtractedRelation{Source: caps[i], Target: caps[i+1], Description: "co-occurs"})
	}
	return ex, nil
}

func TestExtractorInterface(t *testing.T) {
	var e Extractor = keywordExtractor{}
	ex, err := e.Extract(context.Background(), "Alice met Bob in Paris")
	if err != nil {
		t.Fatal(err)
	}
	if len(ex.Entities) != 3 {
		t.Errorf("expected 3 entities (Alice, Bob, Paris), got %+v", ex.Entities)
	}
}
