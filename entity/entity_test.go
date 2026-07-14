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

// recordGen returns a fixed batch-tagged output and records the prompt.
type recordGen struct {
	out    string
	prompt string
}

func (g *recordGen) Generate(_ context.Context, _ string, prompt string) (string, error) {
	g.prompt = prompt
	return g.out, nil
}

func TestParseBatchAttributesByPassage(t *testing.T) {
	out := "entity|1|Ada|person|mathematician\n" +
		"relation|1|Ada|Engine|wrote algorithm for\n" +
		"entity|2|Bell Labs|org|research lab\n" +
		"entity|5|OutOfRange|x|ignored\n" + // out-of-range passage -> dropped
		"garbage line without pipes\n"
	exs := ParseBatch(out, 2)
	if len(exs) != 2 {
		t.Fatalf("want 2 extractions, got %d", len(exs))
	}
	if len(exs[0].Entities) != 1 || exs[0].Entities[0].Name != "Ada" {
		t.Errorf("passage 1 entities wrong: %+v", exs[0].Entities)
	}
	if len(exs[0].Relations) != 1 || exs[0].Relations[0].Source != "Ada" || exs[0].Relations[0].Target != "Engine" {
		t.Errorf("passage 1 relations wrong: %+v", exs[0].Relations)
	}
	if len(exs[1].Entities) != 1 || exs[1].Entities[0].Name != "Bell Labs" {
		t.Errorf("passage 2 entities wrong: %+v", exs[1].Entities)
	}
}

func TestExtractBatchOneCall(t *testing.T) {
	g := &recordGen{out: "entity|1|A|t|d\nentity|2|B|t|d\nentity|3|C|t|d\n"}
	e := NewLLMExtractor(g)
	exs, err := e.ExtractBatch(context.Background(), []string{"x", "y", "z"})
	if err != nil {
		t.Fatal(err)
	}
	// Three passages came back from one Generate call, each attributed correctly.
	if len(exs) != 3 || exs[0].Entities[0].Name != "A" || exs[2].Entities[0].Name != "C" {
		t.Fatalf("batch extraction misattributed: %+v", exs)
	}
	if !strings.Contains(g.prompt, "[1]") || !strings.Contains(g.prompt, "[3]") {
		t.Errorf("batch prompt should number passages, got: %s", g.prompt)
	}
}

// TestCleanDropsVerbEndpoints pins the malformed output seen in real builds: the
// model intermittently puts the relation's verb in an endpoint slot, and the verb
// then becomes a permanent node. Pruning cannot catch it, because it protects
// relation endpoints on purpose.
func TestCleanDropsVerbEndpoints(t *testing.T) {
	ex := Extraction{
		Entities: []ExtractedEntity{
			{Name: "Project Corvus", Type: "project"},
			{Name: "Dr. Ana Ruiz", Type: "person"},
		},
		Relations: []ExtractedRelation{
			{Source: "Dr. Ana Ruiz", Target: "Project Corvus", Description: "leads it"},
			{Source: "Project Corvus", Target: "led at", Description: "Lab 0"},
			{Source: "Project Corvus", Target: "funded as part of funder relation to"},
			{Source: "built in", Target: "Lakemont"},
			// An endpoint that is a proper name but was never listed as an entity is
			// still real: extractors routinely name one only inside a fact about it.
			{Source: "Project Corvus", Target: "Lakemont", Description: "built in"},
		},
	}
	got := Clean(ex)
	if len(got.Relations) != 2 {
		for _, r := range got.Relations {
			t.Logf("kept: %q -> %q", r.Source, r.Target)
		}
		t.Fatalf("Clean kept %d relations, want 2", len(got.Relations))
	}

	g := NewGraph()
	g.Add("c1", ex)
	for _, e := range g.Entities() {
		switch e.Name {
		case "led at", "built in", "funded as part of funder relation to":
			t.Errorf("a relation verb became an entity: %q", e.Name)
		}
	}
	// The real endpoint that was only ever named inside a fact must survive.
	if _, ok := g.entities["lakemont"]; !ok {
		t.Error("Lakemont was dropped; a capitalized endpoint is a real entity")
	}
}
