package entity

import "testing"

func TestCanonicalizeMergesVariants(t *testing.T) {
	g := NewGraph()
	// "Charles Babbage" and "Babbage" are the same person; "analytical engine"
	// is related to both. After canonicalization they must be one node.
	g.Add("c1", Extraction{
		Entities:  []ExtractedEntity{{Name: "Charles Babbage", Type: "Person", Description: "inventor"}},
		Relations: []ExtractedRelation{{Source: "Charles Babbage", Target: "Analytical Engine", Description: "designed it"}},
	})
	g.Add("c2", Extraction{
		Entities:  []ExtractedEntity{{Name: "Babbage", Type: "Person"}},
		Relations: []ExtractedRelation{{Source: "Babbage", Target: "Analytical Engine"}},
	})
	g.Add("c3", Extraction{
		Entities: []ExtractedEntity{{Name: "Ada Lovelace"}, {Name: "Lovelace"}},
	})
	g.Canonicalize()

	ents := g.Entities()
	byName := map[string]Entity{}
	for _, e := range ents {
		byName[e.Name] = e
	}
	if _, ok := byName["babbage"]; ok {
		t.Errorf("'babbage' was not merged into 'charles babbage': %v", names(ents))
	}
	cb, ok := byName["charles babbage"]
	if !ok {
		t.Fatalf("canonical 'charles babbage' missing: %v", names(ents))
	}
	// Merged node carries both chunks and the summed mentions.
	if len(cb.Chunks) < 2 {
		t.Errorf("merged chunks = %v, want >=2", cb.Chunks)
	}
	if _, ok := byName["lovelace"]; ok {
		t.Errorf("'lovelace' not merged into 'ada lovelace': %v", names(ents))
	}
	// The two Babbage->Analytical Engine relations collapse into one with summed weight.
	rels := g.Relations()
	var found *Relation
	for i := range rels {
		if (rels[i].Source == "charles babbage" || rels[i].Target == "charles babbage") &&
			(rels[i].Source == "analytical engine" || rels[i].Target == "analytical engine") {
			found = &rels[i]
		}
	}
	if found == nil {
		t.Fatalf("merged relation missing: %+v", rels)
	}
	if found.Weight < 2 {
		t.Errorf("relation weight = %v, want >=2 (two mentions summed)", found.Weight)
	}
}

func TestPruneDropsGhostsAndGenerics(t *testing.T) {
	g := NewGraph()
	g.Add("c1", Extraction{
		Entities: []ExtractedEntity{{Name: "Real Entity", Type: "Concept", Description: "a thing"}},
		// "it" is generic; "Ghost" appears only as a relation endpoint, no type/desc.
		Relations: []ExtractedRelation{{Source: "it", Target: "Ghost"}},
	})
	g.Canonicalize()
	g.Prune()
	for _, e := range g.Entities() {
		if e.Name == "it" || e.Name == "ghost" {
			t.Errorf("prune kept junk node %q", e.Name)
		}
	}
	if g.Len() == 0 {
		t.Error("prune removed the real entity")
	}
}

func names(es []Entity) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.Name
	}
	return out
}

func TestEditRatio(t *testing.T) {
	if editRatio("babbage", "babbages") < 0.85 {
		t.Error("plural variant should be highly similar")
	}
	if editRatio("cat", "dog") > 0.5 {
		t.Error("unrelated words should be dissimilar")
	}
}

// TestPruneKeepsRelationEndpoints pins the bug that emptied the entity graph. An
// extractor routinely names an entity inside a relationship without repeating it in
// the entities list, so that endpoint arrives with no type, no description and no
// mention count. Treating it as a ghost deleted the entity AND, with it, every
// relationship touching it, so the graph rendered as disconnected dots. An entity
// that anchors a fact is load-bearing and must survive.
func TestPruneKeepsRelationEndpoints(t *testing.T) {
	g := NewGraph()
	g.Add("c1", Extraction{
		Entities: []ExtractedEntity{
			{Name: "Project Helios", Type: "project", Description: "a fusion effort"},
		},
		Relations: []ExtractedRelation{
			// "Caldera reactor" is never listed in Entities; only the relation names it.
			{Source: "Project Helios", Target: "Caldera reactor", Description: "relies on the reactor"},
		},
	})
	g.Canonicalize()
	g.Prune()

	if got := len(g.Relations()); got != 1 {
		t.Fatalf("the relationship was destroyed by pruning: %d relations survive", got)
	}
	names := map[string]bool{}
	for _, e := range g.Entities() {
		names[e.Name] = true
	}
	for _, want := range []string{"project helios", "caldera reactor"} {
		if !names[want] {
			t.Errorf("endpoint %q was pruned away, which dangles its relationship (have %v)", want, names)
		}
	}
}

// TestPruneStillDropsIsolatedGhosts: the protection is for endpoints of real facts,
// not a licence to keep every stray noun the extractor emitted.
func TestPruneStillDropsIsolatedGhosts(t *testing.T) {
	g := NewGraph()
	g.Add("c1", Extraction{
		Entities: []ExtractedEntity{
			{Name: "Verdant Labs", Type: "organization", Description: "a research institute"},
			{Name: "stray"}, // no type, no description, no relationship: a genuine ghost
		},
	})
	g.Canonicalize()
	g.Prune()
	for _, e := range g.Entities() {
		if e.Name == "stray" {
			t.Error("an isolated, typeless, undescribed entity should still be pruned")
		}
	}
}
