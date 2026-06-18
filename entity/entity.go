// Package entity builds a knowledge graph of typed entities and relationships
// extracted from text, the structure used by GraphRAG-style systems. It is the
// richer alternative to a chunk-similarity graph: nodes are entities (people,
// organizations, concepts) and edges are relationships between them, so two
// passages can be connected because they mention the same thing rather than
// because they read alike.
//
// Extraction is pluggable through the Extractor interface; an LLM-backed
// implementation lives in llm.go. The package has no dependency on the rest of
// turbograph so it can be used on its own.
package entity

import (
	"context"
	"sort"
	"strings"
)

// ExtractedEntity is one entity as produced by an Extractor.
type ExtractedEntity struct {
	Name        string
	Type        string
	Description string
}

// ExtractedRelation is one relationship as produced by an Extractor.
type ExtractedRelation struct {
	Source      string
	Target      string
	Description string
}

// Extraction is the result of running an Extractor over a piece of text.
type Extraction struct {
	Entities  []ExtractedEntity
	Relations []ExtractedRelation
}

// Extractor pulls entities and relationships out of a chunk of text.
type Extractor interface {
	Extract(ctx context.Context, text string) (Extraction, error)
}

// Entity is a merged node in the knowledge graph.
type Entity struct {
	Name        string   `json:"name"`        // canonical (lowercased) key
	Display     string   `json:"display"`     // human-facing name
	Type        string   `json:"type"`        // best-known type
	Description string   `json:"description"` // merged descriptions
	Chunks      []string `json:"chunks"`      // chunk ids that mention it
	Mentions    int      `json:"mentions"`
}

// Relation is a merged, undirected edge in the knowledge graph.
type Relation struct {
	Source      string  `json:"source"` // canonical entity name
	Target      string  `json:"target"` // canonical entity name
	Description string  `json:"description"`
	Weight      float32 `json:"weight"` // number of times the relationship was seen
}

// Graph accumulates entities and relationships across many extractions, merging
// by name so the same entity seen in different chunks becomes one node.
type Graph struct {
	entities  map[string]*Entity
	relations map[[2]string]*Relation
	descSeen  map[string]map[string]struct{} // entity -> set of descriptions, to dedupe
}

// NewGraph creates an empty knowledge graph.
func NewGraph() *Graph {
	return &Graph{
		entities:  map[string]*Entity{},
		relations: map[[2]string]*Relation{},
		descSeen:  map[string]map[string]struct{}{},
	}
}

func norm(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

func (g *Graph) ensure(name, display, typ string) *Entity {
	c := norm(name)
	if c == "" {
		return nil
	}
	e, ok := g.entities[c]
	if !ok {
		e = &Entity{Name: c, Display: strings.TrimSpace(display)}
		g.entities[c] = e
		g.descSeen[c] = map[string]struct{}{}
	}
	if e.Display == "" {
		e.Display = strings.TrimSpace(display)
	}
	if e.Type == "" {
		e.Type = strings.TrimSpace(typ)
	}
	return e
}

// Add merges an extraction from the chunk with the given id into the graph.
func (g *Graph) Add(chunkID string, ex Extraction) {
	mentioned := map[string]struct{}{}
	for _, ent := range ex.Entities {
		e := g.ensure(ent.Name, ent.Name, ent.Type)
		if e == nil {
			continue
		}
		e.Mentions++
		if d := strings.TrimSpace(ent.Description); d != "" {
			if _, dup := g.descSeen[e.Name][d]; !dup {
				g.descSeen[e.Name][d] = struct{}{}
				if e.Description != "" {
					e.Description += " "
				}
				e.Description += d
			}
		}
		if _, seen := mentioned[e.Name]; !seen {
			mentioned[e.Name] = struct{}{}
			if chunkID != "" {
				e.Chunks = append(e.Chunks, chunkID)
			}
		}
	}
	for _, rel := range ex.Relations {
		s := g.ensure(rel.Source, rel.Source, "")
		t := g.ensure(rel.Target, rel.Target, "")
		if s == nil || t == nil || s.Name == t.Name {
			continue
		}
		// Relationship endpoints are entities too; record the mention chunk.
		for _, e := range []*Entity{s, t} {
			if _, seen := mentioned[e.Name]; !seen {
				mentioned[e.Name] = struct{}{}
				if chunkID != "" {
					e.Chunks = append(e.Chunks, chunkID)
				}
			}
		}
		key := [2]string{s.Name, t.Name}
		if key[0] > key[1] {
			key[0], key[1] = key[1], key[0]
		}
		r, ok := g.relations[key]
		if !ok {
			r = &Relation{Source: key[0], Target: key[1]}
			g.relations[key] = r
		}
		r.Weight++
		if d := strings.TrimSpace(rel.Description); d != "" && r.Description == "" {
			r.Description = d
		}
	}
}

// Entities returns the merged entities sorted by canonical name.
func (g *Graph) Entities() []Entity {
	out := make([]Entity, 0, len(g.entities))
	for _, e := range g.entities {
		out = append(out, *e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Relations returns the merged relationships sorted by endpoints.
func (g *Graph) Relations() []Relation {
	out := make([]Relation, 0, len(g.relations))
	for _, r := range g.relations {
		out = append(out, *r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Source != out[j].Source {
			return out[i].Source < out[j].Source
		}
		return out[i].Target < out[j].Target
	})
	return out
}

// Len returns the number of entities.
func (g *Graph) Len() int { return len(g.entities) }

// Restore rebuilds a graph from persisted entities and relations, so extraction
// (which is expensive and may use an LLM) does not have to run again on load.
func Restore(entities []Entity, relations []Relation) *Graph {
	g := NewGraph()
	for i := range entities {
		e := entities[i]
		c := norm(e.Name)
		cp := e
		cp.Name = c
		g.entities[c] = &cp
		g.descSeen[c] = map[string]struct{}{}
	}
	for i := range relations {
		r := relations[i]
		key := [2]string{norm(r.Source), norm(r.Target)}
		if key[0] > key[1] {
			key[0], key[1] = key[1], key[0]
		}
		cp := r
		cp.Source, cp.Target = key[0], key[1]
		g.relations[key] = &cp
	}
	return g
}
