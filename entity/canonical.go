package entity

import (
	"slices"
	"sort"
	"strings"
)

// genericNames are surface forms too generic to be a useful entity or to drive a
// short-name merge. They show up when a small model emits a pronoun or a filler
// word as an entity.
var genericNames = map[string]bool{
	"the": true, "a": true, "an": true, "it": true, "he": true, "she": true,
	"they": true, "we": true, "this": true, "that": true, "these": true,
	"those": true, "thing": true, "things": true, "entity": true, "system": true,
	"concept": true, "item": true, "object": true, "one": true, "some": true,
}

// Canonicalize merges entities that are surface variants of the same thing into a
// single node, then rewrites every relation endpoint through the merge so the
// graph stays connected. Without it, "Ada Lovelace" and "Lovelace" remain two
// disconnected PageRank nodes with split chunk lists and split edge weight;
// merging concentrates relevance mass and reconnects the subgraph, which directly
// improves entity-graph retrieval. Variants are detected by a high edit-distance
// ratio (typos, punctuation, plurals) and by short-name or last-name references
// (a single salient token of a longer name). The longest, most-mentioned surface
// form becomes the canonical. Adapted from cognee's name_mapping endpoint rewrite.
func (g *Graph) Canonicalize() {
	if len(g.entities) < 2 {
		return
	}
	// Order so the most complete surface form is chosen as a cluster head first:
	// longest name, then most mentions, then name for determinism.
	names := make([]string, 0, len(g.entities))
	for n := range g.entities {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		a, b := g.entities[names[i]], g.entities[names[j]]
		if la, lb := len(a.Name), len(b.Name); la != lb {
			return la > lb
		}
		if a.Mentions != b.Mentions {
			return a.Mentions > b.Mentions
		}
		return a.Name < b.Name
	})

	parent := make(map[string]string, len(names)) // variant name -> canonical name
	var heads []string
	for _, n := range names {
		canon := ""
		for _, h := range heads {
			if mergeableNames(n, h) {
				canon = h
				break
			}
		}
		if canon == "" {
			heads = append(heads, n)
			parent[n] = n
		} else {
			parent[n] = canon
		}
	}

	// Merge entities into their canonical head.
	merged := make(map[string]*Entity, len(heads))
	descSeen := make(map[string]map[string]struct{}, len(heads))
	for _, n := range names { // names is length-desc, so heads merge before variants
		e := g.entities[n]
		c := parent[n]
		m, ok := merged[c]
		if !ok {
			head := *g.entities[c]
			m = &head
			merged[c] = m
			descSeen[c] = map[string]struct{}{}
		}
		if m != e { // fold a variant in
			m.Mentions += e.Mentions
			if m.Type == "" {
				m.Type = e.Type
			}
		}
		for _, ch := range e.Chunks {
			m.Chunks = appendUnique(m.Chunks, ch)
		}
		for _, d := range splitDescs(e.Description) {
			if _, dup := descSeen[c][d]; !dup {
				descSeen[c][d] = struct{}{}
			}
		}
	}
	// Rebuild merged descriptions in a stable order.
	for c, m := range merged {
		ds := make([]string, 0, len(descSeen[c]))
		for d := range descSeen[c] {
			ds = append(ds, d)
		}
		sort.Strings(ds)
		m.Description = strings.Join(ds, " ")
	}

	// Rewrite relation endpoints through the merge map and re-key, summing weights.
	newRel := make(map[[2]string]*Relation, len(g.relations))
	for _, r := range g.relations {
		s, t := parent[r.Source], parent[r.Target]
		if s == "" {
			s = r.Source
		}
		if t == "" {
			t = r.Target
		}
		if s == t {
			continue // a relation that collapsed onto itself after merging
		}
		key := [2]string{s, t}
		if key[0] > key[1] {
			key[0], key[1] = key[1], key[0]
		}
		nr, ok := newRel[key]
		if !ok {
			nr = &Relation{Source: key[0], Target: key[1]}
			newRel[key] = nr
		}
		nr.Weight += r.Weight
		if nr.Description == "" {
			nr.Description = r.Description
		}
	}

	g.entities = merged
	g.relations = newRel
	g.descSeen = descSeen
}

// Prune drops low-value nodes: generic or empty names, and "ghost" endpoints that
// were never extracted as entities (no type, no description, a single mention)
// and so are usually malformed model output. Their relations are dropped with
// them. It runs after Canonicalize.
func (g *Graph) Prune() {
	drop := make(map[string]bool)
	for name, e := range g.entities {
		if name == "" || genericNames[name] {
			drop[name] = true
			continue
		}
		ghost := e.Type == "" && e.Description == "" && e.Mentions <= 1
		if ghost {
			drop[name] = true
		}
	}
	if len(drop) == 0 {
		return
	}
	for name := range drop {
		delete(g.entities, name)
		delete(g.descSeen, name)
	}
	for key := range g.relations {
		if drop[key[0]] || drop[key[1]] {
			delete(g.relations, key)
		}
	}
}

// mergeableNames reports whether name a is a surface variant of name b. Both are
// already normalized (lowercase, trimmed).
func mergeableNames(a, b string) bool {
	if a == b || a == "" || b == "" {
		return false
	}
	// Typos, punctuation, plurals: a high character-level similarity.
	if editRatio(a, b) >= 0.9 {
		return true
	}
	// Short-name or last-name reference: every token of the shorter name is a
	// salient (non-generic, length >= 4) token of the longer multi-token name.
	short, long := a, b
	if len(a) > len(b) {
		short, long = b, a
	}
	st, lt := strings.Fields(short), strings.Fields(long)
	if len(lt) < 2 || len(st) == 0 || len(st) >= len(lt) {
		return false
	}
	lset := make(map[string]bool, len(lt))
	for _, tk := range lt {
		lset[tk] = true
	}
	for _, tk := range st {
		if len(tk) < 4 || genericNames[tk] || !lset[tk] {
			return false
		}
	}
	return true
}

// editRatio is 1 - levenshtein(a,b)/max(len(a),len(b)), a similarity in [0,1].
func editRatio(a, b string) float64 {
	if a == b {
		return 1
	}
	d := levenshtein(a, b)
	m := max(len(a), len(b))
	if m == 0 {
		return 1
	}
	return 1 - float64(d)/float64(m)
}

// levenshtein is the standard two-row edit distance over bytes (entity names are
// ASCII-dominant; bytes are an adequate and fast approximation).
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min3(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

func appendUnique(s []string, v string) []string {
	if slices.Contains(s, v) {
		return s
	}
	return append(s, v)
}

// splitDescs splits a merged description back into its space-joined sentences for
// re-deduplication. Descriptions are joined with single spaces, so this is a
// best-effort split that keeps whole non-empty fragments.
func splitDescs(d string) []string {
	d = strings.TrimSpace(d)
	if d == "" {
		return nil
	}
	return []string{d}
}
