package graph

import (
	"reflect"
	"sort"
	"testing"
)

// clique adds an all-pairs edge among the given nodes with weight w.
func clique(b *Builder, w float32, nodes ...int) {
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			b.AddEdge(nodes[i], nodes[j], w)
		}
	}
}

func TestTwoCliquesWeakBridge(t *testing.T) {
	// Two 4-cliques joined by a single weak edge should split into two
	// communities, with the weak bridge not strong enough to merge them.
	b := NewBuilder(8)
	clique(b, 1.0, 0, 1, 2, 3)
	clique(b, 1.0, 4, 5, 6, 7)
	b.AddEdge(3, 4, 0.01) // weak bridge
	g := b.Build()

	c := DetectCommunities(g, CommunityOpts{Seed: 42})
	if c.NumCommunities() != 2 {
		t.Fatalf("expected 2 communities, got %d", c.NumCommunities())
	}
	// Nodes 0..3 share a community, nodes 4..7 share another, and the two
	// communities differ.
	if c.Label(0) != c.Label(1) || c.Label(0) != c.Label(2) || c.Label(0) != c.Label(3) {
		t.Errorf("first clique not unified: %v", labelsOf(c, 0, 1, 2, 3))
	}
	if c.Label(4) != c.Label(5) || c.Label(4) != c.Label(6) || c.Label(4) != c.Label(7) {
		t.Errorf("second clique not unified: %v", labelsOf(c, 4, 5, 6, 7))
	}
	if c.Label(0) == c.Label(4) {
		t.Errorf("cliques should be in different communities")
	}
}

func TestSingleClique(t *testing.T) {
	b := NewBuilder(5)
	clique(b, 1.0, 0, 1, 2, 3, 4)
	g := b.Build()

	c := DetectCommunities(g, CommunityOpts{Seed: 7})
	if c.NumCommunities() != 1 {
		t.Fatalf("expected 1 community, got %d", c.NumCommunities())
	}
	for i := 0; i < g.N(); i++ {
		if c.Label(i) != 0 {
			t.Errorf("node %d label = %d, want 0", i, c.Label(i))
		}
	}
}

func TestModularityGoodVsTrivial(t *testing.T) {
	b := NewBuilder(8)
	clique(b, 1.0, 0, 1, 2, 3)
	clique(b, 1.0, 4, 5, 6, 7)
	b.AddEdge(3, 4, 0.01)
	g := b.Build()

	good := DetectCommunities(g, CommunityOpts{Seed: 1})
	q := good.Modularity(g)
	if q <= 0.3 {
		t.Errorf("good partition modularity = %.4f, want > 0.3", q)
	}

	// A single community containing everything should have modularity ~0.
	all := &Communities{labels: make([]int, g.N()), members: [][]int{nil}, num: 1}
	for i := range all.labels {
		all.members[0] = append(all.members[0], i)
	}
	qAll := all.Modularity(g)
	if qAll < -1e-9 || qAll > 1e-9 {
		t.Errorf("all-in-one modularity = %.6f, want ~0", qAll)
	}
}

func TestDeterminism(t *testing.T) {
	b := NewBuilder(12)
	clique(b, 1.0, 0, 1, 2, 3)
	clique(b, 1.0, 4, 5, 6, 7)
	clique(b, 1.0, 8, 9, 10, 11)
	b.AddEdge(3, 4, 0.05)
	b.AddEdge(7, 8, 0.05)
	g := b.Build()

	first := DetectCommunities(g, CommunityOpts{Seed: 99})
	for run := 0; run < 5; run++ {
		c := DetectCommunities(g, CommunityOpts{Seed: 99})
		if !reflect.DeepEqual(first.labels, c.labels) {
			t.Fatalf("run %d labels differ: %v vs %v", run, first.labels, c.labels)
		}
	}

	// A different seed is allowed to differ, but every run with that seed must
	// itself be stable.
	other := DetectCommunities(g, CommunityOpts{Seed: 12345})
	again := DetectCommunities(g, CommunityOpts{Seed: 12345})
	if !reflect.DeepEqual(other.labels, again.labels) {
		t.Fatalf("seed 12345 not stable: %v vs %v", other.labels, again.labels)
	}
}

func TestPartitionCoversAllNodes(t *testing.T) {
	b := NewBuilder(10)
	clique(b, 1.0, 0, 1, 2)
	clique(b, 1.0, 3, 4, 5)
	clique(b, 1.0, 6, 7, 8, 9)
	g := b.Build()

	c := DetectCommunities(g, CommunityOpts{Seed: 3})

	// Every node has exactly one valid label.
	for i := 0; i < g.N(); i++ {
		lab := c.Label(i)
		if lab < 0 || lab >= c.NumCommunities() {
			t.Errorf("node %d has invalid label %d (num=%d)", i, lab, c.NumCommunities())
		}
	}

	// Members must partition the node set: union equals all nodes, no overlaps,
	// and each member's stored label matches its community.
	seen := make([]bool, g.N())
	total := 0
	for cm := 0; cm < c.NumCommunities(); cm++ {
		members := c.Members(cm)
		if !sort.IntsAreSorted(members) {
			t.Errorf("community %d members not sorted: %v", cm, members)
		}
		for _, node := range members {
			if seen[node] {
				t.Errorf("node %d appears in more than one community", node)
			}
			seen[node] = true
			total++
			if c.Label(node) != cm {
				t.Errorf("node %d in members[%d] but Label=%d", node, cm, c.Label(node))
			}
		}
	}
	if total != g.N() {
		t.Errorf("members cover %d nodes, want %d", total, g.N())
	}
	for i, ok := range seen {
		if !ok {
			t.Errorf("node %d missing from all communities", i)
		}
	}
}

func TestDisconnectedNodesAreSingletons(t *testing.T) {
	// Five nodes, no edges at all. Each must be its own community.
	b := NewBuilder(5)
	g := b.Build()

	c := DetectCommunities(g, CommunityOpts{Seed: 2})
	if c.NumCommunities() != 5 {
		t.Fatalf("expected 5 singleton communities, got %d", c.NumCommunities())
	}
	for cm := 0; cm < c.NumCommunities(); cm++ {
		if len(c.Members(cm)) != 1 {
			t.Errorf("community %d size = %d, want 1", cm, len(c.Members(cm)))
		}
	}

	// Isolated node mixed into a connected graph still forms its own community.
	b2 := NewBuilder(4)
	clique(b2, 1.0, 0, 1, 2)
	// node 3 left disconnected
	g2 := b2.Build()
	c2 := DetectCommunities(g2, CommunityOpts{Seed: 2})
	if c2.NumCommunities() != 2 {
		t.Fatalf("expected 2 communities (clique + isolated), got %d", c2.NumCommunities())
	}
	if c2.Label(3) == c2.Label(0) {
		t.Errorf("isolated node 3 should not join the clique")
	}
}

func TestEmptyGraph(t *testing.T) {
	g := NewBuilder(0).Build()
	c := DetectCommunities(g, CommunityOpts{Seed: 1})
	if c.NumCommunities() != 0 {
		t.Errorf("empty graph communities = %d, want 0", c.NumCommunities())
	}
	if got := c.Label(0); got != -1 {
		t.Errorf("Label out of range = %d, want -1", got)
	}
	if c.Members(0) != nil {
		t.Errorf("Members out of range should be nil")
	}
	if q := c.Modularity(g); q != 0 {
		t.Errorf("empty modularity = %v, want 0", q)
	}
}

func TestDefaultMaxIter(t *testing.T) {
	// MaxIter <= 0 should fall back to the default and still converge.
	b := NewBuilder(6)
	clique(b, 1.0, 0, 1, 2)
	clique(b, 1.0, 3, 4, 5)
	g := b.Build()

	c := DetectCommunities(g, CommunityOpts{Seed: 5, MaxIter: 0})
	if c.NumCommunities() != 2 {
		t.Errorf("expected 2 communities with default MaxIter, got %d", c.NumCommunities())
	}
}

func labelsOf(c *Communities, nodes ...int) []int {
	out := make([]int, len(nodes))
	for i, n := range nodes {
		out[i] = c.Label(n)
	}
	return out
}
