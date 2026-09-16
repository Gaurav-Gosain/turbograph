package graph

import (
	"reflect"
	"testing"
)

// labelsRange returns the labels of nodes in [from, to).
func labelsRange(c *Communities, from, to int) []int {
	out := make([]int, 0, to-from)
	for i := from; i < to; i++ {
		out = append(out, c.Label(i))
	}
	return out
}

func checkPartition(t *testing.T, c *Communities, n int) {
	t.Helper()
	if c.NodeCount() != n {
		t.Fatalf("NodeCount = %d, want %d", c.NodeCount(), n)
	}
	seen := make([]bool, n)
	for node := 0; node < n; node++ {
		lab := c.Label(node)
		if lab < 0 || lab >= c.NumCommunities() {
			t.Fatalf("node %d label %d out of range [0,%d)", node, lab, c.NumCommunities())
		}
		seen[node] = true
	}
	for comm := 0; comm < c.NumCommunities(); comm++ {
		prev := -1
		for _, node := range c.Members(comm) {
			if node <= prev {
				t.Fatalf("community %d members not ascending: %v", comm, c.Members(comm))
			}
			if c.Label(node) != comm {
				t.Fatalf("community %d lists node %d whose label is %d", comm, node, c.Label(node))
			}
			prev = node
		}
	}
}

// TestUpdateCommunitiesKeepsExistingLabels is the core contract: appending an
// isolated node must not disturb the communities already detected, and the new
// node's community id must be allocated after the preserved range.
func TestUpdateCommunitiesKeepsExistingLabels(t *testing.T) {
	b1 := NewBuilder(8)
	clique(b1, 1.0, 0, 1, 2, 3)
	clique(b1, 1.0, 4, 5, 6, 7)
	b1.AddEdge(3, 4, 0.01)
	prev := DetectCommunities(b1.Build(), CommunityOpts{Seed: 42})
	if prev.NumCommunities() != 2 {
		t.Fatalf("setup: expected 2 communities, got %d", prev.NumCommunities())
	}

	// A new graph with node 8 appended, isolated from both cliques.
	b2 := NewBuilder(9)
	clique(b2, 1.0, 0, 1, 2, 3)
	clique(b2, 1.0, 4, 5, 6, 7)
	b2.AddEdge(3, 4, 0.01)
	g2 := b2.Build()

	updated := UpdateCommunities(prev, g2, CommunityOpts{Seed: 42})
	checkPartition(t, updated, 9)

	if !reflect.DeepEqual(labelsRange(updated, 0, 8), labelsRange(prev, 0, 8)) {
		t.Errorf("existing labels changed: before %v after %v",
			labelsRange(prev, 0, 8), labelsRange(updated, 0, 8))
	}
	if updated.Label(8) < prev.NumCommunities() {
		t.Errorf("isolated node took old community id %d (< %d)", updated.Label(8), prev.NumCommunities())
	}
	if updated.NumCommunities() != prev.NumCommunities()+1 {
		t.Errorf("NumCommunities = %d, want %d", updated.NumCommunities(), prev.NumCommunities()+1)
	}
}

// TestUpdateCommunitiesJoinsTouchedCommunity appends a node wired to the second
// clique and expects it to join that community while the first stays untouched.
func TestUpdateCommunitiesJoinsTouchedCommunity(t *testing.T) {
	b1 := NewBuilder(8)
	clique(b1, 1.0, 0, 1, 2, 3)
	clique(b1, 1.0, 4, 5, 6, 7)
	b1.AddEdge(3, 4, 0.01)
	prev := DetectCommunities(b1.Build(), CommunityOpts{Seed: 42})

	b2 := NewBuilder(9)
	clique(b2, 1.0, 0, 1, 2, 3)
	clique(b2, 1.0, 4, 5, 6, 7)
	b2.AddEdge(3, 4, 0.01)
	clique(b2, 1.0, 4, 5, 6, 7, 8)
	g2 := b2.Build()

	updated := UpdateCommunities(prev, g2, CommunityOpts{Seed: 42})
	checkPartition(t, updated, 9)

	if updated.Label(8) != updated.Label(4) {
		t.Errorf("appended node label %d, want it to join second clique (label %d)",
			updated.Label(8), updated.Label(4))
	}
	if !reflect.DeepEqual(labelsRange(updated, 0, 4), labelsRange(prev, 0, 4)) {
		t.Errorf("untouched first clique changed: before %v after %v",
			labelsRange(prev, 0, 4), labelsRange(updated, 0, 4))
	}
	if updated.NumCommunities() != 2 {
		t.Errorf("NumCommunities = %d, want 2", updated.NumCommunities())
	}
}

// TestUpdateCommunitiesDeterministic runs the same incremental update twice and
// requires bit-for-bit identical partitions.
func TestUpdateCommunitiesDeterministic(t *testing.T) {
	b1 := NewBuilder(6)
	clique(b1, 1.0, 0, 1, 2)
	clique(b1, 1.0, 3, 4, 5)
	prev := DetectCommunities(b1.Build(), CommunityOpts{Seed: 5})

	b2 := NewBuilder(8)
	clique(b2, 1.0, 0, 1, 2)
	clique(b2, 1.0, 3, 4, 5)
	clique(b2, 1.0, 3, 4, 5, 6, 7)
	g2 := b2.Build()

	a := UpdateCommunities(prev, g2, CommunityOpts{Seed: 5})
	b := UpdateCommunities(prev, g2, CommunityOpts{Seed: 5})
	if !reflect.DeepEqual(a.labels, b.labels) {
		t.Fatalf("labels differ between runs: %v vs %v", a.labels, b.labels)
	}
	if !reflect.DeepEqual(a.members, b.members) {
		t.Fatalf("members differ between runs")
	}
}

// TestUpdateCommunitiesGrowsAcrossBatches chains two appends and checks that the
// first community keeps its identity the whole way.
func TestUpdateCommunitiesGrowsAcrossBatches(t *testing.T) {
	b1 := NewBuilder(4)
	clique(b1, 1.0, 0, 1, 2, 3)
	c1 := DetectCommunities(b1.Build(), CommunityOpts{Seed: 9})

	b2 := NewBuilder(8)
	clique(b2, 1.0, 0, 1, 2, 3)
	clique(b2, 1.0, 4, 5, 6, 7)
	c2 := UpdateCommunities(c1, b2.Build(), CommunityOpts{Seed: 9})

	b3 := NewBuilder(12)
	clique(b3, 1.0, 0, 1, 2, 3)
	clique(b3, 1.0, 4, 5, 6, 7)
	clique(b3, 1.0, 8, 9, 10, 11)
	c3 := UpdateCommunities(c2, b3.Build(), CommunityOpts{Seed: 9})

	checkPartition(t, c2, 8)
	checkPartition(t, c3, 12)
	if !reflect.DeepEqual(labelsRange(c3, 0, 4), labelsRange(c1, 0, 4)) {
		t.Errorf("first clique label drifted: %v -> %v", labelsRange(c1, 0, 4), labelsRange(c3, 0, 4))
	}
	if !reflect.DeepEqual(labelsRange(c3, 4, 8), labelsRange(c2, 4, 8)) {
		t.Errorf("second clique label drifted: %v -> %v", labelsRange(c2, 4, 8), labelsRange(c3, 4, 8))
	}
	if c1.NumCommunities() != 1 || c2.NumCommunities() != 2 || c3.NumCommunities() != 3 {
		t.Errorf("NumCommunities = %d/%d/%d, want 1/2/3",
			c1.NumCommunities(), c2.NumCommunities(), c3.NumCommunities())
	}
}

// TestUpdateCommunitiesRejectsShrink guards the precondition: a partition for a
// larger graph cannot be extended to a smaller one.
func TestUpdateCommunitiesRejectsShrink(t *testing.T) {
	b1 := NewBuilder(4)
	clique(b1, 1.0, 0, 1, 2, 3)
	prev := DetectCommunities(b1.Build(), CommunityOpts{Seed: 1})

	b2 := NewBuilder(2)
	clique(b2, 1.0, 0, 1)
	if got := UpdateCommunities(prev, b2.Build(), CommunityOpts{Seed: 1}); got != nil {
		t.Fatalf("expected nil for a shrunk graph, got %+v", got)
	}
}
