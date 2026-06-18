package graph

import (
	"math"
	"testing"
)

func TestBuildSymmetry(t *testing.T) {
	b := NewBuilder(4)
	b.AddEdge(0, 1, 1.0)
	b.AddEdge(1, 2, 2.0)
	b.AddEdge(2, 3, 0.5)
	b.AddEdge(0, 0, 5.0) // self-loop ignored
	b.AddEdge(0, 1, 0.2) // weaker duplicate kept at max
	g := b.Build()

	if g.N() != 4 {
		t.Fatalf("N=%d", g.N())
	}
	if g.Degree(0) != 1 || g.Degree(1) != 2 {
		t.Fatalf("degrees wrong: %d %d", g.Degree(0), g.Degree(1))
	}
	// Edge (0,1) weight should be max(1.0, 0.2) = 1.0 on both sides.
	check := func(a, want int, wexp float32) {
		found := false
		g.Neighbors(a, func(j int, w float32) {
			if j == want {
				found = true
				if math.Abs(float64(w-wexp)) > 1e-6 {
					t.Errorf("edge (%d,%d) weight %.3f want %.3f", a, want, w, wexp)
				}
			}
		})
		if !found {
			t.Errorf("missing edge (%d,%d)", a, want)
		}
	}
	check(0, 1, 1.0)
	check(1, 0, 1.0)
	check(1, 2, 2.0)
}

func sum(v []float32) float32 {
	var s float32
	for _, x := range v {
		s += x
	}
	return s
}

func TestPPRDistributionSumsToOne(t *testing.T) {
	b := NewBuilder(20)
	for i := 0; i < 19; i++ {
		b.AddEdge(i, i+1, 1.0)
	}
	g := b.Build()
	r := g.PersonalizedPageRank(map[int]float32{0: 1}, DefaultPPR())
	if s := sum(r); math.Abs(float64(s-1.0)) > 1e-3 {
		t.Errorf("PPR mass not conserved: %.4f", s)
	}
}

// TestPPRDecaysWithDistance checks the central property: on a chain seeded at one
// end, scores decrease monotonically with graph distance from the seed.
func TestPPRDecaysWithDistance(t *testing.T) {
	n := 30
	b := NewBuilder(n)
	for i := 0; i < n-1; i++ {
		b.AddEdge(i, i+1, 1.0)
	}
	g := b.Build()
	r := g.PersonalizedPageRank(map[int]float32{0: 1}, DefaultPPR())
	// Mass decays monotonically moving outward from the seed's neighborhood. The
	// degree-1 seed funnels its mass to node 1, so the peak sits at node 1; from
	// there scores strictly decrease with distance.
	for i := 2; i < 12; i++ {
		if r[i] >= r[i-1] {
			t.Errorf("score did not decay at %d: r[%d]=%.5f >= r[%d]=%.5f", i, i, r[i], i-1, r[i-1])
		}
	}
	if r[1] <= r[15] {
		t.Errorf("near node should outrank far node: r[1]=%.5f r[15]=%.5f", r[1], r[15])
	}
}

// TestPPRProximityRanking verifies that a node directly connected to the seed
// outranks a node two hops away.
func TestPPRProximityRanking(t *testing.T) {
	// star with hub 0 and a tail off leaf 1: 0-1, 0-2, 0-3, 1-4
	b := NewBuilder(5)
	b.AddEdge(0, 1, 1)
	b.AddEdge(0, 2, 1)
	b.AddEdge(0, 3, 1)
	b.AddEdge(1, 4, 1)
	g := b.Build()
	r := g.PersonalizedPageRank(map[int]float32{0: 1}, DefaultPPR())
	if r[1] <= r[4] {
		t.Errorf("one-hop node 1 (%.4f) should outrank two-hop node 4 (%.4f)", r[1], r[4])
	}
	// The hub seed receives mass back from all three leaves, so it stays dominant.
	if r[0] <= r[1] {
		t.Errorf("hub seed should rank highest, but r[0]=%.4f <= r[1]=%.4f", r[0], r[1])
	}
}

func TestPPREmptySeeds(t *testing.T) {
	g := NewBuilder(3).Build()
	r := g.PersonalizedPageRank(nil, DefaultPPR())
	if sum(r) != 0 {
		t.Errorf("empty seeds should yield zero vector, got sum %.4f", sum(r))
	}
}

// TestPPRWeightingBias checks that heavier edges carry more mass.
func TestPPRWeightingBias(t *testing.T) {
	b := NewBuilder(3)
	b.AddEdge(0, 1, 0.1)
	b.AddEdge(0, 2, 0.9)
	g := b.Build()
	r := g.PersonalizedPageRank(map[int]float32{0: 1}, DefaultPPR())
	if r[2] <= r[1] {
		t.Errorf("heavier edge should give node 2 more mass: r[1]=%.4f r[2]=%.4f", r[1], r[2])
	}
}
