// Package graph implements a compact weighted graph and Personalized PageRank.
// In the RAG pipeline the graph connects chunks by semantic similarity (and any
// structural edges such as document adjacency); seeding PageRank with the
// vector-search hits and propagating over the graph surfaces context that is
// relevant by association rather than by direct similarity alone.
package graph

import "math"

// Builder accumulates weighted undirected edges before they are frozen into the
// immutable CSR form used for traversal.
type Builder struct {
	n   int
	adj []map[int]float32
}

// NewBuilder creates a builder for n nodes.
func NewBuilder(n int) *Builder {
	adj := make([]map[int]float32, n)
	for i := range adj {
		adj[i] = make(map[int]float32)
	}
	return &Builder{n: n, adj: adj}
}

// AddEdge adds weight w to the undirected edge (a, b). Repeated edges keep the
// maximum weight rather than summing, which keeps a single strong similarity
// from being double counted when both endpoints list each other as neighbors.
func (b *Builder) AddEdge(a, bb int, w float32) {
	if a == bb || a < 0 || bb < 0 || a >= b.n || bb >= b.n || w <= 0 {
		return
	}
	if cur, ok := b.adj[a][bb]; !ok || w > cur {
		b.adj[a][bb] = w
		b.adj[bb][a] = w
	}
}

// Build freezes the accumulated edges into a Graph.
func (b *Builder) Build() *Graph {
	rowPtr := make([]int, b.n+1)
	total := 0
	for i := 0; i < b.n; i++ {
		total += len(b.adj[i])
	}
	cols := make([]int32, total)
	wts := make([]float32, total)
	outSum := make([]float32, b.n)
	pos := 0
	for i := 0; i < b.n; i++ {
		rowPtr[i] = pos
		var s float32
		for j, w := range b.adj[i] {
			cols[pos] = int32(j)
			wts[pos] = w
			s += w
			pos++
		}
		outSum[i] = s
	}
	rowPtr[b.n] = pos
	return &Graph{n: b.n, rowPtr: rowPtr, cols: cols, wts: wts, outSum: outSum}
}

// Graph is an immutable weighted undirected graph in compressed sparse row form.
type Graph struct {
	n      int
	rowPtr []int
	cols   []int32
	wts    []float32
	outSum []float32
}

// Snapshot is a serializable view of the graph's CSR arrays.
type Snapshot struct {
	N      int
	RowPtr []int
	Cols   []int32
	Wts    []float32
	OutSum []float32
}

// Snapshot returns the underlying CSR arrays for serialization.
func (g *Graph) Snapshot() Snapshot {
	return Snapshot{N: g.n, RowPtr: g.rowPtr, Cols: g.cols, Wts: g.wts, OutSum: g.outSum}
}

// Restore rebuilds a graph from a snapshot.
func Restore(s Snapshot) *Graph {
	return &Graph{n: s.N, rowPtr: s.RowPtr, cols: s.Cols, wts: s.Wts, outSum: s.OutSum}
}

// N returns the number of nodes.
func (g *Graph) N() int { return g.n }

// Degree returns the number of neighbors of node i.
func (g *Graph) Degree(i int) int { return g.rowPtr[i+1] - g.rowPtr[i] }

// Neighbors calls fn for each neighbor of i with its edge weight.
func (g *Graph) Neighbors(i int, fn func(j int, w float32)) {
	for p := g.rowPtr[i]; p < g.rowPtr[i+1]; p++ {
		fn(int(g.cols[p]), g.wts[p])
	}
}

// PPRParams configures Personalized PageRank.
type PPRParams struct {
	Damping   float32 // restart probability is 1-Damping; typical Damping 0.85
	MaxIter   int     // iteration cap
	Tolerance float32 // L1 convergence threshold
}

// DefaultPPR returns sensible defaults.
func DefaultPPR() PPRParams {
	return PPRParams{Damping: 0.85, MaxIter: 100, Tolerance: 1e-6}
}

// PersonalizedPageRank runs PPR with restart distribution seeds (node -> mass).
// The seed mass is normalized internally. It returns a stationary score per node.
//
// The iteration is r' = (1-d)*s + d * W^T_norm r, where W_norm is the weighted
// transition with each column... here the graph is undirected so the transition
// from j distributes j's mass across its neighbors proportional to edge weight.
func (g *Graph) PersonalizedPageRank(seeds map[int]float32, p PPRParams) []float32 {
	if p.Damping <= 0 {
		p = DefaultPPR()
	}
	s := make([]float32, g.n)
	var total float32
	for node, m := range seeds {
		if node >= 0 && node < g.n && m > 0 {
			s[node] += m
			total += m
		}
	}
	if total == 0 {
		return s
	}
	inv := 1.0 / total
	for i := range s {
		s[i] *= inv
	}

	r := make([]float32, g.n)
	copy(r, s)
	next := make([]float32, g.n)
	d := p.Damping
	for iter := 0; iter < p.MaxIter; iter++ {
		// Restart component.
		for i := range next {
			next[i] = (1 - d) * s[i]
		}
		// Propagation: push each node's mass to neighbors, weighted.
		var dangling float32
		for i := 0; i < g.n; i++ {
			ri := r[i]
			if ri == 0 {
				continue
			}
			if g.outSum[i] == 0 {
				dangling += ri
				continue
			}
			share := d * ri / g.outSum[i]
			for q := g.rowPtr[i]; q < g.rowPtr[i+1]; q++ {
				next[g.cols[q]] += share * g.wts[q]
			}
		}
		// Dangling mass (isolated nodes) returns to the restart distribution.
		if dangling > 0 {
			for i := range next {
				next[i] += d * dangling * s[i]
			}
		}
		var diff float32
		for i := range r {
			diff += float32(math.Abs(float64(next[i] - r[i])))
			r[i] = next[i]
		}
		if diff < p.Tolerance {
			break
		}
	}
	return r
}
