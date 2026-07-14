package index

import (
	"math"
	"sort"
	"sync"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// HNSW is a Hierarchical Navigable Small World graph for sublinear approximate
// nearest-neighbor search. It is the standard structure used by production vector
// databases (FAISS, Qdrant, Weaviate, hnswlib) because it gives high recall at a
// small fraction of a flat scan's cost once a corpus grows past a few thousand
// vectors.
//
// Vectors are stored L2-normalized, so cosine similarity is a plain inner product
// and distance is 1 - dot. Search and build use exact float distance, which is
// SIMD-friendly and gives the best graph quality. The original (un-normalized)
// vectors are quantized with the supplied TurboQuant quantizer and retained as
// codes so a caller can also run the compact asymmetric estimator if desired.
type HNSW struct {
	mu sync.RWMutex

	dim    int
	m      int     // max neighbors per node on upper layers
	m0     int     // max neighbors on layer 0
	efCons int     // search width during construction
	mL     float64 // level generation factor, 1/ln(M)
	rng    *quant.PCG

	data  []float32 // flat normalized vectors, row-major
	ids   []string
	idOrd map[string]int32
	nodes []hnswNode

	entry int32 // entry point node, -1 if empty
	top   int   // current top level
}

type hnswNode struct {
	level   int
	friends [][]int32 // friends[l] is the neighbor list at level l
}

// HNSWConfig parameterizes graph construction. Zero fields take sensible
// defaults matching common library settings.
type HNSWConfig struct {
	M              int    // neighbor degree (default 16)
	EfConstruction int    // build-time search width (default 200)
	Seed           uint64 // determinism
}

// NewHNSW creates an empty graph. Distances use the exact normalized vectors.
//
// It used to also TurboQuant-encode every inserted vector and retain the codes. The
// codes were never read by anything: they are the residue of a quantized-traversal
// experiment that was measured at a 13x slowdown and abandoned. Producing them cost
// most of the time spent opening a store and most of the memory spent building one, so
// they are gone. Quantization is still what the flat index and the lean storage modes
// are built on; it just has no business here.
func NewHNSW(dim int, cfg HNSWConfig) *HNSW {
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	return &HNSW{
		dim:    dim,
		m:      cfg.M,
		m0:     2 * cfg.M,
		efCons: cfg.EfConstruction,
		mL:     1.0 / math.Log(float64(cfg.M)),
		rng:    quant.NewPCG(cfg.Seed),
		idOrd:  make(map[string]int32),
		entry:  -1,
	}
}

// Len returns the number of indexed vectors.
func (h *HNSW) Len() int { return len(h.ids) }

// Ord returns the ordinal for an id.
func (h *HNSW) Ord(id string) (int, bool) {
	o, ok := h.idOrd[id]
	return int(o), ok
}

// Vector returns the stored normalized vector for an ordinal (read-only).
func (h *HNSW) Vector(ord int) []float32 {
	return h.data[ord*h.dim : ord*h.dim+h.dim]
}

// Code returns the TurboQuant code for an ordinal.

func normalize(dst, src []float32) {
	var n float64
	for _, v := range src {
		n += float64(v) * float64(v)
	}
	n = math.Sqrt(n)
	if n == 0 {
		copy(dst, src)
		return
	}
	inv := float32(1.0 / n)
	for i, v := range src {
		dst[i] = v * inv
	}
}

// dotf returns the inner product of two equal-length vectors. It is the single
// hottest function in the index. Its implementation is selected at build time:
// an AVX kernel on amd64 (see dotf_amd64.go), and a portable Go version
// everywhere else (see dotf.go). Both are exercised by the same tests.

// distToNode returns 1 - cosine between the query (normalized) and node i.
func (h *HNSW) distToNode(query []float32, i int32) float32 {
	v := h.data[int(i)*h.dim : int(i)*h.dim+h.dim]
	return 1 - dotf(query, v)
}

// randomLevel draws an exponentially distributed level.
func (h *HNSW) randomLevel() int {
	u := h.rng.Float64()
	if u < 1e-300 {
		u = 1e-300
	}
	return int(-math.Log(u) * h.mL)
}

// Add inserts a vector under id and returns its ordinal. It is safe to call
// concurrently with reads but not with other Adds.
func (h *HNSW) Add(id string, vec []float32) int {
	h.mu.Lock()
	defer h.mu.Unlock()

	cur := int32(len(h.ids))
	norm := make([]float32, h.dim)
	normalize(norm, vec)
	h.data = append(h.data, norm...)
	h.ids = append(h.ids, id)
	h.idOrd[id] = cur

	level := h.randomLevel()
	node := hnswNode{level: level, friends: make([][]int32, level+1)}
	h.nodes = append(h.nodes, node)

	if h.entry == -1 {
		h.entry = cur
		h.top = level
		return int(cur)
	}

	q := norm
	ep := h.entry
	// Descend from the top to just above the new node's level, greedily.
	for l := h.top; l > level; l-- {
		ep = h.greedyDescend(q, ep, l)
	}
	// Insert into every level the node participates in.
	start := min(level, h.top)
	for l := start; l >= 0; l-- {
		cands := h.searchLayer(q, ep, h.efCons, l)
		maxConn := h.m
		if l == 0 {
			maxConn = h.m0
		}
		neighbors := h.selectNeighbors(q, cands, maxConn)
		h.nodes[cur].friends[l] = neighbors
		// Add reverse links and prune over-full neighbors.
		for _, nb := range neighbors {
			h.nodes[nb].friends[l] = append(h.nodes[nb].friends[l], cur)
			if len(h.nodes[nb].friends[l]) > maxConn {
				h.nodes[nb].friends[l] = h.pruneConnections(nb, l, maxConn)
			}
		}
		if len(cands) > 0 {
			ep = cands[0].id
		}
	}
	if level > h.top {
		h.top = level
		h.entry = cur
	}
	return int(cur)
}

// greedyDescend walks one layer toward the query, returning the closest node.
func (h *HNSW) greedyDescend(query []float32, ep int32, level int) int32 {
	best := ep
	bestD := h.distToNode(query, ep)
	for {
		improved := false
		for _, nb := range h.nodes[best].friends[level] {
			d := h.distToNode(query, nb)
			if d < bestD {
				bestD, best, improved = d, nb, true
			}
		}
		if !improved {
			return best
		}
	}
}

type candidate struct {
	id   int32
	dist float32
}

// searchLayer runs the HNSW ef-search at one level from entry point ep, returning
// the ef closest nodes sorted ascending by distance.
func (h *HNSW) searchLayer(query []float32, ep int32, ef, level int) []candidate {
	return h.searchLayerFiltered(query, ep, ef, level, nil)
}

// searchLayerFiltered is searchLayer with an optional accept predicate. When
// accept is non-nil, the traversal still expands through every node (so graph
// connectivity is preserved) but only accepted nodes enter the result set. This
// is the integrated pre-filtering used by production vector databases: it keeps
// recall high for moderately selective filters without fragmenting the graph.
func (h *HNSW) searchLayerFiltered(query []float32, ep int32, ef, level int, accept func(int32) bool) []candidate {
	visited := h.acquireVisited()
	defer h.releaseVisited(visited)

	d0 := h.distToNode(query, ep)
	visited.mark(int(ep))
	cand := &minHeap{{ep, d0}}
	result := &maxHeap{}
	if accept == nil || accept(ep) {
		*result = append(*result, candidate{ep, d0})
	}

	for cand.Len() > 0 {
		c := cand.pop()
		// Stop once the nearest unexplored candidate is farther than the current
		// worst result and the result set is already full.
		if result.Len() >= ef && c.dist > (*result)[0].dist {
			break
		}
		for _, nb := range h.nodes[c.id].friends[level] {
			if visited.seen(int(nb)) {
				continue
			}
			visited.mark(int(nb))
			d := h.distToNode(query, nb)
			// Expand through nb for connectivity regardless of the filter.
			if result.Len() < ef || d < (*result)[0].dist {
				cand.push(candidate{nb, d})
				if accept == nil || accept(nb) {
					result.push(candidate{nb, d})
					if result.Len() > ef {
						result.pop()
					}
				}
			}
		}
	}
	out := make([]candidate, result.Len())
	copy(out, *result)
	sort.Slice(out, func(a, b int) bool { return out[a].dist < out[b].dist })
	return out
}

// selectNeighbors applies the HNSW heuristic: keep a candidate only if it is
// closer to the query than to any already-selected neighbor. This spreads links
// across directions and avoids clustering them on one side, which is what gives
// HNSW its good long-range connectivity.
func (h *HNSW) selectNeighbors(query []float32, cands []candidate, m int) []int32 {
	selected := make([]int32, 0, m)
	for _, c := range cands {
		if len(selected) >= m {
			break
		}
		good := true
		cv := h.data[int(c.id)*h.dim : int(c.id)*h.dim+h.dim]
		for _, s := range selected {
			sv := h.data[int(s)*h.dim : int(s)*h.dim+h.dim]
			if 1-dotf(cv, sv) < c.dist {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, c.id)
		}
	}
	return selected
}

// pruneConnections re-selects an over-full neighbor list down to m using the same
// heuristic, keeping the most useful links.
func (h *HNSW) pruneConnections(node int32, level, m int) []int32 {
	nv := h.data[int(node)*h.dim : int(node)*h.dim+h.dim]
	friends := h.nodes[node].friends[level]
	cands := make([]candidate, len(friends))
	for i, f := range friends {
		fv := h.data[int(f)*h.dim : int(f)*h.dim+h.dim]
		cands[i] = candidate{f, 1 - dotf(nv, fv)}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].dist < cands[b].dist })
	return h.selectNeighbors(nv, cands, m)
}

// Search returns the top-k ids for query at the given search width ef. Larger ef
// trades latency for recall. If ef < k it is raised to k.
func (h *HNSW) Search(query []float32, k, ef int) []Result {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entry == -1 || k <= 0 {
		return nil
	}
	if ef < k {
		ef = k
	}
	norm := make([]float32, h.dim)
	normalize(norm, query)

	ep := h.entry
	for l := h.top; l > 0; l-- {
		ep = h.greedyDescend(norm, ep, l)
	}
	cands := h.searchLayer(norm, ep, ef, 0)
	if len(cands) > k {
		cands = cands[:k]
	}
	out := make([]Result, len(cands))
	for i, c := range cands {
		out[i] = Result{ID: h.ids[c.id], Score: 1 - c.dist}
	}
	return out
}

// SearchFiltered is Search restricted to ids for which accept returns true. The
// predicate is evaluated by ordinal. For very selective filters, prefer a flat
// scan over the matching subset; this method is best when a meaningful fraction
// of the corpus passes.
func (h *HNSW) SearchFiltered(query []float32, k, ef int, accept func(id string) bool) []Result {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entry == -1 || k <= 0 {
		return nil
	}
	if ef < k {
		ef = k
	}
	norm := make([]float32, h.dim)
	normalize(norm, query)
	pred := func(ord int32) bool { return accept(h.ids[ord]) }

	ep := h.entry
	for l := h.top; l > 0; l-- {
		ep = h.greedyDescend(norm, ep, l)
	}
	cands := h.searchLayerFiltered(norm, ep, ef, 0, pred)
	if len(cands) > k {
		cands = cands[:k]
	}
	out := make([]Result, len(cands))
	for i, c := range cands {
		out[i] = Result{ID: h.ids[c.id], Score: 1 - c.dist}
	}
	return out
}

// Graph is the HNSW link structure, without the vectors. It is everything that is
// expensive to compute and cheap to store: building it means a graph search per
// inserted vector, which dominates the cost of opening a store, while the links
// themselves are a few bytes per node.
type Graph struct {
	Entry   int32       `json:"entry"`
	Top     int         `json:"top"`
	Levels  []int32     `json:"levels"`  // per node
	Friends [][][]int32 `json:"friends"` // per node, per level, neighbor ordinals
}

// Snapshot exports the link structure for persistence.
func (h *HNSW) Snapshot() Graph {
	h.mu.RLock()
	defer h.mu.RUnlock()
	g := Graph{Entry: h.entry, Top: h.top,
		Levels:  make([]int32, len(h.nodes)),
		Friends: make([][][]int32, len(h.nodes)),
	}
	for i, n := range h.nodes {
		g.Levels[i] = int32(n.level)
		g.Friends[i] = n.friends
	}
	return g
}

// RestoreHNSW builds an index from vectors and a previously exported link structure,
// skipping link construction entirely. The ids and vectors must be in the order they
// were originally added, because the graph refers to nodes by ordinal.
//
// It reports false if the graph does not describe these vectors, in which case the
// caller must rebuild. A stale graph is not an error, it is a cache miss.
func RestoreHNSW(dim int, cfg HNSWConfig, ids []string, vecs [][]float32, g Graph) (*HNSW, bool) {
	if len(ids) != len(vecs) || len(g.Levels) != len(ids) || len(g.Friends) != len(ids) {
		return nil, false
	}
	h := NewHNSW(dim, cfg)
	norm := make([]float32, dim)
	for i, v := range vecs {
		if len(v) != dim {
			return nil, false
		}
		cur := int32(len(h.ids))
		normalize(norm, v)
		h.data = append(h.data, norm...)
		h.ids = append(h.ids, ids[i])
		h.idOrd[ids[i]] = cur
	}
	h.nodes = make([]hnswNode, len(ids))
	for i := range ids {
		h.nodes[i] = hnswNode{level: int(g.Levels[i]), friends: g.Friends[i]}
	}
	h.entry, h.top = g.Entry, g.Top
	return h, true
}

// AdoptFlat builds an index from a previously exported link structure and a flat,
// row-major vector buffer that it TAKES OWNERSHIP OF, normalizing it in place.
//
// It exists because the alternative is to copy. A store's vectors arrive from disk as
// one contiguous block, which is exactly the layout the index wants, and copying them
// into a second identical block cost the largest allocation in the load path and doubled
// the peak memory of opening a store. The caller must not write to flat afterwards; it
// is the index's buffer now.
//
// It reports false if the graph does not describe these vectors, in which case the
// caller rebuilds. A stale graph is a cache miss, not an error.
func AdoptFlat(dim int, cfg HNSWConfig, ids []string, flat []float32, g Graph) (*HNSW, bool) {
	n := len(ids)
	if n == 0 || len(flat) != n*dim || len(g.Levels) != n || len(g.Friends) != n {
		return nil, false
	}
	h := NewHNSW(dim, cfg)
	// Normalize in place: distances are computed against unit vectors, and normalize is
	// safe when its destination and source are the same slice.
	for i := 0; i < n; i++ {
		v := flat[i*dim : (i+1)*dim]
		normalize(v, v)
	}
	h.data = flat
	h.ids = ids
	for i, id := range ids {
		h.idOrd[id] = int32(i)
	}
	h.nodes = make([]hnswNode, n)
	for i := 0; i < n; i++ {
		h.nodes[i] = hnswNode{level: int(g.Levels[i]), friends: g.Friends[i]}
	}
	h.entry, h.top = g.Entry, g.Top
	return h, true
}
