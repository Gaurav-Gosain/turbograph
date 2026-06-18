package graph

// Community detection groups densely connected nodes into communities using
// label propagation. In the RAG pipeline each community is a cluster of related
// chunks that can be summarized independently, which powers GraphRAG-style
// "global" queries where an answer is assembled from per-community summaries
// rather than from individually retrieved chunks.

// CommunityOpts configures label-propagation community detection.
type CommunityOpts struct {
	// MaxIter caps the number of propagation sweeps. Label propagation usually
	// converges in a handful of sweeps, so a small cap is enough; it mainly
	// guards against rare oscillation that never reaches a fixed point.
	MaxIter int
	// Seed makes the run fully deterministic. It drives the per-sweep node visit
	// order and is the only source of randomness, so two runs with the same seed
	// on the same graph produce identical labels regardless of wall-clock or
	// global RNG state.
	Seed uint64
}

// Communities is the result of community detection: a flat label per node plus
// the inverse mapping from a community id to its member nodes.
type Communities struct {
	labels  []int   // labels[node] is the compacted community id in [0, num)
	members [][]int // members[c] lists the nodes of community c, ascending
	num     int
}

// NumCommunities returns the number of distinct communities.
func (c *Communities) NumCommunities() int { return c.num }

// Label returns the community id of a node, or -1 if the node is out of range.
func (c *Communities) Label(node int) int {
	if node < 0 || node >= len(c.labels) {
		return -1
	}
	return c.labels[node]
}

// Members returns the nodes belonging to community c in ascending order. The
// returned slice is owned by the Communities value and must not be mutated.
func (c *Communities) Members(comm int) []int {
	if comm < 0 || comm >= c.num {
		return nil
	}
	return c.members[comm]
}

// splitMix64 is a tiny, fast, well-distributed PRNG. It is used instead of
// math/rand so the visit order depends only on Seed and never on global state,
// which keeps detection reproducible across processes and test runs.
type splitMix64 struct{ state uint64 }

func (s *splitMix64) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// shuffle performs an in-place Fisher-Yates shuffle driven entirely by the RNG,
// so the resulting permutation is a deterministic function of the seed.
func shuffle(order []int, rng *splitMix64) {
	for i := len(order) - 1; i > 0; i-- {
		// Unbiased-enough modulo is acceptable here: the visit order only needs
		// to be deterministic and reasonably mixed, not cryptographically fair.
		j := int(rng.next() % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
}

// DetectCommunities runs label-propagation community detection on g.
//
// Each node starts in its own singleton community. In every sweep the nodes are
// visited in a deterministic shuffled order, and each node adopts the label that
// carries the greatest summed edge weight among its neighbors, breaking ties by
// preferring the smallest label so the outcome is deterministic. Updates are
// applied immediately (asynchronous propagation), which speeds convergence. The
// process stops when a full sweep changes no label or when MaxIter is reached.
// Labels are finally compacted to the contiguous range [0, C).
func DetectCommunities(g *Graph, opts CommunityOpts) *Communities {
	if opts.MaxIter <= 0 {
		opts.MaxIter = 20
	}
	n := g.N()

	// Start every node in its own community.
	labels := make([]int, n)
	for i := range labels {
		labels[i] = i
	}

	if n == 0 {
		return &Communities{labels: labels, members: nil, num: 0}
	}

	order := make([]int, n)
	for i := range order {
		order[i] = i
	}

	rng := &splitMix64{state: opts.Seed}

	// Reusable scratch maps cleared each node to avoid per-node allocation.
	weightByLabel := make(map[int]float32)

	for iter := 0; iter < opts.MaxIter; iter++ {
		shuffle(order, rng)
		changed := false

		for _, node := range order {
			if g.Degree(node) == 0 {
				// Isolated nodes have no neighbor labels to adopt, so they keep
				// their singleton label and form their own community.
				continue
			}

			for k := range weightByLabel {
				delete(weightByLabel, k)
			}
			g.Neighbors(node, func(j int, w float32) {
				weightByLabel[labels[j]] += w
			})

			// Pick the heaviest label, breaking ties toward the smallest label
			// for determinism.
			best := labels[node]
			var bestW float32 = -1
			for lab, w := range weightByLabel {
				if w > bestW || (w == bestW && lab < best) {
					best = lab
					bestW = w
				}
			}

			if best != labels[node] {
				labels[node] = best
				changed = true
			}
		}

		if !changed {
			break
		}
	}

	return compact(labels)
}

// compact renumbers arbitrary labels into the dense range [0, C) and builds the
// per-community member lists. Members within a community come out in ascending
// node order because nodes are scanned in order.
func compact(labels []int) *Communities {
	remap := make(map[int]int)
	out := make([]int, len(labels))
	var members [][]int
	for node, lab := range labels {
		c, ok := remap[lab]
		if !ok {
			c = len(members)
			remap[lab] = c
			members = append(members, nil)
		}
		out[node] = c
		members[c] = append(members[c], node)
	}
	return &Communities{labels: out, members: members, num: len(members)}
}

// Modularity returns the Newman modularity Q of the partition for a weighted
// undirected graph. Q measures how much edge weight falls inside communities
// versus what would be expected if edges were rewired at random while preserving
// node strengths. It is the standard quality score for a partition, ranging up
// to just under 1; values well above 0 indicate genuine community structure
// while a single all-in-one partition yields Q near 0.
//
//	Q = (1 / 2m) * sum_ij [ A_ij - (k_i k_j) / 2m ] * delta(c_i, c_j)
//
// where A_ij is the edge weight, k_i is node i's total incident weight, m is
// half the total weight, and delta is 1 when i and j share a community.
func (c *Communities) Modularity(g *Graph) float64 {
	n := g.N()
	if n == 0 {
		return 0
	}

	// k holds each node's strength (summed incident edge weight). twoM is the
	// total incident weight over all nodes, i.e. 2m, since each undirected edge
	// contributes its weight to both endpoints.
	k := make([]float64, n)
	var twoM float64
	for i := 0; i < n; i++ {
		var s float64
		g.Neighbors(i, func(_ int, w float32) {
			s += float64(w)
		})
		k[i] = s
		twoM += s
	}
	if twoM == 0 {
		return 0
	}

	// Accumulate the two terms separately. The A_ij term sums the weight of
	// every intra-community edge (counted twice, once per direction, which
	// matches the i,j double sum). The expectation term sums (k_i k_j)/2m over
	// ordered same-community pairs, which factors per community into the squared
	// total strength of its members.
	var intra float64
	for i := 0; i < n; i++ {
		ci := c.labels[i]
		g.Neighbors(i, func(j int, w float32) {
			if c.labels[j] == ci {
				intra += float64(w)
			}
		})
	}

	sumK := make([]float64, c.num)
	for i := 0; i < n; i++ {
		sumK[c.labels[i]] += k[i]
	}
	var expected float64
	for _, sk := range sumK {
		expected += sk * sk
	}

	return intra/twoM - expected/(twoM*twoM)
}
