package rag

// mmrRerank reorders candidates by Maximal Marginal Relevance, trading relevance
// against redundancy. Graph retrieval tends to surface several near-duplicate
// neighbors of the same hit; MMR keeps the most relevant while penalizing
// candidates too similar to those already chosen.
//
// rel[i] is the relevance of candidate i (higher is better). vecs[i] is its
// L2-normalized embedding so a dot product is cosine similarity. lambda in [0,1]
// weights relevance against diversity: 1 is pure relevance, 0 is pure diversity.
// It returns the selected candidate indices in order, at most k of them.
func mmrRerank(rel []float32, vecs [][]float32, lambda float32, k int) []int {
	n := len(rel)
	if k > n {
		k = n
	}
	if k <= 0 {
		return nil
	}
	selected := make([]int, 0, k)
	chosen := make([]bool, n)
	// maxSim[i] tracks the largest similarity of candidate i to any selected item.
	maxSim := make([]float32, n)

	for len(selected) < k {
		best := -1
		var bestScore float32
		for i := 0; i < n; i++ {
			if chosen[i] {
				continue
			}
			score := lambda*rel[i] - (1-lambda)*maxSim[i]
			if best == -1 || score > bestScore {
				best, bestScore = i, score
			}
		}
		if best == -1 {
			break
		}
		chosen[best] = true
		selected = append(selected, best)
		// Update redundancy penalties against the newly selected item.
		bv := vecs[best]
		for i := 0; i < n; i++ {
			if chosen[i] {
				continue
			}
			if s := dot32(vecs[i], bv); s > maxSim[i] {
				maxSim[i] = s
			}
		}
	}
	return selected
}

func dot32(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
