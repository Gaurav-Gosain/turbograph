package index

import (
	"fmt"
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

func TestHNSWFilteredSearch(t *testing.T) {
	rng := quant.NewPCG(4)
	d, n := 128, 3000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})

	// Accept only even-id documents; every returned id must satisfy the predicate.
	accept := func(id string) bool {
		return len(id) > 0 && (id[len(id)-1]-'0')%2 == 0
	}
	for trial := 0; trial < 20; trial++ {
		q := rows[randIdx(rng, n)]
		res := h.SearchFiltered(q, 10, 128, accept)
		if len(res) == 0 {
			t.Fatal("filtered search returned nothing")
		}
		for _, r := range res {
			if !accept(r.ID) {
				t.Errorf("filter violated: %s", r.ID)
			}
		}
	}
}

func TestHNSWFilteredRecall(t *testing.T) {
	rng := quant.NewPCG(6)
	d, n := 96, 4000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})

	// Filter to a labeled half of the corpus and compare against brute force over
	// that same half, so we measure recall of the filtered path specifically.
	keep := func(ord int) bool { return ord%2 == 0 }
	accept := func(id string) bool {
		var ord int
		fmt.Sscanf(id, "%d", &ord)
		return keep(ord)
	}
	var sum float64
	const queries = 40
	for tq := 0; tq < queries; tq++ {
		q := rows[randIdx(rng, n)]
		// brute over the kept subset
		type sc struct {
			i int
			s float64
		}
		qn := float64(vnorm(q))
		var all []sc
		for i := range rows {
			if keep(i) {
				all = append(all, sc{i, dot(q, rows[i]) / (qn*float64(vnorm(rows[i])) + 1e-9)})
			}
		}
		for a := 0; a < 10; a++ {
			best := a
			for b := a + 1; b < len(all); b++ {
				if all[b].s > all[best].s {
					best = b
				}
			}
			all[a], all[best] = all[best], all[a]
		}
		truth := make([]int, 10)
		for a := 0; a < 10; a++ {
			truth[a] = all[a].i
		}
		res := h.SearchFiltered(q, 10, 128, accept)
		sum += recallAt(res, truth, 10)
	}
	recall := sum / queries
	if recall < 0.90 {
		t.Errorf("filtered recall@10=%.3f below 0.90", recall)
	}
	t.Logf("filtered recall@10=%.3f", recall)
}

var _ = strings.TrimSpace
