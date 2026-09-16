package rag

import (
	"context"
	"fmt"
	"math"
	"testing"
)

// bruteCosineTop returns the true dense top-k ordinals by a flat cosine scan.
func bruteCosineTop(qv []float32, embeds [][]float32, k int) []int {
	type scored struct {
		ord int
		s   float32
	}
	all := make([]scored, len(embeds))
	for i, e := range embeds {
		if len(e) != len(qv) {
			continue
		}
		var dot, na, nb float32
		for j := range qv {
			dot += qv[j] * e[j]
			na += qv[j] * qv[j]
			nb += e[j] * e[j]
		}
		denom := float32(math.Sqrt(float64(na)*float64(nb)) + 1e-9)
		all[i] = scored{i, dot / denom}
	}
	// partial selection sort for the top k, then return ordinals in rank order
	top := all[:min(k, len(all))]
	for i := 0; i < len(top); i++ {
		best := i
		for j := i + 1; j < len(all); j++ {
			if all[j].s > all[best].s {
				best = j
			}
		}
		all[i], all[best] = all[best], all[i]
	}
	out := make([]int, 0, len(top))
	for _, t := range top {
		out = append(out, t.ord)
	}
	return out
}

// seedMapTopK returns the top-k ordinals of a dense seed score map.
func seedMapTopK(sim map[int]float32, k int) []int {
	ords := make([]int, 0, len(sim))
	for o := range sim {
		ords = append(ords, o)
	}
	for i := 0; i < len(ords); i++ {
		best := i
		for j := i + 1; j < len(ords); j++ {
			if sim[ords[j]] > sim[ords[best]] {
				best = j
			}
		}
		ords[i], ords[best] = ords[best], ords[i]
		if i == k-1 {
			break
		}
	}
	if len(ords) > k {
		ords = ords[:k]
	}
	return ords
}

func seedRecall(sim map[int]float32, truth []int, k int) float64 {
	if len(sim) == 0 {
		return 0
	}
	got := seedMapTopK(sim, k)
	inTruth := make(map[int]bool, len(truth))
	for _, o := range truth {
		inTruth[o] = true
	}
	hit := 0
	for _, o := range got {
		if inTruth[o] {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

// TestEfGatherSeedRecallNoRegression pins the O4 guarantee: widening ef_gather for
// the PageRank seed frontier must never lose dense recall compared to the
// single-width default, and must hold a sane floor. Until this passes with a
// positive gain, Config.EfGather stays default OFF.
func TestEfGatherSeedRecallNoRegression(t *testing.T) {
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.05,
		Chunk: ChunkConfig{TargetWords: 200}})
	const n = 2000
	docs := make([]Document, n)
	for i := range docs {
		docs[i] = Document{ID: fmt.Sprintf("d%d", i),
			Text: fmt.Sprintf("alpha beta gamma topic %d delta epsilon zeta eta theta kappa lambda mu", i)}
	}
	if err := st.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	st.mu.RLock()
	embeds := make([][]float32, len(st.embeds))
	copy(embeds, st.embeds)
	st.mu.RUnlock()

	const seedK = 24
	for _, qi := range []int{0, 500, 1999} {
		query := docs[qi].Text
		v, err := embedQuery(context.Background(), st.embedder, []string{query})
		if err != nil {
			t.Fatal(err)
		}
		truth := bruteCosineTop(v[0], embeds, seedK)

		st.cfg.EfGather = 0
		_, simDefault, _ := st.seedScores(query, v[0], seedK, nil)
		st.cfg.EfGather = 4096
		_, simGather, _ := st.seedScores(query, v[0], seedK, nil)

		recDefault := seedRecall(simDefault, truth, seedK)
		recGather := seedRecall(simGather, truth, seedK)
		if recGather < recDefault-0.01 {
			t.Errorf("query %q: gather recall %.4f < default %.4f (ef_gather must not regress seeds)",
				query, recGather, recDefault)
		}
		if recGather < 0.9 {
			t.Errorf("gather recall %.4f below 0.9 floor for %q", recGather, query)
		}
	}
}
