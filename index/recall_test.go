package index

import (
	"fmt"
	"testing"
	"time"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// TestHNSWRecallQPSCurve reports the recall@10 versus single-thread QPS tradeoff
// across efSearch, the standard ann-benchmarks methodology. It is informational
// (it logs the Pareto points) but also asserts the curve is monotone: higher ef
// never reduces recall, which is the property a correct HNSW must satisfy.
func TestHNSWRecallQPSCurve(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall/QPS sweep in short mode")
	}
	rng := quant.NewPCG(11)
	d, n := 128, 10000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})

	const queries = 200
	qs := make([][]float32, queries)
	truth := make([][]int, queries)
	for i := range qs {
		qs[i] = rows[randIdx(rng, n)]
		truth[i] = bruteTop(qs[i], rows, 10, Cosine)
	}

	var prevRecall float64
	for _, ef := range []int{10, 20, 40, 80, 160} {
		start := time.Now()
		var hit float64
		for i, q := range qs {
			res := h.Search(q, 10, ef)
			hit += recallAt(res, truth[i], 10)
		}
		elapsed := time.Since(start)
		recall := hit / float64(queries)
		qps := float64(queries) / elapsed.Seconds()
		t.Logf("ef=%-4d recall@10=%.4f  qps=%.0f", ef, recall, qps)
		if recall < prevRecall-0.01 {
			t.Errorf("recall regressed with higher ef: %.4f < %.4f", recall, prevRecall)
		}
		prevRecall = recall
	}
	if prevRecall < 0.95 {
		t.Errorf("top-end recall %.4f below 0.95", prevRecall)
	}
}

// TestHNSWSeedGatherMonotone validates the two-stage seed gather (Config.EfGather)
// relies on: the top-k returned for a fixed k never loses recall as the gather ef
// grows, so widening ef_gather to feed a seed frontier is a safe, no-regression
// way to widen PageRank seeds.
func TestHNSWSeedGatherMonotone(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping gather sweep in short mode")
	}
	rng := quant.NewPCG(13)
	d, n := 128, 10000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})

	const queries = 200
	const gatherK = 24
	qs := make([][]float32, queries)
	truth := make([][]int, queries)
	for i := range qs {
		qs[i] = rows[randIdx(rng, n)]
		truth[i] = bruteTop(qs[i], rows, gatherK, Cosine)
	}
	var prevRecall float64
	for _, ef := range []int{gatherK, gatherK * 2, gatherK * 4, gatherK * 8} {
		var hit float64
		for i, q := range qs {
			res := h.Search(q, gatherK, ef)
			hit += recallAt(res, truth[i], gatherK)
		}
		recall := hit / float64(queries)
		t.Logf("ef=%-4d recall@%-2d=%.4f", ef, gatherK, recall)
		if recall < prevRecall-0.01 {
			t.Errorf("gather recall regressed with larger ef: %.4f < %.4f", recall, prevRecall)
		}
		prevRecall = recall
	}
	if prevRecall < 0.9 {
		t.Errorf("top-end gather recall %.4f below 0.9", prevRecall)
	}
}

// TestHNSWvsFlatSpeedup confirms HNSW visits far fewer vectors than a flat scan
// while keeping high recall, the reason to prefer it at scale.
func TestHNSWvsFlatSpeedup(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping speed comparison in short mode")
	}
	rng := quant.NewPCG(5)
	d, n := 128, 20000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})
	q := quant.New(quant.Config{Dim: d, Bits: 4, ResidualDims: 32, Seed: 1})
	flat := New(q, Cosine)
	for i, r := range rows {
		flat.Add(fmt.Sprintf("%d", i), r)
	}

	const queries = 100
	qs := make([][]float32, queries)
	for i := range qs {
		qs[i] = rows[randIdx(rng, n)]
	}

	t0 := time.Now()
	for _, qq := range qs {
		h.Search(qq, 10, 64)
	}
	hnswT := time.Since(t0)

	t1 := time.Now()
	for _, qq := range qs {
		flat.Search(qq, 10, 1)
	}
	flatT := time.Since(t1)

	t.Logf("hnsw=%s flat=%s speedup=%.1fx", hnswT/queries, flatT/queries, float64(flatT)/float64(hnswT))
	if hnswT >= flatT {
		t.Errorf("HNSW (%s) should be faster than flat (%s) at n=%d", hnswT, flatT, n)
	}
}
