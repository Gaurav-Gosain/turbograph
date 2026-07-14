package index

import (
	"fmt"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

func buildHNSW(t testing.TB, rows [][]float32, d int, cfg HNSWConfig) *HNSW {
	h := NewHNSW(d, cfg)
	for i, r := range rows {
		h.Add(fmt.Sprintf("%d", i), r)
	}
	return h
}

func TestHNSWRecallVsBrute(t *testing.T) {
	rng := quant.NewPCG(1)
	d, n := 128, 6000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 7})
	if h.Len() != n {
		t.Fatalf("len=%d", h.Len())
	}
	var sum float64
	const queries = 100
	for t2 := 0; t2 < queries; t2++ {
		query := rows[randIdx(rng, n)]
		truth := bruteTop(query, rows, 10, Cosine)
		res := h.Search(query, 10, 64)
		sum += recallAt(res, truth, 10)
	}
	recall := sum / queries
	if recall < 0.95 {
		t.Errorf("HNSW recall@10=%.3f below 0.95", recall)
	}
	t.Logf("HNSW recall@10=%.3f", recall)
}

func TestHNSWEfImprovesRecall(t *testing.T) {
	rng := quant.NewPCG(2)
	d, n := 96, 4000
	rows := buildCorpus(rng, n, d)
	h := buildHNSW(t, rows, d, HNSWConfig{M: 12, EfConstruction: 100, Seed: 3})
	measure := func(ef int) float64 {
		var sum float64
		const queries = 60
		for i := 0; i < queries; i++ {
			query := rows[randIdx(rng, n)]
			truth := bruteTop(query, rows, 10, Cosine)
			sum += recallAt(h.Search(query, 10, ef), truth, 10)
		}
		return sum / queries
	}
	lo := measure(10)
	hi := measure(120)
	if hi < lo {
		t.Errorf("higher ef did not help: ef10=%.3f ef120=%.3f", lo, hi)
	}
	t.Logf("recall ef=10:%.3f ef=120:%.3f", lo, hi)
}

func TestHNSWEmptyAndSingle(t *testing.T) {
	h := NewHNSW(16, HNSWConfig{})
	if got := h.Search(make([]float32, 16), 5, 32); got != nil {
		t.Errorf("empty search should be nil, got %v", got)
	}
	h.Add("only", []float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	res := h.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 5, 32)
	if len(res) != 1 || res[0].ID != "only" {
		t.Errorf("single-element search wrong: %v", res)
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	rng := quant.NewPCG(7)
	d, n := 768, 100_000
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = randVec(rng, d)
	}
	h := buildHNSW(b, rows, d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(rows[i%n], 10, 64)
	}
}
