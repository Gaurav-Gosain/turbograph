package index

import (
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

func randIdx(rng *quant.PCG, n int) int {
	return int(rng.Uint64() >> 1 % uint64(n))
}

func randVec(rng *quant.PCG, d int) []float32 {
	v := make([]float32, d)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func l2(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i] - b[i])
		s += d * d
	}
	return s
}

// bruteTop returns ordinals of the true top-k under the metric.
func bruteTop(query []float32, rows [][]float32, k int, metric Metric) []int {
	type sc struct {
		i int
		s float64
	}
	all := make([]sc, len(rows))
	for i, r := range rows {
		switch metric {
		case L2:
			all[i] = sc{i, l2(query, r)}
		case Cosine:
			den := math.Sqrt(dot(query, query)) * math.Sqrt(dot(r, r))
			if den == 0 {
				all[i] = sc{i, 0}
			} else {
				all[i] = sc{i, dot(query, r) / den}
			}
		default:
			all[i] = sc{i, dot(query, r)}
		}
	}
	if metric == L2 {
		sort.Slice(all, func(a, b int) bool { return all[a].s < all[b].s })
	} else {
		sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].i
	}
	return out
}

func buildCorpus(rng *quant.PCG, n, d int) [][]float32 {
	// Clustered data so neighborhoods are well defined, like real embeddings.
	const clusters = 25
	centers := make([][]float32, clusters)
	for c := range centers {
		centers[c] = randVec(rng, d)
	}
	rows := make([][]float32, n)
	for i := range rows {
		c := centers[randIdx(rng, clusters)]
		v := make([]float32, d)
		for j := range v {
			v[j] = c[j] + 0.35*float32(rng.NormFloat64())
		}
		rows[i] = v
	}
	return rows
}

func recallAt(approx []Result, truth []int, k int) float64 {
	set := make(map[string]bool, len(approx))
	for _, r := range approx {
		set[r.ID] = true
	}
	hit := 0
	for i := 0; i < k && i < len(truth); i++ {
		if set[fmt.Sprintf("%d", truth[i])] {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

func TestIndexRecall(t *testing.T) {
	rng := quant.NewPCG(1)
	d, n := 256, 5000
	rows := buildCorpus(rng, n, d)
	for _, metric := range []Metric{Cosine, InnerProduct, L2} {
		q := quant.New(quant.Config{Dim: d, Bits: 4, Rounds: 3, ResidualDims: 32, Seed: 7})
		ix := NewWithVectors(q, metric, d)
		for i, r := range rows {
			ix.Add(fmt.Sprintf("%d", i), r)
		}
		var sum float64
		const queries = 50
		for t2 := 0; t2 < queries; t2++ {
			query := rows[randIdx(rng, n)]
			truth := bruteTop(query, rows, 10, metric)
			res := ix.Search(query, 10, 10)
			sum += recallAt(res, truth, 10)
		}
		recall := sum / queries
		if recall < 0.90 {
			t.Errorf("metric=%d recall@10=%.3f below 0.90", metric, recall)
		}
		t.Logf("metric=%d recall@10=%.3f", metric, recall)
	}
}

func TestOverscanImprovesRecall(t *testing.T) {
	rng := quant.NewPCG(2)
	d, n := 256, 4000
	rows := buildCorpus(rng, n, d)
	q := quant.New(quant.Config{Dim: d, Bits: 3, Rounds: 3, ResidualDims: 32, Seed: 7})
	ix := NewWithVectors(q, Cosine, d)
	for i, r := range rows {
		ix.Add(fmt.Sprintf("%d", i), r)
	}
	measure := func(overscan int) float64 {
		var sum float64
		const queries = 50
		for t2 := 0; t2 < queries; t2++ {
			query := rows[randIdx(rng, n)]
			truth := bruteTop(query, rows, 10, Cosine)
			res := ix.Search(query, 10, overscan)
			sum += recallAt(res, truth, 10)
		}
		return sum / queries
	}
	low := measure(1)
	high := measure(20)
	if high < low {
		t.Errorf("overscan did not help: low=%.3f high=%.3f", low, high)
	}
	t.Logf("recall overscan=1: %.3f, overscan=20: %.3f", low, high)
}

func TestBatchMatchesSerial(t *testing.T) {
	rng := quant.NewPCG(3)
	d, n := 128, 200
	rows := buildCorpus(rng, n, d)
	q := quant.New(quant.Config{Dim: d, Bits: 4, ResidualDims: 16, Seed: 7})

	serial := New(q, Cosine)
	flat := make([]float32, 0, n*d)
	ids := make([]string, n)
	for i, r := range rows {
		serial.Add(fmt.Sprintf("%d", i), r)
		flat = append(flat, r...)
		ids[i] = fmt.Sprintf("%d", i)
	}
	batch := New(q, Cosine)
	batch.AddBatch(ids, flat, d)

	if serial.Len() != batch.Len() {
		t.Fatal("length mismatch")
	}
	query := rows[0]
	rs := serial.Search(query, 5, 5)
	rb := batch.Search(query, 5, 5)
	for i := range rs {
		if rs[i].ID != rb[i].ID {
			t.Errorf("batch/serial differ at %d: %s vs %s", i, rs[i].ID, rb[i].ID)
		}
		if math.Abs(float64(rs[i].Score-rb[i].Score)) > 1e-5 {
			t.Errorf("batch/serial score differ at %d", i)
		}
	}
}
