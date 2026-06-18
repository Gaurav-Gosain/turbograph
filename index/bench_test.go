package index

import (
	"fmt"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// benchData builds a quantized index and a parallel slice of raw vectors for a
// fair quantized-vs-float comparison.
func benchData(b *testing.B, n, d, bits int) (*Index, [][]float32) {
	rng := quant.NewPCG(7)
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = randVec(rng, d)
	}
	q := quant.New(quant.Config{Dim: d, Bits: bits, Rounds: 3, ResidualDims: 32, Seed: 1})
	ix := New(q, Cosine)
	flat := make([]float32, 0, n*d)
	ids := make([]string, n)
	for i := range rows {
		flat = append(flat, rows[i]...)
		ids[i] = fmt.Sprintf("%d", i)
	}
	ix.AddBatch(ids, flat, d)
	return ix, rows
}

func BenchmarkSearchQuantized(b *testing.B) {
	for _, n := range []int{100_000} {
		ix, rows := benchData(b, n, 768, 4)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ix.Search(rows[i%n], 10, 1)
			}
		})
	}
}

// BenchmarkSearchBruteFloat is the full-precision baseline: an exact cosine scan
// over float32 vectors, single-threaded like a naive implementation.
func BenchmarkSearchBruteFloat(b *testing.B) {
	n, d := 100_000, 768
	_, rows := benchData(b, n, d, 4)
	flat := make([]float32, 0, n*d)
	norms := make([]float32, n)
	for i := range rows {
		flat = append(flat, rows[i]...)
		norms[i] = vnorm(rows[i])
	}
	b.ResetTimer()
	b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
		for it := 0; it < b.N; it++ {
			q := rows[it%n]
			qn := vnorm(q)
			var bestI int
			var bestS float32 = -1e30
			for i := 0; i < n; i++ {
				v := flat[i*d : i*d+d]
				var s float32
				for j := 0; j < d; j++ {
					s += q[j] * v[j]
				}
				s /= qn * norms[i]
				if s > bestS {
					bestS, bestI = s, i
				}
			}
			_ = bestI
		}
	})
}

func BenchmarkEncode(b *testing.B) {
	rng := quant.NewPCG(1)
	d := 768
	q := quant.New(quant.Config{Dim: d, Bits: 4, Rounds: 3, ResidualDims: 32, Seed: 1})
	v := randVec(rng, d)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q.Encode(v)
	}
}
