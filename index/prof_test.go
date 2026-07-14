package index

import (
	"fmt"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

func BenchmarkHNSWSearchProf(b *testing.B) {
	rng := quant.NewPCG(7)
	d, n := 768, 20000
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = randVec(rng, d)
	}
	h := NewHNSW(d, HNSWConfig{M: 16, EfConstruction: 200, Seed: 1})
	for i, r := range rows {
		h.Add(fmt.Sprintf("%d", i), r)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(rows[i%n], 10, 64)
	}
}
