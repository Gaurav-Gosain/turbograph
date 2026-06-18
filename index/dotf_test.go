package index

import (
	"math"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// TestDotfMatchesReference checks the build-selected dotf (the AVX kernel on
// amd64) against the portable reference across many lengths, including non
// multiples of 8 and the tail-only cases, so the SIMD path can never silently
// diverge.
func TestDotfMatchesReference(t *testing.T) {
	rng := quant.NewPCG(99)
	for _, n := range []int{0, 1, 3, 7, 8, 9, 15, 16, 31, 32, 33, 64, 100, 127, 768, 1024, 1025} {
		for trial := 0; trial < 8; trial++ {
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
				b[i] = float32(rng.NormFloat64())
			}
			got := dotf(a, b)
			want := dotfGo(a, b)
			tol := 1e-3 * (float32(math.Abs(float64(want))) + 1)
			if d := float32(math.Abs(float64(got - want))); d > tol {
				t.Fatalf("n=%d: dotf=%g dotfGo=%g diff=%g", n, got, want, d)
			}
		}
	}
}

func benchVecs(n int) ([]float32, []float32) {
	rng := quant.NewPCG(1)
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
		b[i] = float32(rng.NormFloat64())
	}
	return a, b
}

func BenchmarkDotfSIMD(b *testing.B) {
	a, c := benchVecs(768)
	b.ResetTimer()
	var s float32
	for i := 0; i < b.N; i++ {
		s += dotf(a, c)
	}
	_ = s
}

func BenchmarkDotfGo(b *testing.B) {
	a, c := benchVecs(768)
	b.ResetTimer()
	var s float32
	for i := 0; i < b.N; i++ {
		s += dotfGo(a, c)
	}
	_ = s
}
