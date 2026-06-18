package quant

import (
	"math"
	"math/bits"
)

// Rotation is an orthonormal, data-oblivious transform built from alternating
// Rademacher sign flips and fast Walsh-Hadamard transforms (an SRHT-style
// operator). Applied to a vector it spreads energy evenly across coordinates so
// that, for a unit input, each output coordinate behaves like an independent
// zero-mean Gaussian with variance 1/D. That is the property TurboQuant relies
// on to make per-coordinate scalar quantization near optimal.
//
// The transform is orthonormal, so it preserves both norms and inner products
// exactly. The same Rotation must be applied to database and query vectors for
// asymmetric distance estimation to be consistent.
type Rotation struct {
	dim    int // padded dimension, a power of two
	in     int // original input dimension
	rounds int // number of sign-flip + Hadamard passes
	signs  [][]float32
}

// NewRotation builds a rotation for inputs of dimension inDim. Vectors are
// zero-padded to the next power of two. rounds controls mixing quality; three
// passes is sufficient to make coordinates close to Gaussian for any input.
func NewRotation(inDim, rounds int, rng *PCG) *Rotation {
	if inDim <= 0 {
		panic("quant: rotation dimension must be positive")
	}
	if rounds <= 0 {
		rounds = 3
	}
	dim := nextPow2(inDim)
	r := &Rotation{dim: dim, in: inDim, rounds: rounds, signs: make([][]float32, rounds)}
	for k := range r.signs {
		s := make([]float32, dim)
		for i := range s {
			if rng.Uint64()&1 == 0 {
				s[i] = 1
			} else {
				s[i] = -1
			}
		}
		r.signs[k] = s
	}
	return r
}

// Dim returns the padded working dimension (a power of two).
func (r *Rotation) Dim() int { return r.dim }

// InputDim returns the original, unpadded input dimension.
func (r *Rotation) InputDim() int { return r.in }

// Apply writes the rotated image of src into dst. src may be shorter than the
// padded dimension; it is zero-extended. dst must have length Dim().
func (r *Rotation) Apply(dst, src []float32) {
	if len(dst) != r.dim {
		panic("quant: rotation dst length mismatch")
	}
	for i := 0; i < r.dim; i++ {
		if i < len(src) {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
	norm := float32(1.0 / math.Sqrt(float64(r.dim)))
	for k := 0; k < r.rounds; k++ {
		s := r.signs[k]
		for i := 0; i < r.dim; i++ {
			dst[i] *= s[i]
		}
		fwht(dst)
		for i := 0; i < r.dim; i++ {
			dst[i] *= norm
		}
	}
}

// applyInverse maps a rotated vector back to the original frame in place. Because
// each pass is sign-flip then Hadamard, and the normalized Hadamard is its own
// inverse, the inverse replays the passes in reverse: Hadamard then sign-flip.
func (r *Rotation) applyInverse(v []float32) {
	if len(v) != r.dim {
		panic("quant: rotation inverse length mismatch")
	}
	norm := float32(1.0 / math.Sqrt(float64(r.dim)))
	for k := r.rounds - 1; k >= 0; k-- {
		fwht(v)
		for i := 0; i < r.dim; i++ {
			v[i] *= norm
		}
		s := r.signs[k]
		for i := 0; i < r.dim; i++ {
			v[i] *= s[i]
		}
	}
}

// fwht performs an in-place, unnormalized fast Walsh-Hadamard transform. The
// length of a must be a power of two. Cost is O(n log n).
func fwht(a []float32) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j] = x + y
				a[j+h] = x - y
			}
		}
	}
}

func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	return 1 << bits.Len(uint(n-1))
}
