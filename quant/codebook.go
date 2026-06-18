package quant

import (
	"math"
	"sort"
)

// Codebook is an optimal scalar quantizer for the standard normal distribution,
// computed by the Lloyd-Max iteration. After a random rotation the coordinates
// of a unit vector are close to i.i.d. N(0, 1/D); scaled by sqrt(D) they are
// close to N(0,1), which is exactly the distribution this codebook is tuned for.
//
// For the optimal (centroid) quantizer the reconstruction level of a cell is the
// conditional mean of the source inside that cell, which yields two identities
// used throughout the estimators:
//
//	E[Q(X)^2] = E[X*Q(X)]            (reconstruction is the MMSE estimate)
//	MSE = E[(X-Q(X))^2] = 1 - E[Q(X)^2]
type Codebook struct {
	bits   int       // bits per coordinate; number of levels is 1<<bits
	levels []float32 // reconstruction values, ascending
	bounds []float32 // len(levels)-1 ascending decision thresholds
	mse    float64   // expected squared error against N(0,1)
	eq2    float64   // E[Q(X)^2] == E[X*Q(X)]
}

// buildCodebook runs Lloyd-Max for the standard normal and returns the optimal
// b-bit quantizer. b must be in [1, 8].
func buildCodebook(b int) *Codebook {
	if b < 1 || b > 8 {
		panic("quant: bits per coordinate must be in [1,8]")
	}
	const (
		span   = 9.0
		points = 1 << 18
	)
	// Precompute the standard normal density on a fine, symmetric grid.
	dx := 2 * span / float64(points)
	xs := make([]float64, points)
	w := make([]float64, points) // phi(x)*dx, the probability mass of each sample
	norm := 1.0 / math.Sqrt(2*math.Pi)
	for i := 0; i < points; i++ {
		x := -span + (float64(i)+0.5)*dx
		xs[i] = x
		w[i] = norm * math.Exp(-0.5*x*x) * dx
	}

	// Cumulative mass for quantile-based initialization, which starts Lloyd-Max
	// close to the optimum and avoids poor local minima at higher bit counts.
	cum := make([]float64, points)
	acc := 0.0
	for i := 0; i < points; i++ {
		acc += w[i]
		cum[i] = acc
	}
	quantile := func(p float64) float64 {
		target := p * acc
		lo, hi := 0, points-1
		for lo < hi {
			mid := (lo + hi) / 2
			if cum[mid] < target {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		return xs[lo]
	}

	k := 1 << b
	levels := make([]float64, k)
	for j := 0; j < k; j++ {
		levels[j] = quantile((float64(j) + 0.5) / float64(k))
	}

	bounds := make([]float64, k-1)
	for iter := 0; iter < 1000; iter++ {
		for j := 0; j < k-1; j++ {
			bounds[j] = 0.5 * (levels[j] + levels[j+1])
		}
		num := make([]float64, k)
		den := make([]float64, k)
		cell := 0
		for i := 0; i < points; i++ {
			for cell < k-1 && xs[i] > bounds[cell] {
				cell++
			}
			num[cell] += xs[i] * w[i]
			den[cell] += w[i]
		}
		maxShift := 0.0
		for j := 0; j < k; j++ {
			if den[j] > 0 {
				nl := num[j] / den[j]
				if d := math.Abs(nl - levels[j]); d > maxShift {
					maxShift = d
				}
				levels[j] = nl
			}
		}
		if maxShift < 1e-12 {
			break
		}
	}

	// Final distortion statistics.
	var mse, eq2 float64
	cell := 0
	for j := 0; j < k-1; j++ {
		bounds[j] = 0.5 * (levels[j] + levels[j+1])
	}
	for i := 0; i < points; i++ {
		for cell < k-1 && xs[i] > bounds[cell] {
			cell++
		}
		d := xs[i] - levels[cell]
		mse += d * d * w[i]
		eq2 += levels[cell] * levels[cell] * w[i]
	}

	cb := &Codebook{
		bits:   b,
		levels: toF32(levels),
		bounds: toF32(bounds),
		mse:    mse,
		eq2:    eq2,
	}
	return cb
}

// Bits returns the number of bits per coordinate.
func (c *Codebook) Bits() int { return c.bits }

// Levels returns the reconstruction values (read-only).
func (c *Codebook) Levels() []float32 { return c.levels }

// MSE returns the expected squared quantization error against N(0,1).
func (c *Codebook) MSE() float64 { return c.mse }

// Encode maps a standard-normal-scaled value to its codebook index.
func (c *Codebook) Encode(z float32) uint8 {
	// Linear scan is fastest for the small level counts used here (<=256).
	lo := sort.Search(len(c.bounds), func(i int) bool { return c.bounds[i] >= z })
	return uint8(lo)
}

// Decode returns the reconstruction value for a code.
func (c *Codebook) Decode(code uint8) float32 { return c.levels[code] }

func toF32(a []float64) []float32 {
	out := make([]float32, len(a))
	for i, v := range a {
		out[i] = float32(v)
	}
	return out
}
