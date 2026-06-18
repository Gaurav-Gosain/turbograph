package quant

import "math"

// PCG is a small, fast, deterministic pseudo-random generator (xoshiro256**
// seeded through splitmix64). It exists so the data-oblivious randomness used by
// the quantizer (sign flips, JL projections) is reproducible from a single seed
// and independent of the global math/rand state.
type PCG struct {
	s [4]uint64
}

// NewPCG seeds a generator from a 64-bit seed.
func NewPCG(seed uint64) *PCG {
	p := &PCG{}
	sm := seed
	for i := range p.s {
		sm += 0x9E3779B97F4A7C15
		z := sm
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		p.s[i] = z ^ (z >> 31)
	}
	return p
}

// Uint64 returns the next 64-bit value.
func (p *PCG) Uint64() uint64 {
	rotl := func(x uint64, k uint) uint64 { return (x << k) | (x >> (64 - k)) }
	result := rotl(p.s[1]*5, 7) * 9
	t := p.s[1] << 17
	p.s[2] ^= p.s[0]
	p.s[3] ^= p.s[1]
	p.s[1] ^= p.s[2]
	p.s[0] ^= p.s[3]
	p.s[2] ^= t
	p.s[3] = rotl(p.s[3], 45)
	return result
}

// Float64 returns a value uniformly distributed in [0, 1).
func (p *PCG) Float64() float64 {
	return float64(p.Uint64()>>11) * (1.0 / 9007199254740992.0)
}

// NormFloat64 returns a standard normal sample via the Box-Muller transform.
func (p *PCG) NormFloat64() float64 {
	u1 := p.Float64()
	if u1 < 1e-300 {
		u1 = 1e-300
	}
	u2 := p.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}
