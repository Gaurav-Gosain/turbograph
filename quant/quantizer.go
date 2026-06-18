package quant

import "math"

// Config parameterizes a Quantizer.
type Config struct {
	Dim          int    // input vector dimension
	Bits         int    // bits per coordinate, 1..8
	Rounds       int    // rotation mixing passes (default 3)
	ResidualDims int    // QJL residual projections m, 0..64; 0 disables debiasing
	Seed         uint64 // determinism seed
}

// Quantizer implements TurboQuant: a data-oblivious vector quantizer that rotates
// inputs to an isotropic frame, applies an optimal per-coordinate scalar
// quantizer, and (optionally) captures a 1-bit QJL sketch of the quantization
// residual so inner products can be estimated without bias.
//
// A single Quantizer is immutable after construction and safe for concurrent use
// by encoders and queries.
type Quantizer struct {
	rot      *Rotation
	cb       *Codebook
	dim      int
	bits     int
	m        int
	proj     []float32 // m*dim flattened Rademacher projection vectors
	invSqrtD float32
	debias   float32 // main-term correction when m == 0
}

// Code is the compressed representation of one vector.
type Code struct {
	Codes   []uint8 // per-coordinate codebook indices, length Dim()
	Norm    float32 // original Euclidean norm
	ResNorm float32 // residual norm in the standardized (z) space
	Signs   uint64  // QJL residual sign bits, one per projection
}

// New builds a Quantizer from cfg.
func New(cfg Config) *Quantizer {
	if cfg.Bits < 1 || cfg.Bits > 8 {
		panic("quant: Bits must be in [1,8]")
	}
	if cfg.ResidualDims < 0 || cfg.ResidualDims > 64 {
		panic("quant: ResidualDims must be in [0,64]")
	}
	rng := NewPCG(cfg.Seed)
	rot := NewRotation(cfg.Dim, cfg.Rounds, rng)
	cb := buildCodebook(cfg.Bits)
	d := rot.Dim()
	q := &Quantizer{
		rot:      rot,
		cb:       cb,
		dim:      d,
		bits:     cfg.Bits,
		m:        cfg.ResidualDims,
		invSqrtD: float32(1.0 / math.Sqrt(float64(d))),
		debias:   float32(1.0 / cb.eq2),
	}
	if q.m > 0 {
		q.proj = make([]float32, q.m*d)
		for i := range q.proj {
			if rng.Uint64()&1 == 0 {
				q.proj[i] = 1
			} else {
				q.proj[i] = -1
			}
		}
	}
	return q
}

// Dim returns the padded working dimension.
func (q *Quantizer) Dim() int { return q.dim }

// Bits returns bits per coordinate.
func (q *Quantizer) Bits() int { return q.bits }

// Codebook exposes the underlying scalar quantizer (read-only use).
func (q *Quantizer) Codebook() *Codebook { return q.cb }

// Rotation exposes the underlying rotation (read-only use).
func (q *Quantizer) Rotation() *Rotation { return q.rot }

// Encode compresses a vector. The returned Code owns its Codes slice.
func (q *Quantizer) Encode(x []float32) Code {
	buf := make([]float32, q.dim)
	res := make([]float32, q.dim)
	codes := make([]uint8, q.dim)
	c := Code{Codes: codes}
	q.EncodeInto(x, buf, res, &c)
	return c
}

// EncodeInto encodes x reusing caller-provided scratch buffers (buf and res must
// have length Dim(), and c.Codes must too). It performs no allocation, which
// matters when encoding large corpora.
func (q *Quantizer) EncodeInto(x, buf, res []float32, c *Code) {
	q.rot.Apply(buf, x)
	var n2 float64
	for _, v := range buf {
		n2 += float64(v) * float64(v)
	}
	norm := math.Sqrt(n2)
	c.Norm = float32(norm)
	if norm == 0 {
		for i := range c.Codes {
			c.Codes[i] = q.cb.Encode(0)
		}
		c.ResNorm = 0
		c.Signs = 0
		return
	}
	// Standardize: z_i = sqrt(D) * rotated_unit_i ~ N(0,1).
	scale := q.invSqrtD // = 1/sqrt(D); z = buf/norm * sqrt(D) = buf * (sqrt(D)/norm)
	zscale := float32(1.0) / (scale * float32(norm))
	var resN2 float64
	for i, v := range buf {
		z := v * zscale
		code := q.cb.Encode(z)
		c.Codes[i] = code
		r := z - q.cb.levels[code]
		res[i] = r
		resN2 += float64(r) * float64(r)
	}
	c.ResNorm = float32(math.Sqrt(resN2))
	c.Signs = 0
	if q.m > 0 {
		for j := 0; j < q.m; j++ {
			pv := q.proj[j*q.dim : (j+1)*q.dim]
			var dot float32
			for i, r := range res {
				dot += pv[i] * r
			}
			if dot >= 0 {
				c.Signs |= 1 << uint(j)
			}
		}
	}
}

// EncodeBatch encodes many vectors with shared scratch buffers. rows is a flat
// row-major matrix with stride equal to the input dimension.
func (q *Quantizer) EncodeBatch(rows []float32, n, stride int) []Code {
	out := make([]Code, n)
	buf := make([]float32, q.dim)
	res := make([]float32, q.dim)
	for i := 0; i < n; i++ {
		out[i].Codes = make([]uint8, q.dim)
		q.EncodeInto(rows[i*stride:i*stride+stride], buf, res, &out[i])
	}
	return out
}

// Decode reconstructs an approximation of the original vector (in the original,
// unrotated frame, truncated to the input dimension).
func (q *Quantizer) Decode(c Code) []float32 {
	// Reconstruct rotated unit vector, rescale by norm, invert the rotation.
	y := make([]float32, q.dim)
	s := c.Norm * q.invSqrtD
	for i, code := range c.Codes {
		y[i] = q.cb.levels[code] * s
	}
	// The rotation is orthonormal and symmetric per pass only up to sign order;
	// invert by replaying passes in reverse. Each pass is its own inverse up to
	// the sign diagonal, so we apply signs and Hadamard in reverse order.
	q.rot.applyInverse(y)
	return y[:q.rot.in]
}
