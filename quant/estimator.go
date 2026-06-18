package quant

import "math"

// sqrtHalfPi is sqrt(pi/2), the QJL inner-product de-biasing constant.
const sqrtHalfPi = 1.2533141373155003

// Query is a prepared query vector. Building it once amortizes the rotation and
// the per-coordinate lookup-table construction across a whole index scan.
//
// The main inner-product term is evaluated as an asymmetric distance computation
// (ADC): for code c, sum_i qlut[i*k + c_i] is exactly sum_i rotated_query_i *
// level(c_i). The query stays in full precision; only the database is quantized.
//
// Two estimators are exposed because they have different bias/variance tradeoffs:
//
//   - Score is the low-variance main term. It is slightly biased in scale but
//     order-preserving, which makes it the right choice for ranking and
//     candidate generation in nearest-neighbor search.
//   - IP/L2/Cosine add the 1-bit QJL residual correction (when the quantizer was
//     built with ResidualDims > 0). That removes the bias so absolute magnitudes
//     are accurate, at the cost of the variance the sketch introduces.
type Query struct {
	q      *Quantizer
	qrot   []float32 // rotated query, length Dim()
	qnorm2 float32   // squared norm of the query
	qlut   []float32 // Dim() * (1<<bits) lookup table
	projq  []float32 // m precomputed query projections <g_j, qrot>
	k      int
}

// PrepareQuery rotates q and builds its lookup tables.
func (qz *Quantizer) PrepareQuery(q []float32) *Query {
	qrot := make([]float32, qz.dim)
	qz.rot.Apply(qrot, q)
	var n2 float32
	for _, v := range qrot {
		n2 += v * v
	}
	k := 1 << qz.bits
	lut := make([]float32, qz.dim*k)
	levels := qz.cb.levels
	for i := 0; i < qz.dim; i++ {
		base := i * k
		qi := qrot[i]
		for c := 0; c < k; c++ {
			lut[base+c] = qi * levels[c]
		}
	}
	out := &Query{q: qz, qrot: qrot, qnorm2: n2, qlut: lut, k: k}
	if qz.m > 0 {
		projq := make([]float32, qz.m)
		for j := 0; j < qz.m; j++ {
			pv := qz.proj[j*qz.dim : (j+1)*qz.dim]
			var dot float32
			for i, qi := range qrot {
				dot += pv[i] * qi
			}
			projq[j] = dot
		}
		out.projq = projq
	}
	return out
}

// rawMain returns sum_i rotated_query_i * level(code_i): the asymmetric dot
// product between the full-precision query and the quantized reconstruction.
func (qr *Query) rawMain(c Code) float32 {
	lut := qr.qlut
	k := qr.k
	var main float32
	codes := c.Codes
	for i := 0; i < len(codes); i++ {
		main += lut[i*k+int(codes[i])]
	}
	return main
}

// residual evaluates the QJL residual correction <query, x_residual>/||x||.
func (qr *Query) residual(c Code) float32 {
	var acc float32
	signs := c.Signs
	for j := 0; j < qr.q.m; j++ {
		if signs&(1<<uint(j)) != 0 {
			acc += qr.projq[j]
		} else {
			acc -= qr.projq[j]
		}
	}
	return acc * float32(sqrtHalfPi) / float32(qr.q.m) * c.ResNorm * qr.q.invSqrtD
}

// unitScore is the low-variance directional estimate of <query, x>/||x||.
func (qr *Query) unitScore(c Code) float32 {
	return qr.rawMain(c) * qr.q.invSqrtD * qr.q.debias
}

// unitIP is the unbiased directional estimate of <query, x>/||x||.
func (qr *Query) unitIP(c Code) float32 {
	ip := qr.rawMain(c) * qr.q.invSqrtD
	if qr.q.m == 0 {
		return ip * qr.q.debias
	}
	return ip + qr.residual(c)
}

// Score returns a low-variance, order-preserving estimate of <query, x>. Use it
// for ranking and candidate generation; it is the fastest path and the most
// accurate for ordering.
func (qr *Query) Score(c Code) float32 {
	return c.Norm * qr.unitScore(c)
}

// CosineScore returns a low-variance, order-preserving score whose order matches
// cosine similarity. It omits the query-norm factor (constant across a scan), so
// it is proportional to cosine and ideal for ranking by direction.
func (qr *Query) CosineScore(c Code) float32 {
	return qr.unitScore(c)
}

// IP returns an unbiased estimate of the inner product <query, x>.
func (qr *Query) IP(c Code) float32 {
	return c.Norm * qr.unitIP(c)
}

// L2 returns an unbiased estimate of the squared Euclidean distance
// ||query - x||^2.
func (qr *Query) L2(c Code) float32 {
	ip := c.Norm * qr.unitIP(c)
	d := qr.qnorm2 + c.Norm*c.Norm - 2*ip
	if d < 0 {
		d = 0
	}
	return d
}

// L2Score returns a low-variance, order-preserving score whose ascending order
// matches squared Euclidean distance. Use it for nearest-neighbor ranking.
func (qr *Query) L2Score(c Code) float32 {
	ip := c.Norm * qr.unitScore(c)
	return qr.qnorm2 + c.Norm*c.Norm - 2*ip
}

// Cosine estimates the cosine similarity between the query and x.
func (qr *Query) Cosine(c Code) float32 {
	qn := float32(math.Sqrt(float64(qr.qnorm2)))
	if qn == 0 || c.Norm == 0 {
		return 0
	}
	return qr.unitIP(c) / qn
}
