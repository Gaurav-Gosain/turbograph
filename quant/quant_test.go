package quant

import (
	"math"
	"testing"
)

// knownGaussianMSE holds the optimal (Lloyd-Max) scalar-quantizer distortion for
// a standard normal source at 1..5 bits. These are textbook values and pin down
// the codebook construction.
var knownGaussianMSE = map[int]float64{
	1: 0.363380,
	2: 0.117482,
	3: 0.034539,
	4: 0.009497,
	5: 0.002499,
}

func TestCodebookMatchesKnownDistortion(t *testing.T) {
	for b, want := range knownGaussianMSE {
		cb := buildCodebook(b)
		if rel := math.Abs(cb.mse-want) / want; rel > 0.01 {
			t.Errorf("bits=%d mse=%.6f want=%.6f rel=%.3f", b, cb.mse, want, rel)
		}
		// Centroid-quantizer identity: MSE == 1 - E[Q^2] == 1 - E[X Q].
		if d := math.Abs(cb.mse - (1 - cb.eq2)); d > 1e-6 {
			t.Errorf("bits=%d identity broken: mse=%.6f 1-eq2=%.6f", b, cb.mse, 1-cb.eq2)
		}
	}
}

func TestCodebookSymmetryAndOrder(t *testing.T) {
	for b := 1; b <= 6; b++ {
		cb := buildCodebook(b)
		for i := 1; i < len(cb.levels); i++ {
			if cb.levels[i] <= cb.levels[i-1] {
				t.Fatalf("bits=%d levels not ascending at %d", b, i)
			}
		}
		// Symmetric distribution => levels symmetric about zero.
		k := len(cb.levels)
		for i := 0; i < k/2; i++ {
			if d := math.Abs(float64(cb.levels[i] + cb.levels[k-1-i])); d > 1e-3 {
				t.Errorf("bits=%d asymmetric levels %d: %.4f vs %.4f", b, i, cb.levels[i], cb.levels[k-1-i])
			}
		}
	}
}

func randVec(rng *PCG, d int) []float32 {
	v := make([]float32, d)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm(a []float32) float32 { return float32(math.Sqrt(float64(dot(a, a)))) }

func TestRotationPreservesGeometry(t *testing.T) {
	rng := NewPCG(7)
	rot := NewRotation(300, 3, rng)
	for trial := 0; trial < 50; trial++ {
		a := randVec(rng, 300)
		b := randVec(rng, 300)
		ra := make([]float32, rot.Dim())
		rb := make([]float32, rot.Dim())
		rot.Apply(ra, a)
		rot.Apply(rb, b)
		if rel := math.Abs(float64(norm(ra)-norm(a))) / float64(norm(a)); rel > 1e-4 {
			t.Fatalf("norm not preserved: rel=%g", rel)
		}
		if rel := math.Abs(float64(dot(ra, rb)-dot(a, b))) / (float64(norm(a)*norm(b)) + 1e-9); rel > 1e-4 {
			t.Fatalf("inner product not preserved: rel=%g", rel)
		}
	}
}

func TestRotationInverse(t *testing.T) {
	rng := NewPCG(11)
	rot := NewRotation(200, 3, rng)
	a := randVec(rng, 200)
	r := make([]float32, rot.Dim())
	rot.Apply(r, a)
	rot.applyInverse(r)
	for i := 0; i < 200; i++ {
		if d := math.Abs(float64(r[i] - a[i])); d > 1e-3 {
			t.Fatalf("inverse mismatch at %d: %.5f vs %.5f", i, r[i], a[i])
		}
	}
}

// TestRotationSpreadsEnergy checks the central TurboQuant premise: a spiky input
// becomes approximately isotropic after rotation, so its standardized
// coordinates have unit variance and near-zero excess concentration.
func TestRotationSpreadsEnergy(t *testing.T) {
	rng := NewPCG(3)
	d := 1024
	rot := NewRotation(d, 3, rng)
	x := make([]float32, d)
	x[0] = 1 // maximally spiky unit vector
	r := make([]float32, rot.Dim())
	rot.Apply(r, x)
	var max float32
	for _, v := range r {
		if av := float32(math.Abs(float64(v))); av > max {
			max = av
		}
	}
	// Energy spread evenly => no coordinate should dominate. Uniform spread is
	// 1/sqrt(D); allow generous slack.
	if max > 8.0/float32(math.Sqrt(float64(d))) {
		t.Fatalf("rotation failed to spread energy: max coord %.4f", max)
	}
}

func TestEncodeDecodeReconstruction(t *testing.T) {
	rng := NewPCG(42)
	d := 512
	prev := math.Inf(1)
	for _, b := range []int{1, 2, 3, 4} {
		q := New(Config{Dim: d, Bits: b, Rounds: 3, Seed: 1})
		var num, den float64
		for trial := 0; trial < 30; trial++ {
			x := randVec(rng, d)
			c := q.Encode(x)
			xh := q.Decode(c)
			for i := range x {
				e := float64(x[i] - xh[i])
				num += e * e
				den += float64(x[i]) * float64(x[i])
			}
		}
		rel := num / den
		if rel >= prev {
			t.Errorf("bits=%d relerr=%.4f did not improve on previous %.4f", b, rel, prev)
		}
		prev = rel
	}
}

// TestInnerProductUnbiased verifies the QJL-corrected estimator is unbiased and
// low-error against brute-force inner products.
func TestInnerProductUnbiased(t *testing.T) {
	rng := NewPCG(99)
	d := 768
	q := New(Config{Dim: d, Bits: 4, Rounds: 3, ResidualDims: 32, Seed: 5})
	const trials = 400
	var sumErr, sumAbs, sumTrueSq float64
	for tr := 0; tr < trials; tr++ {
		x := randVec(rng, d)
		y := randVec(rng, d)
		// Correlate y with x so inner products span a realistic range.
		for i := range y {
			y[i] += 0.5 * x[i]
		}
		c := q.Encode(y)
		qq := q.PrepareQuery(x)
		est := qq.IP(c)
		tru := dot(x, y)
		sumErr += float64(est - tru)
		sumAbs += math.Abs(float64(est - tru))
		sumTrueSq += float64(tru) * float64(tru)
	}
	bias := sumErr / trials
	rmsScale := math.Sqrt(sumTrueSq / trials)
	relBias := math.Abs(bias) / rmsScale
	relMAE := (sumAbs / trials) / rmsScale
	if relBias > 0.05 {
		t.Errorf("IP estimator biased: relBias=%.4f", relBias)
	}
	if relMAE > 0.20 {
		t.Errorf("IP estimator inaccurate: relMAE=%.4f", relMAE)
	}
	t.Logf("relBias=%.4f relMAE=%.4f", relBias, relMAE)
}

func TestL2Estimator(t *testing.T) {
	rng := NewPCG(123)
	d := 512
	q := New(Config{Dim: d, Bits: 4, Rounds: 3, ResidualDims: 32, Seed: 2})
	var sumAbs, sumTrue float64
	for tr := 0; tr < 200; tr++ {
		x := randVec(rng, d)
		y := randVec(rng, d)
		c := q.Encode(y)
		qq := q.PrepareQuery(x)
		est := qq.L2(c)
		var tru float32
		for i := range x {
			dd := x[i] - y[i]
			tru += dd * dd
		}
		sumAbs += math.Abs(float64(est - tru))
		sumTrue += float64(tru)
	}
	if rel := sumAbs / sumTrue; rel > 0.15 {
		t.Errorf("L2 estimator inaccurate: rel=%.4f", rel)
	}
}

func TestDeterminism(t *testing.T) {
	d := 256
	x := randVec(NewPCG(1), d)
	q1 := New(Config{Dim: d, Bits: 4, ResidualDims: 16, Seed: 77})
	q2 := New(Config{Dim: d, Bits: 4, ResidualDims: 16, Seed: 77})
	c1 := q1.Encode(x)
	c2 := q2.Encode(x)
	if c1.Norm != c2.Norm || c1.Signs != c2.Signs {
		t.Fatal("nondeterministic encode")
	}
	for i := range c1.Codes {
		if c1.Codes[i] != c2.Codes[i] {
			t.Fatalf("nondeterministic codes at %d", i)
		}
	}
}

// TestRankingRecall is the property that matters for search: top-k by estimated
// inner product should mostly match top-k by exact inner product.
func TestRankingRecall(t *testing.T) {
	rng := NewPCG(2024)
	d := 384
	n := 2000
	q := New(Config{Dim: d, Bits: 4, Rounds: 3, ResidualDims: 32, Seed: 9})
	rows := make([][]float32, n)
	codes := make([]Code, n)
	for i := 0; i < n; i++ {
		rows[i] = randVec(rng, d)
		codes[i] = q.Encode(rows[i])
	}
	// Plant neighborhood structure: vectors carry a variable amount of the query
	// direction, so the true top-k is well separated rather than noise-dominated
	// (as it would be for purely independent Gaussians).
	query := randVec(rng, d)
	qn := norm(query)
	for i := 0; i < n; i++ {
		g := float32(rng.Float64()) * 2.5
		for j := range rows[i] {
			rows[i][j] += g * query[j] / qn
		}
		codes[i] = q.Encode(rows[i])
	}
	qq := q.PrepareQuery(query)

	exact := make([]pair, n)
	approx := make([]pair, n)
	for i := 0; i < n; i++ {
		exact[i] = pair{i, dot(query, rows[i])}
		approx[i] = pair{i, qq.Score(codes[i])}
	}
	// Candidate-generation recall: the true top-20 should be recoverable from the
	// approximate top-100. This is exactly how the quantizer is used in practice,
	// with an exact re-rank over the short candidate list.
	const topk, window = 20, 100
	exTop := topKSet(exact, topk)
	apWin := topKSet(approx, window)
	hit := 0
	for i := range exTop {
		if apWin[i] {
			hit++
		}
	}
	recall := float64(hit) / float64(topk)
	if recall < 0.95 {
		t.Errorf("recall@%d within top-%d too low: %.2f", topk, window, recall)
	}
	t.Logf("recall@%d within top-%d = %.2f", topk, window, recall)
}

type pair struct {
	i int
	s float32
}

func topKSet(p []pair, k int) map[int]bool {
	// simple selection of top k by score
	cp := make([]pair, len(p))
	copy(cp, p)
	for a := 0; a < k; a++ {
		best := a
		for b := a + 1; b < len(cp); b++ {
			if cp[b].s > cp[best].s {
				best = b
			}
		}
		cp[a], cp[best] = cp[best], cp[a]
	}
	out := make(map[int]bool, k)
	for a := 0; a < k; a++ {
		out[cp[a].i] = true
	}
	return out
}
