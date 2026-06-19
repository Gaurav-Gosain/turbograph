package quant

import (
	"math"
	"sort"
	"time"
)

// BenchOptions configures a synthetic benchmark of a quantizer configuration.
// Zero fields take sensible defaults.
type BenchOptions struct {
	N        int    // database vectors (default 5000)
	Queries  int    // query vectors (default 100)
	TopK     int    // recall cutoff (default 10)
	Clusters int    // cluster centers, for realistic structure (default 16)
	Seed     uint64 // determinism
}

func (o *BenchOptions) withDefaults() {
	if o.N <= 0 {
		o.N = 5000
	}
	if o.Queries <= 0 {
		o.Queries = 100
	}
	if o.TopK <= 0 {
		o.TopK = 10
	}
	if o.Clusters <= 0 {
		o.Clusters = 16
	}
}

// BenchResult reports the accuracy, size, and speed tradeoff of a quantizer
// configuration. The accuracy and size fields are deterministic for a fixed
// config and seed; the throughput fields depend on the host.
type BenchResult struct {
	Dim, Bits, ResidualDims int
	CodeBytes               int     // bytes to store one encoded vector (packed)
	CompressionRatio        float64 // float32 input bytes / code bytes
	RecallAtK               float64 // quantized Score ranking vs exact float ranking
	CodebookMSE             float64 // textbook Lloyd-Max distortion of the codebook
	EncodeVecsPerSec        float64
	QueryScoresPerSec       float64
}

// Benchmark builds a quantizer for cfg and measures, on synthetic clustered unit
// vectors: recall@TopK of the low-variance Score ranking against the exact
// float ranking, the packed code size and compression ratio, and encode/query
// throughput. It is the basis of the `turbograph quant bench` tool and is usable
// directly as a library call.
func Benchmark(cfg Config, opt BenchOptions) BenchResult {
	opt.withDefaults()
	rng := NewPCG(opt.Seed ^ 0x9e3779b97f4a7c15)
	q := New(cfg)
	d := cfg.Dim

	centers := make([][]float32, opt.Clusters)
	for i := range centers {
		centers[i] = randUnit(d, rng)
	}
	db := make([][]float32, opt.N)
	for i := range db {
		db[i] = perturbUnit(centers[i%opt.Clusters], 0.35, rng)
	}
	queries := make([][]float32, opt.Queries)
	for i := range queries {
		queries[i] = perturbUnit(centers[i%opt.Clusters], 0.35, rng)
	}

	t0 := time.Now()
	codes := make([]Code, opt.N)
	for i := range db {
		codes[i] = q.Encode(db[i])
	}
	encDur := time.Since(t0)

	// Measure only the quantized scoring loop: exact search, ranking, and recall
	// bookkeeping are computed outside the timed window so QueryScoresPerSec
	// reflects scoring throughput rather than the harness around it.
	var hit, total int
	var scoreNanos int64
	scores := make([]float32, opt.N)
	for _, qv := range queries {
		exact := topKExact(db, qv, opt.TopK)
		qr := q.PrepareQuery(qv)
		t := time.Now()
		for i := range codes {
			scores[i] = qr.Score(codes[i])
		}
		scoreNanos += time.Since(t).Nanoseconds()
		approx := topKFromScores(scores, opt.TopK)
		hit += overlapCount(exact, approx)
		total += opt.TopK
	}
	scoreSecs := float64(scoreNanos) / 1e9

	codeBytes := (q.Dim()*cfg.Bits+7)/8 + 4 // packed codes + stored norm
	if cfg.ResidualDims > 0 {
		codeBytes += 12 // residual norm (4) + sign bits (8)
	}
	return BenchResult{
		Dim:               d,
		Bits:              cfg.Bits,
		ResidualDims:      cfg.ResidualDims,
		CodeBytes:         codeBytes,
		CompressionRatio:  float64(d*4) / float64(codeBytes),
		RecallAtK:         float64(hit) / float64(total),
		CodebookMSE:       q.Codebook().MSE(),
		EncodeVecsPerSec:  float64(opt.N) / encDur.Seconds(),
		QueryScoresPerSec: float64(opt.Queries*opt.N) / scoreSecs,
	}
}

func randUnit(d int, rng *PCG) []float32 {
	v := make([]float32, d)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return normalize(v)
}

func perturbUnit(center []float32, noise float64, rng *PCG) []float32 {
	v := make([]float32, len(center))
	for i := range v {
		v[i] = center[i] + float32(noise*rng.NormFloat64())
	}
	return normalize(v)
}

func normalize(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		v[0] = 1
		return v
	}
	inv := float32(1 / math.Sqrt(n))
	for i := range v {
		v[i] *= inv
	}
	return v
}

type scored struct {
	id  int
	val float32
}

func topKExact(db [][]float32, q []float32, k int) []int {
	s := make([]scored, len(db))
	for i, v := range db {
		var dot float32
		for j := range v {
			dot += v[j] * q[j]
		}
		s[i] = scored{i, dot}
	}
	return topIDs(s, k)
}

// topKFromScores returns the indices of the k highest scores using a single
// partial selection (O(n*k) for small k) rather than a full O(n log n) sort.
func topKFromScores(scores []float32, k int) []int {
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]int, 0, k)
	taken := make([]bool, len(scores))
	for range k {
		best, bestVal := -1, float32(-math.MaxFloat32)
		for i, v := range scores {
			if !taken[i] && v > bestVal {
				best, bestVal = i, v
			}
		}
		if best < 0 {
			break
		}
		taken[best] = true
		out = append(out, best)
	}
	return out
}

func topIDs(s []scored, k int) []int {
	sort.Slice(s, func(a, b int) bool { return s[a].val > s[b].val })
	if k > len(s) {
		k = len(s)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = s[i].id
	}
	return out
}

func overlapCount(a, b []int) int {
	set := make(map[int]struct{}, len(a))
	for _, x := range a {
		set[x] = struct{}{}
	}
	n := 0
	for _, x := range b {
		if _, ok := set[x]; ok {
			n++
		}
	}
	return n
}
