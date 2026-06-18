// Package index provides a flat, quantized approximate-nearest-neighbor index
// built on TurboQuant codes. Search is a parallel linear scan that ranks by the
// low-variance quantized score and then re-ranks a short candidate list with the
// unbiased estimator, which gives high recall at a fraction of the memory of
// storing full-precision vectors.
package index

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// Metric selects the similarity used for ranking.
type Metric int

const (
	// Cosine ranks by cosine similarity (highest first). Inputs need not be
	// normalized; the estimator divides out norms.
	Cosine Metric = iota
	// InnerProduct ranks by inner product (highest first).
	InnerProduct
	// L2 ranks by Euclidean distance (smallest first).
	L2
)

// Result is a single search hit.
type Result struct {
	ID    string
	Score float32 // similarity for Cosine/InnerProduct (higher better), distance for L2 (lower better)
}

// Index is an append-only quantized vector index. It is safe for concurrent
// search once built, but Add must not race with Search.
//
// Ranking always uses the low-variance quantized Score, which is the most
// accurate estimator for ordering. If the index is created with KeepVectors, the
// original full-precision vectors are retained so a short candidate list can be
// re-ranked exactly, recovering near-perfect recall at the cost of memory.
type Index struct {
	q      *quant.Quantizer
	metric Metric
	codes  []quant.Code
	ids    []string

	keepVecs bool
	vdim     int
	vecs     []float32 // flat row-major originals, present iff keepVecs
	vnorms   []float32 // cached norms for exact cosine
}

// New creates an empty index over the given quantizer and metric. Candidate
// generation ranks by the quantized score only.
func New(q *quant.Quantizer, metric Metric) *Index {
	return &Index{q: q, metric: metric}
}

// NewWithVectors creates an index that also retains full-precision vectors of the
// given dimension, enabling exact re-ranking of candidates during Search.
func NewWithVectors(q *quant.Quantizer, metric Metric, dim int) *Index {
	return &Index{q: q, metric: metric, keepVecs: true, vdim: dim}
}

// Len returns the number of indexed vectors.
func (ix *Index) Len() int { return len(ix.codes) }

// Quantizer returns the underlying quantizer.
func (ix *Index) Quantizer() *quant.Quantizer { return ix.q }

// Add encodes and stores a vector under id, returning its ordinal.
func (ix *Index) Add(id string, vec []float32) int {
	ord := len(ix.codes)
	ix.codes = append(ix.codes, ix.q.Encode(vec))
	ix.ids = append(ix.ids, id)
	if ix.keepVecs {
		ix.vecs = append(ix.vecs, vec...)
		ix.vnorms = append(ix.vnorms, vnorm(vec))
	}
	return ord
}

func vnorm(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

// AddBatch encodes vectors in parallel and appends them. rows is row-major with
// the given stride. It returns the ordinal of the first added vector.
func (ix *Index) AddBatch(ids []string, rows []float32, stride int) int {
	n := len(ids)
	start := len(ix.codes)
	codes := make([]quant.Code, n)
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := min(lo+chunk, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			buf := make([]float32, ix.q.Dim())
			res := make([]float32, ix.q.Dim())
			for i := lo; i < hi; i++ {
				codes[i].Codes = make([]uint8, ix.q.Dim())
				ix.q.EncodeInto(rows[i*stride:i*stride+stride], buf, res, &codes[i])
			}
		}(lo, hi)
	}
	wg.Wait()
	ix.codes = append(ix.codes, codes...)
	ix.ids = append(ix.ids, ids...)
	if ix.keepVecs {
		ix.vecs = append(ix.vecs, rows[:n*stride]...)
		for i := 0; i < n; i++ {
			ix.vnorms = append(ix.vnorms, vnorm(rows[i*stride:i*stride+stride]))
		}
	}
	return start
}

// Snapshot is a serializable view of an index, excluding the quantizer (which is
// reproduced from its Config).
type Snapshot struct {
	Metric   Metric
	Codes    []quant.Code
	IDs      []string
	KeepVecs bool
	VDim     int
	Vecs     []float32
	VNorms   []float32
}

// Snapshot returns a copy-free view suitable for serialization.
func (ix *Index) Snapshot() Snapshot {
	return Snapshot{
		Metric: ix.metric, Codes: ix.codes, IDs: ix.ids,
		KeepVecs: ix.keepVecs, VDim: ix.vdim, Vecs: ix.vecs, VNorms: ix.vnorms,
	}
}

// Restore rebuilds an index from a snapshot and a matching quantizer.
func Restore(q *quant.Quantizer, s Snapshot) *Index {
	return &Index{
		q: q, metric: s.Metric, codes: s.Codes, ids: s.IDs,
		keepVecs: s.KeepVecs, vdim: s.VDim, vecs: s.Vecs, vnorms: s.VNorms,
	}
}

// higherIsBetter reports whether larger scores rank first for the metric.
func (ix *Index) higherIsBetter() bool { return ix.metric != L2 }

// score computes the ranking score of code c against a prepared query.
func (ix *Index) score(qr *quant.Query, c quant.Code) float32 {
	switch ix.metric {
	case L2:
		return qr.L2Score(c)
	case Cosine:
		return qr.CosineScore(c)
	default:
		return qr.Score(c)
	}
}

// exactScore computes the true metric between query (with norm qn) and the
// stored full-precision vector at ord. Only valid when keepVecs is set.
func (ix *Index) exactScore(query []float32, qn float32, ord int) float32 {
	v := ix.vecs[ord*ix.vdim : ord*ix.vdim+ix.vdim]
	var d float32
	switch ix.metric {
	case L2:
		for i, x := range query {
			e := x - v[i]
			d += e * e
		}
		return d
	default:
		for i, x := range query {
			d += x * v[i]
		}
		if ix.metric == Cosine {
			den := qn * ix.vnorms[ord]
			if den == 0 {
				return 0
			}
			return d / den
		}
		return d
	}
}

// Search returns the top-k results for query. It scans all codes in parallel,
// ranking by the low-variance quantized score, and keeps a candidate pool of
// size k*overscan. If the index retains full-precision vectors, the pool is
// re-ranked exactly; otherwise the quantized ranking is returned directly.
func (ix *Index) Search(query []float32, k, overscan int) []Result {
	n := len(ix.codes)
	if k > n {
		k = n
	}
	if k == 0 {
		return nil
	}
	pool := k
	if overscan > 1 {
		pool = min(k*overscan, n)
	}
	qr := ix.q.PrepareQuery(query)
	cand := ix.scan(qr, pool)
	if ix.keepVecs && overscan > 1 {
		qn := vnorm(query)
		for i := range cand {
			cand[i].score = ix.exactScore(query, qn, cand[i].ord)
		}
		ix.sortHits(cand)
	}
	if len(cand) > k {
		cand = cand[:k]
	}
	out := make([]Result, len(cand))
	for i, h := range cand {
		out[i] = Result{ID: ix.ids[h.ord], Score: h.score}
	}
	return out
}

type hit struct {
	ord   int
	score float32
}

func (ix *Index) sortHits(h []hit) {
	if ix.higherIsBetter() {
		sort.Slice(h, func(a, b int) bool { return h[a].score > h[b].score })
	} else {
		sort.Slice(h, func(a, b int) bool { return h[a].score < h[b].score })
	}
}

// scan ranks all codes against qr and returns the best `keep` as a sorted slice.
func (ix *Index) scan(qr *quant.Query, keep int) []hit {
	n := len(ix.codes)
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	parts := make([][]hit, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := min(lo+chunk, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			th := newTopHeap(keep, ix.higherIsBetter())
			for i := lo; i < hi; i++ {
				th.push(hit{ord: i, score: ix.score(qr, ix.codes[i])})
			}
			parts[w] = th.drain()
		}(w, lo, hi)
	}
	wg.Wait()

	merged := parts[0]
	for w := 1; w < workers; w++ {
		merged = append(merged, parts[w]...)
	}
	ix.sortHits(merged)
	if len(merged) > keep {
		merged = merged[:keep]
	}
	return merged
}
