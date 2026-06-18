package lexical

import (
	"math"
	"sort"
)

// Result is a scored document, returned ranked best-first by Search and RRF.
type Result struct {
	ID    string
	Score float32
}

// Config tunes the BM25 scoring function. The zero value is not valid; obtain
// sensible defaults from DefaultConfig.
type Config struct {
	// K1 controls term-frequency saturation. Higher values let repeated terms
	// keep adding score; lower values saturate sooner. Typical range 1.2..2.0.
	K1 float64
	// B controls document-length normalization, in [0,1]. At 1 the penalty for
	// long documents is full; at 0 length is ignored.
	B float64
}

// DefaultConfig returns the canonical Okapi BM25 parameters.
func DefaultConfig() Config {
	return Config{K1: 1.2, B: 0.75}
}

// posting records that a term occurs tf times in the document at index ord. We
// store an ordinal rather than the string ID so postings stay compact and so
// scoring can index doc-length and ID arrays directly.
type posting struct {
	ord uint32
	tf  uint32
}

// Index is an in-memory BM25 index over a corpus. It is safe for concurrent
// Search once building has finished, but Add is not safe to call concurrently
// with itself or with Search.
type Index struct {
	cfg Config

	ids      []string // ordinal -> document ID
	docLen   []uint32 // ordinal -> token count
	postings map[string][]posting

	totalLen uint64 // sum of docLen, for the average

	finalized bool
	idf       map[string]float64 // populated by finalize
	avgLen    float64
}

// New returns an empty index using cfg. Out-of-range parameters are clamped to
// keep scoring well defined.
func New(cfg Config) *Index {
	if cfg.K1 < 0 {
		cfg.K1 = 0
	}
	if cfg.B < 0 {
		cfg.B = 0
	} else if cfg.B > 1 {
		cfg.B = 1
	}
	return &Index{
		cfg:      cfg,
		postings: make(map[string][]posting),
		idf:      make(map[string]float64),
	}
}

// Add inserts a single document. Adding mutates the index, so any precomputed
// IDF is invalidated and recomputed lazily on the next Search. Empty documents
// (no surviving tokens) still occupy an ordinal so corpus statistics stay
// honest, but they can never match a query.
func (ix *Index) Add(id, text string) {
	ix.finalized = false

	ord := uint32(len(ix.ids))
	ix.ids = append(ix.ids, id)

	toks := tokenize(text)
	ix.docLen = append(ix.docLen, uint32(len(toks)))
	ix.totalLen += uint64(len(toks))

	// Collapse repeated terms into a single posting carrying the term frequency.
	tf := make(map[string]uint32, len(toks))
	for _, t := range toks {
		tf[t]++
	}
	for term, n := range tf {
		ix.postings[term] = append(ix.postings[term], posting{ord: ord, tf: n})
	}
}

// Build inserts a batch of documents and finalizes the index. It is a
// convenience over repeated Add calls. ids and texts must have equal length.
func Build(cfg Config, ids, texts []string) *Index {
	ix := New(cfg)
	n := min(len(ids), len(texts))
	for i := 0; i < n; i++ {
		ix.Add(ids[i], texts[i])
	}
	ix.finalize()
	return ix
}

// Len reports the number of documents in the index.
func (ix *Index) Len() int { return len(ix.ids) }

// finalize computes the average document length and the IDF of every term.
// Doing this once amortizes the log over all future queries. The probabilistic
// IDF, ln(1 + (N - df + 0.5)/(df + 0.5)), is the non-negative variant: the
// leading 1 keeps the value positive even for terms that appear in more than
// half the corpus, which the bare Robertson/Sparck-Jones form would drive
// negative.
func (ix *Index) finalize() {
	n := len(ix.ids)
	if n == 0 {
		ix.avgLen = 0
		ix.finalized = true
		return
	}
	ix.avgLen = float64(ix.totalLen) / float64(n)

	clear(ix.idf)
	nf := float64(n)
	for term, plist := range ix.postings {
		df := float64(len(plist))
		ix.idf[term] = math.Log(1 + (nf-df+0.5)/(df+0.5))
	}
	ix.finalized = true
}

// Search returns the top-k documents for query, ranked by descending BM25
// score. Only query terms contribute, so cost is proportional to the combined
// length of the matched postings lists rather than the corpus size. Ties break
// by document ID so results are fully deterministic. A non-positive k or a
// query with no scorable terms yields nil.
// Finalize precomputes IDF and average document length. Search calls it lazily,
// but callers that share an Index across goroutines should call it once after the
// last Add so concurrent Search calls never trigger the mutation themselves.
func (ix *Index) Finalize() {
	if !ix.finalized {
		ix.finalize()
	}
}

func (ix *Index) Search(query string, k int) []Result {
	if k <= 0 || len(ix.ids) == 0 {
		return nil
	}
	if !ix.finalized {
		ix.finalize()
	}

	qterms := tokenize(query)
	if len(qterms) == 0 {
		return nil
	}

	// A query repeating a term should not double-count its IDF, so dedupe.
	seen := make(map[string]struct{}, len(qterms))
	scores := make(map[uint32]float64)
	for _, term := range qterms {
		if _, dup := seen[term]; dup {
			continue
		}
		seen[term] = struct{}{}

		plist, ok := ix.postings[term]
		if !ok {
			continue
		}
		idf := ix.idf[term]
		for _, p := range plist {
			scores[p.ord] += ix.termScore(idf, p.tf, ix.docLen[p.ord])
		}
	}
	if len(scores) == 0 {
		return nil
	}

	results := make([]Result, 0, len(scores))
	for ord, s := range scores {
		results = append(results, Result{ID: ix.ids[ord], Score: float32(s)})
	}
	sortResults(results)

	if k < len(results) {
		results = results[:k]
	}
	return results
}

// termScore is the BM25 per-term contribution: the IDF times saturated,
// length-normalized term frequency.
func (ix *Index) termScore(idf float64, tf, dl uint32) float64 {
	f := float64(tf)
	var norm float64
	if ix.avgLen > 0 {
		norm = ix.cfg.B * float64(dl) / ix.avgLen
	}
	denom := f + ix.cfg.K1*(1-ix.cfg.B+norm)
	if denom == 0 {
		return 0
	}
	return idf * (f * (ix.cfg.K1 + 1)) / denom
}

// sortResults orders by descending score, breaking ties on ID for determinism.
func sortResults(rs []Result) {
	sort.Slice(rs, func(i, j int) bool {
		if rs[i].Score != rs[j].Score {
			return rs[i].Score > rs[j].Score
		}
		return rs[i].ID < rs[j].ID
	})
}
