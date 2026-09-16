// Package bench is turbograph's retrieval benchmark harness: dataset loaders, a
// deterministic offline embedder for regression gating, and an evaluation runner
// that scores the real ingestion and retrieval pipeline. The same code powers the
// committed CI regression suite (offline, deterministic) and the `turbograph
// bench` command that reproduces the headline numbers with a real model.
package bench

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// HashEmbedder is a deterministic, dependency-free bag-of-words embedder used for
// offline regression tests. Each token is hashed into the vector with a signed
// hashing trick and weighted by term frequency, then the vector is L2-normalized.
// It is not a semantic model: cosine similarity reflects shared vocabulary, which
// is enough to verify that the retrieval pipeline ranks lexically-related text
// correctly and to catch regressions without a network or a model server. It is
// never used for the published benchmark numbers, which use a real embedder.
//
// Stopwords and single-character tokens are dropped before hashing, matching the
// lexical index's view of a document. Without that, every document in the corpus
// shares a handful of high-frequency words and cosine similarity saturates at a
// floor high enough to hide a genuinely relevant hit behind unrelated text, which
// would make the deterministic suite insensitive to ranking regressions. With the
// same token view on both signals, dense and lexical agree on what is the same
// text, which is exactly the regime where hybrid fusion is well-defined.
type HashEmbedder struct {
	Dim int // embedding dimension; 256 is a good default for the tests
}

// stopwords mirrors lexical's small function-word set so the dense and sparse
// sides agree on what is discriminative text.
var stopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {},
	"by": {}, "for": {}, "from": {}, "has": {}, "he": {}, "in": {}, "is": {},
	"it": {}, "its": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
	"to": {}, "was": {}, "were": {}, "will": {}, "with": {}, "this": {},
	"these": {}, "those": {}, "but": {}, "not": {}, "they": {}, "their": {},
}

func (e HashEmbedder) dim() int {
	if e.Dim <= 0 {
		return 256
	}
	return e.Dim
}

// hardDim is the offline suite's embedding width. The hashing trick makes two
// distinct tokens collide with probability ~1/dim when they land in the same
// bucket, and at 256 buckets that noise is visible as a spurious cosine floor
// between unrelated documents. 2048 keeps the small deterministic corpora
// effectively collision-free while staying fast at test scale.
const hardDim = 2048

// hardEmbedder returns the embedder used by the deterministic regression suites.
func hardEmbedder() HashEmbedder { return HashEmbedder{Dim: hardDim} }

// Embed implements rag.Embedder.
func (e HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	d := e.dim()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = e.vector(t, d)
	}
	return out, nil
}

func (e HashEmbedder) vector(text string, d int) []float32 {
	v := make([]float32, d)
	for _, tok := range tokenize(text) {
		h := fnv.New32a()
		h.Write([]byte(tok))
		sum := h.Sum32()
		idx := int(sum % uint32(d))
		// A second hash bit gives the term a stable +/- sign, halving collisions.
		sign := float32(1)
		if sum&0x80000000 != 0 {
			sign = -1
		}
		v[idx] += sign
	}
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// tokenize lowercases, splits on non-alphanumeric runes, and drops stopwords and
// single-character tokens, mirroring the lexical index's token view.
func tokenize(text string) []string {
	var out []string
	for _, f := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(f) < 2 {
			continue
		}
		if _, stop := stopwords[f]; stop {
			continue
		}
		out = append(out, f)
	}
	return out
}
