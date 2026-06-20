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
type HashEmbedder struct {
	Dim int // embedding dimension; 256 is a good default for the tests
}

func (e HashEmbedder) dim() int {
	if e.Dim <= 0 {
		return 256
	}
	return e.Dim
}

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

// tokenize lowercases and splits on non-alphanumeric runes.
func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
