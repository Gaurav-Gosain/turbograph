package eval

import (
	"math"
	"math/rand"
	"sort"
	"strings"
	"unicode"
)

// This file adds answer-quality metrics: token-overlap F1 and exact match for a
// generated answer against a gold answer, plus a bootstrap confidence interval
// for any per-case metric. They are deterministic and LLM-free, so they belong in
// the reproducible eval path (a model-based correctness judge is a separate,
// non-deterministic tool). The normalization follows the standard SQuAD/HotpotQA
// recipe and mirrors cognee's f1/exact_match metrics.

// normalizeAnswer lowercases, strips punctuation, drops the articles a/an/the, and
// collapses whitespace, the standard answer-normalization for QA scoring.
func normalizeAnswer(s string) []string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteByte(' ')
		}
	}
	fields := strings.Fields(b.String())
	out := fields[:0]
	for _, f := range fields {
		if f == "a" || f == "an" || f == "the" {
			continue
		}
		out = append(out, f)
	}
	return out
}

// ExactMatch reports whether the prediction equals the gold answer after
// normalization.
func ExactMatch(pred, gold string) bool {
	p, g := normalizeAnswer(pred), normalizeAnswer(gold)
	if len(p) != len(g) {
		return false
	}
	for i := range p {
		if p[i] != g[i] {
			return false
		}
	}
	return true
}

// AnswerF1 is the token-overlap F1 between a predicted and a gold answer after
// normalization, the standard partial-credit QA metric.
func AnswerF1(pred, gold string) float64 {
	p, g := normalizeAnswer(pred), normalizeAnswer(gold)
	if len(p) == 0 && len(g) == 0 {
		return 1
	}
	if len(p) == 0 || len(g) == 0 {
		return 0
	}
	gcount := make(map[string]int, len(g))
	for _, t := range g {
		gcount[t]++
	}
	tp := 0
	for _, t := range p {
		if gcount[t] > 0 {
			gcount[t]--
			tp++
		}
	}
	if tp == 0 {
		return 0
	}
	precision := float64(tp) / float64(len(p))
	recall := float64(tp) / float64(len(g))
	return 2 * precision * recall / (precision + recall)
}

// CoverMatch reports whether every token of the gold answer appears in the
// prediction (after normalization). It is the verbosity-robust complement to
// ExactMatch: a correct answer wrapped in explanation ("The reactor was built in
// Northgate." for gold "Northgate") still counts, while a wrong answer does not.
// This is the standard "cover"/answer-recall criterion used for short-answer QA
// where the generator is not constrained to emit only the span. An empty gold
// answer is vacuously covered.
func CoverMatch(pred, gold string) bool {
	g := normalizeAnswer(gold)
	if len(g) == 0 {
		return true
	}
	have := make(map[string]struct{})
	for _, t := range normalizeAnswer(pred) {
		have[t] = struct{}{}
	}
	for _, t := range g {
		if _, ok := have[t]; !ok {
			return false
		}
	}
	return true
}

// BootstrapCI returns the mean of scores and a 95% confidence interval estimated
// by resampling with replacement. It lets a benchmark say whether a delta between
// two runs is real or noise on a small set. A nil or single-element input returns
// the mean with a zero-width interval.
func BootstrapCI(scores []float64, iters int, seed int64) (mean, lo, hi float64) {
	n := len(scores)
	if n == 0 {
		return 0, 0, 0
	}
	var sum float64
	for _, s := range scores {
		sum += s
	}
	mean = sum / float64(n)
	if n == 1 || iters <= 0 {
		return mean, mean, mean
	}
	rng := rand.New(rand.NewSource(seed))
	means := make([]float64, iters)
	for i := range means {
		var s float64
		for range n {
			s += scores[rng.Intn(n)]
		}
		means[i] = s / float64(n)
	}
	sort.Float64s(means)
	lo = means[int(math.Round(0.025*float64(iters-1)))]
	hi = means[int(math.Round(0.975*float64(iters-1)))]
	return mean, lo, hi
}
