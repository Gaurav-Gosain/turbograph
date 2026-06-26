package eval

import (
	"math"
	"testing"
)

func TestExactMatchAndF1(t *testing.T) {
	if !ExactMatch("The Analytical Engine", "analytical engine") {
		t.Error("normalization should make these an exact match")
	}
	if ExactMatch("Babbage", "Lovelace") {
		t.Error("different answers must not match")
	}
	// Partial overlap gives partial F1 between 0 and 1.
	f := AnswerF1("Charles Babbage invented it", "Charles Babbage")
	if f <= 0 || f >= 1 {
		t.Fatalf("partial F1 out of range: %v", f)
	}
	if AnswerF1("nothing shared", "completely different") != 0 {
		t.Error("no token overlap should be F1=0")
	}
	if AnswerF1("same words", "same words") != 1 {
		t.Error("identical answers should be F1=1")
	}
}

func TestCoverMatch(t *testing.T) {
	if !CoverMatch("The reactor was built in the town of Northgate.", "Northgate") {
		t.Error("a verbose answer containing the gold span should cover")
	}
	if !CoverMatch("Charles Babbage", "Charles Babbage") {
		t.Error("identical answers cover")
	}
	if CoverMatch("It was built in Aldon City.", "Northgate") {
		t.Error("a wrong answer must not cover")
	}
	if CoverMatch("Priya", "Priya Anand") {
		t.Error("a partial answer missing a gold token must not cover")
	}
}

func TestBootstrapCI(t *testing.T) {
	scores := make([]float64, 100)
	for i := range scores {
		if i < 70 {
			scores[i] = 1
		}
	}
	mean, lo, hi := BootstrapCI(scores, 2000, 1)
	if math.Abs(mean-0.70) > 1e-9 {
		t.Fatalf("mean = %v, want 0.70", mean)
	}
	if !(lo < mean && mean < hi) {
		t.Fatalf("CI should bracket the mean: [%v, %v] around %v", lo, hi, mean)
	}
	if lo < 0.55 || hi > 0.85 {
		t.Fatalf("CI implausibly wide for n=100: [%v, %v]", lo, hi)
	}
}
