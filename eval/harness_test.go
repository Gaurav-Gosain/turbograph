package eval

import (
	"math"
	"strings"
	"testing"
)

func TestLoadSuiteMultipleLines(t *testing.T) {
	// Two valid cases separated by a blank line and surrounding whitespace.
	input := `
{"query":"q1","relevant":["a","b"]}

  {"query":"q2","relevant":["c"]}
`
	cases, err := LoadSuite(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cases) != 2 {
		t.Fatalf("got %d cases, want 2", len(cases))
	}
	if cases[0].Query != "q1" || len(cases[0].Relevant) != 2 {
		t.Errorf("case 0 = %+v", cases[0])
	}
	if cases[1].Query != "q2" || cases[1].Relevant[0] != "c" {
		t.Errorf("case 1 = %+v", cases[1])
	}
}

func TestLoadSuiteMalformedReportsLineNumber(t *testing.T) {
	// Valid line 1, blank line 2, malformed line 3.
	input := "{\"query\":\"ok\",\"relevant\":[\"a\"]}\n\n{not json}\n"
	_, err := LoadSuite(strings.NewReader(input))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "line 3") {
		t.Errorf("error should name line 3, got: %v", err)
	}
}

func TestLoadSuiteEmpty(t *testing.T) {
	cases, err := LoadSuite(strings.NewReader("\n  \n"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cases) != 0 {
		t.Errorf("got %d cases, want 0", len(cases))
	}
}

func TestRunAggregatesMean(t *testing.T) {
	cases := []Case{
		{Query: "q1", Relevant: []string{"a"}},
		{Query: "q2", Relevant: []string{"b"}},
	}
	// Deterministic retrieve:
	//  q1 -> [a, z]: a relevant at rank 1.
	//  q2 -> [z, b]: b relevant at rank 2.
	retrieve := func(q string) []string {
		switch q {
		case "q1":
			return []string{"a", "z"}
		case "q2":
			return []string{"z", "b"}
		}
		return nil
	}
	rep := Run(cases, 2, retrieve)

	if rep.K != 2 {
		t.Errorf("K = %d, want 2", rep.K)
	}
	if len(rep.Cases) != 2 {
		t.Fatalf("got %d case results, want 2", len(rep.Cases))
	}

	// Per-case expectations.
	// q1: Recall@2 = 1/1 = 1. Precision@2 = 1/2. MRR = 1.
	//     NDCG@2: DCG=1/log2(2)=1, IDCG=1 => 1. ContextPrecision = (1/1)/1 = 1.
	// q2: Recall@2 = 1. Precision@2 = 1/2. MRR = 1/2.
	//     NDCG@2: DCG=1/log2(3), IDCG=1/log2(2)=1 => 1/log2(3).
	//     ContextPrecision = (1/2)/1 = 1/2.
	wantMeanRecall := (1.0 + 1.0) / 2
	wantMeanPrecision := (0.5 + 0.5) / 2
	wantMeanMRR := (1.0 + 0.5) / 2
	wantMeanNDCG := (1.0 + 1/math.Log2(3)) / 2
	wantMeanCP := (1.0 + 0.5) / 2

	approx(t, "mean Recall", rep.Mean.RecallAtK, wantMeanRecall)
	approx(t, "mean Precision", rep.Mean.PrecisionAtK, wantMeanPrecision)
	approx(t, "mean MRR", rep.Mean.MRR, wantMeanMRR)
	approx(t, "mean NDCG", rep.Mean.NDCGAtK, wantMeanNDCG)
	approx(t, "mean ContextPrecision", rep.Mean.ContextPrecisionAtK, wantMeanCP)
}

func TestRunEmptyCases(t *testing.T) {
	rep := Run(nil, 3, func(string) []string { return nil })
	if rep.K != 3 || len(rep.Cases) != 0 {
		t.Errorf("unexpected report: %+v", rep)
	}
	// Mean of no cases is the zero value.
	if rep.Mean != (Metrics{}) {
		t.Errorf("mean of empty run = %+v, want zero", rep.Mean)
	}
}

func TestRunThreeCases(t *testing.T) {
	cases := []Case{
		{Query: "a", Relevant: []string{"1"}},
		{Query: "b", Relevant: []string{"2"}},
		{Query: "c", Relevant: []string{"3"}},
	}
	// Each query returns its single relevant id at rank 1 => all metrics 1.
	retrieve := func(q string) []string {
		switch q {
		case "a":
			return []string{"1"}
		case "b":
			return []string{"2"}
		case "c":
			return []string{"3"}
		}
		return nil
	}
	rep := Run(cases, 1, retrieve)
	approx(t, "mean Recall all perfect", rep.Mean.RecallAtK, 1.0)
	approx(t, "mean MRR all perfect", rep.Mean.MRR, 1.0)
	approx(t, "mean NDCG all perfect", rep.Mean.NDCGAtK, 1.0)
	approx(t, "mean ContextPrecision all perfect", rep.Mean.ContextPrecisionAtK, 1.0)
}
