package eval

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Case is a single evaluation example: a query and the ids of the documents or
// chunks considered relevant for it.
type Case struct {
	Query    string   `json:"query"`
	Relevant []string `json:"relevant"`
}

// Metrics bundles the computed scores for one query or for an aggregate mean.
type Metrics struct {
	RecallAtK           float64 `json:"recall_at_k"`
	PrecisionAtK        float64 `json:"precision_at_k"`
	MRR                 float64 `json:"mrr"`
	NDCGAtK             float64 `json:"ndcg_at_k"`
	ContextPrecisionAtK float64 `json:"context_precision_at_k"`
}

// CaseResult pairs a query with its per-case metric scores.
type CaseResult struct {
	Query   string  `json:"query"`
	Metrics Metrics `json:"metrics"`
}

// Report is the full output of an evaluation run: the cut-off k, per-case
// results, and the mean of each metric across cases.
type Report struct {
	K     int          `json:"k"`
	Cases []CaseResult `json:"cases"`
	Mean  Metrics      `json:"mean"`
}

// LoadSuite parses a JSONL stream, one Case per non-empty line. Blank lines
// (including whitespace-only lines) are skipped so files can be visually
// grouped. A malformed line produces an error that names the 1-based line
// number to make the failure easy to locate.
func LoadSuite(r io.Reader) ([]Case, error) {
	var cases []Case
	scanner := bufio.NewScanner(r)
	// Allow long lines: relevant id lists can be sizeable. 1 MiB per line.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		var c Case
		if err := json.Unmarshal([]byte(text), &c); err != nil {
			return nil, fmt.Errorf("eval: parse error on line %d: %w", line, err)
		}
		cases = append(cases, c)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("eval: read error after line %d: %w", line, err)
	}
	return cases, nil
}

// relevantSet builds a lookup set from a Case's relevant id slice, deduping ids.
func relevantSet(ids []string) map[string]struct{} {
	set := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		set[id] = struct{}{}
	}
	return set
}

// Run evaluates every case by calling retrieve for its query, scoring the
// returned ranking against the case's relevant set at cut-off k, and returns a
// Report whose Mean field holds the arithmetic mean of each metric over all
// cases. The mean of an empty case list is all zeros.
func Run(cases []Case, k int, retrieve func(query string) []string) Report {
	report := Report{K: k, Cases: make([]CaseResult, 0, len(cases))}
	var sum Metrics
	for _, c := range cases {
		relevant := relevantSet(c.Relevant)
		ranked := retrieve(c.Query)
		m := Metrics{
			RecallAtK:           RecallAtK(ranked, relevant, k),
			PrecisionAtK:        PrecisionAtK(ranked, relevant, k),
			MRR:                 MRR(ranked, relevant),
			NDCGAtK:             NDCGAtK(ranked, relevant, k),
			ContextPrecisionAtK: ContextPrecisionAtK(ranked, relevant, k),
		}
		report.Cases = append(report.Cases, CaseResult{Query: c.Query, Metrics: m})

		sum.RecallAtK += m.RecallAtK
		sum.PrecisionAtK += m.PrecisionAtK
		sum.MRR += m.MRR
		sum.NDCGAtK += m.NDCGAtK
		sum.ContextPrecisionAtK += m.ContextPrecisionAtK
	}
	if n := float64(len(cases)); n > 0 {
		report.Mean = Metrics{
			RecallAtK:           sum.RecallAtK / n,
			PrecisionAtK:        sum.PrecisionAtK / n,
			MRR:                 sum.MRR / n,
			NDCGAtK:             sum.NDCGAtK / n,
			ContextPrecisionAtK: sum.ContextPrecisionAtK / n,
		}
	}
	return report
}
