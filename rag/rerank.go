package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Generator produces a completion for a system and user prompt. It is the
// minimal surface the reranker and other LLM-assisted steps need; the Ollama
// client satisfies it once a model is bound.
type Generator interface {
	Generate(ctx context.Context, system, prompt string) (string, error)
}

const rerankSystem = "You are a search relevance judge. " +
	"Given a query and numbered passages, rate how well each passage helps answer the query " +
	"on a scale of 0 to 10. Respond with only a JSON array of objects like " +
	`[{"i":0,"score":7},{"i":1,"score":2}], one per passage, and nothing else.`

// Rerank reorders retrieved results with a single pointwise LLM call, then blends
// the model score with the original fused score so the model refines rather than
// overrides retrieval. It is fail-open: any error or unparseable output returns
// the input truncated to topK, so enabling it can never make results worse than
// the base ranking. Passages are truncated to keep the prompt bounded.
func Rerank(ctx context.Context, gen Generator, query string, res []Retrieved, topK int) []Retrieved {
	if gen == nil || len(res) <= 1 {
		return trim(res, topK)
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "Query: %s\n\nPassages:\n", query)
	for i, r := range res {
		fmt.Fprintf(&sb, "[%d] %s\n", i, truncateWords(r.Chunk.Text, 120))
	}
	sb.WriteString("\nJSON:")
	out, err := gen.Generate(ctx, rerankSystem, sb.String())
	if err != nil {
		return trim(res, topK)
	}
	scores := parseRerank(out, len(res))
	if scores == nil {
		return trim(res, topK)
	}

	// Normalize both signals to [0,1] and blend; the model leads, retrieval anchors.
	var maxModel, maxBase float32
	for i := range res {
		if scores[i] > maxModel {
			maxModel = scores[i]
		}
		if res[i].Score > maxBase {
			maxBase = res[i].Score
		}
	}
	out2 := make([]Retrieved, len(res))
	copy(out2, res)
	const wModel = 0.7
	for i := range out2 {
		m, b := float32(0), float32(0)
		if maxModel > 0 {
			m = scores[i] / maxModel
		}
		if maxBase > 0 {
			b = out2[i].Score / maxBase
		}
		out2[i].Score = wModel*m + (1-wModel)*b
	}
	sort.SliceStable(out2, func(a, b int) bool { return out2[a].Score > out2[b].Score })
	return trim(out2, topK)
}

// parseRerank extracts a score per passage index from the model output, tolerating
// surrounding prose or code fences. Returns nil if nothing usable is found.
func parseRerank(out string, n int) []float32 {
	start := strings.IndexByte(out, '[')
	end := strings.LastIndexByte(out, ']')
	if start < 0 || end <= start {
		return nil
	}
	var items []struct {
		I     int     `json:"i"`
		Score float32 `json:"score"`
	}
	if err := json.Unmarshal([]byte(out[start:end+1]), &items); err != nil || len(items) == 0 {
		return nil
	}
	scores := make([]float32, n)
	any := false
	for _, it := range items {
		if it.I >= 0 && it.I < n {
			scores[it.I] = it.Score
			any = true
		}
	}
	if !any {
		return nil
	}
	return scores
}

func trim(res []Retrieved, k int) []Retrieved {
	if k > 0 && len(res) > k {
		return res[:k]
	}
	return res
}

func truncateWords(s string, n int) string {
	f := strings.Fields(s)
	if len(f) <= n {
		return strings.Join(f, " ")
	}
	return strings.Join(f[:n], " ") + "..."
}

// ShouldAbstain reports whether retrieval is too weak to answer from the corpus.
// It uses the raw cosine Similarity of the top hit (an objective signal), not the
// blended Score, so the threshold is comparable across queries. A store with no
// results, or whose best hit is below minTopSim, should abstain rather than let
// the model answer from parametric memory.
func ShouldAbstain(res []Retrieved, minTopSim float32) bool {
	if len(res) == 0 {
		return true
	}
	if minTopSim <= 0 {
		return false
	}
	return res[0].Similarity < minTopSim
}
