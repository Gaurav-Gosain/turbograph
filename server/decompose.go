package server

import (
	"context"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// decomposeSystem instructs the model to break a multi-hop question into focused
// retrieval subqueries. Adapted from cognee's decomposition prompt.
const decomposeSystem = "Decompose the user's question into 1 to 5 focused retrieval subqueries, " +
	"ordered from broadest to most specific. Each subquery retrieves one fact or hop needed to answer. " +
	"If the question is already focused, return it unchanged as a single line. " +
	"Output only the subqueries, one per line, no numbering, no commentary."

// decomposeQuery asks the model to split a question into subqueries for multi-hop
// retrieval. It returns at most five, always including a usable set: on any error
// or an empty result it falls open to the original query. The model is called
// once.
func (s *Server) decomposeQuery(ctx context.Context, model, query string) []string {
	if s.gen == nil || model == "" {
		return []string{query}
	}
	out, err := s.gen.Generate(ctx, model, decomposeSystem, "Question: "+query+"\n\nSubqueries:")
	if err != nil {
		return []string{query}
	}
	var subs []string
	seen := map[string]struct{}{}
	for _, line := range strings.Split(out, "\n") {
		q := strings.TrimSpace(line)
		q = strings.TrimLeft(q, "-*0123456789. )")
		q = strings.TrimSpace(q)
		if q == "" || len(q) > 4*len(query)+120 {
			continue
		}
		key := strings.ToLower(q)
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		subs = append(subs, q)
		if len(subs) == 5 {
			break
		}
	}
	if len(subs) == 0 {
		return []string{query}
	}
	return subs
}

// retrieveDecomposed runs retrieval for each subquery and unions the candidates,
// keeping the best score per chunk. This widens recall on multi-hop questions:
// each hop gets its own seed set, so evidence that lives in different documents
// and never co-occurs with the full question still surfaces. With a single
// subquery it is identical to one Retrieve call.
func (s *Server) retrieveDecomposed(ctx context.Context, st *rag.Store, subs []string, p rag.RetrieveParams) ([]rag.Retrieved, error) {
	if len(subs) <= 1 {
		q := ""
		if len(subs) == 1 {
			q = subs[0]
		}
		return st.Retrieve(ctx, q, p)
	}
	best := map[string]rag.Retrieved{}
	order := []string{}
	for _, sub := range subs {
		res, err := st.Retrieve(ctx, sub, p)
		if err != nil {
			return nil, err
		}
		for _, r := range res {
			id := r.Chunk.ID
			if cur, ok := best[id]; !ok || r.Score > cur.Score {
				if !ok {
					order = append(order, id)
				}
				best[id] = r
			}
		}
	}
	out := make([]rag.Retrieved, 0, len(order))
	for _, id := range order {
		out = append(out, best[id])
	}
	// Re-sort by the fused (max) score, highest first.
	sortByScoreDesc(out)
	return out, nil
}

func sortByScoreDesc(res []rag.Retrieved) {
	for i := 1; i < len(res); i++ {
		for j := i; j > 0 && res[j].Score > res[j-1].Score; j-- {
			res[j], res[j-1] = res[j-1], res[j]
		}
	}
}
