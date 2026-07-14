package bench

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// Arm is one retrieval configuration to score.
type Arm struct {
	Name   string
	Params rag.RetrieveParams
}

// ArmResult is one arm's score on the dataset.
// QueryP50/P95 are END TO END: they embed the query and then search. Against a local
// Ollama the embedding is a network round trip of well over a hundred milliseconds, so
// this number says almost nothing about how fast turbograph searches, and reporting it
// as "query latency" would credit the index with the embedder's cost. Search speed is
// measured separately, with an in-process embedder and no network in the loop; see the
// Speed benchmarks in rag.
type ArmResult struct {
	Name     string        `json:"name"`
	Report   eval.Report   `json:"report"`
	QueryP50 time.Duration `json:"query_p50"`
	QueryP95 time.Duration `json:"query_p95"`
}

// Ablate scores several retrieval configurations against ONE store.
//
// It exists because the interesting question is never "how good is turbograph", it is
// "which part of turbograph is doing the work". Answering that by running the whole
// benchmark once per configuration means re-embedding the corpus every time, which on
// a real dataset is minutes per arm and so does not get done. Building once and scoring
// many arms makes the ablation cheap enough to actually run, and it also removes the
// confound: every arm is scored against exactly the same index.
//
// It also times the queries, because retrieval quality that costs a second per query is
// a different product from the same quality at ten milliseconds.
func Ablate(ctx context.Context, embedder rag.Embedder, cfg rag.Config, ds *Dataset, arms []Arm, opt Options) ([]ArmResult, *rag.Store, error) {
	if opt.K <= 0 {
		opt.K = 10
	}
	store := rag.New(embedder, cfg)
	if err := store.Build(ctx, ds.Docs); err != nil {
		return nil, nil, fmt.Errorf("bench: build: %w", err)
	}

	out := make([]ArmResult, 0, len(arms))
	for ai, arm := range arms {
		topK := arm.Params.TopK
		if topK < opt.K {
			topK = opt.K
		}
		if topK < 10 {
			topK = 10
		}
		lat := make([]time.Duration, 0, len(ds.Cases))
		done := 0
		retrieve := func(query string) []string {
			p := arm.Params
			p.TopK = topK
			t0 := time.Now()
			res, err := store.Retrieve(ctx, query, p)
			lat = append(lat, time.Since(t0))
			done++
			if opt.OnProgress != nil {
				opt.OnProgress(done, len(ds.Cases))
			}
			if err != nil {
				return nil
			}
			ranked := make([]string, 0, len(res))
			if opt.DocLevel {
				seen := map[string]struct{}{}
				for _, r := range res {
					id := r.Chunk.DocID
					if _, dup := seen[id]; dup {
						continue
					}
					seen[id] = struct{}{}
					ranked = append(ranked, id)
				}
			} else {
				for _, r := range res {
					ranked = append(ranked, r.Chunk.ID)
				}
			}
			return ranked
		}
		if opt.OnArm != nil {
			opt.OnArm(ai, len(arms), arm.Name)
		}
		rep := eval.Run(ds.Cases, opt.K, retrieve)
		p50, p95 := percentiles(lat)
		out = append(out, ArmResult{Name: arm.Name, Report: rep, QueryP50: p50, QueryP95: p95})
	}
	return out, store, nil
}

// percentiles returns the median and 95th percentile latency. A mean would hide the
// tail, and the tail is what a user actually notices.
func percentiles(d []time.Duration) (p50, p95 time.Duration) {
	if len(d) == 0 {
		return 0, 0
	}
	s := make([]time.Duration, len(d))
	copy(s, d)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[len(s)*50/100], s[min(len(s)*95/100, len(s)-1)]
}
