package bench

import (
	"context"
	"fmt"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// Options configures an evaluation run.
type Options struct {
	K          int                // cutoff for the @k metrics (default 10)
	DocLevel   bool               // collapse retrieved chunks to documents (BEIR convention)
	Params     rag.RetrieveParams // retrieval knobs; TopK defaults to max(K, 10)
	OnProgress func(done, total int)
}

// Evaluate ingests a dataset into a fresh store built on embedder and cfg, then
// scores retrieval for every labeled query. With DocLevel set, the ranking
// collapses chunks to their document ids before scoring, which matches BEIR
// document-level qrels; otherwise chunk ids are scored directly.
func Evaluate(ctx context.Context, embedder rag.Embedder, cfg rag.Config, ds *Dataset, opt Options) (eval.Report, error) {
	if opt.K <= 0 {
		opt.K = 10
	}
	topK := opt.Params.TopK
	if topK < opt.K {
		topK = opt.K
		if topK < 10 {
			topK = 10
		}
	}
	store := rag.New(embedder, cfg)
	if err := store.Build(ctx, ds.Docs); err != nil {
		return eval.Report{}, fmt.Errorf("bench: build: %w", err)
	}

	done := 0
	retrieve := func(query string) []string {
		p := opt.Params
		p.TopK = topK
		res, err := store.Retrieve(ctx, query, p)
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
				if _, ok := seen[id]; ok {
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
	return eval.Run(ds.Cases, opt.K, retrieve), nil
}
