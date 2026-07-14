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
	// OnArm reports which ablation arm is being scored, so a long run says what it is
	// doing rather than appearing to hang.
	OnArm func(i, n int, name string)
}

// Evaluate ingests a dataset into a fresh store built on embedder and cfg, then
// scores retrieval for every labeled query. With DocLevel set, the ranking
// collapses chunks to their document ids before scoring, which matches BEIR
// document-level qrels; otherwise chunk ids are scored directly.
func Evaluate(ctx context.Context, embedder rag.Embedder, cfg rag.Config, ds *Dataset, opt Options) (eval.Report, error) {
	if opt.K <= 0 {
		opt.K = 10
	}
	store := rag.New(embedder, cfg)
	if err := store.Build(ctx, ds.Docs); err != nil {
		return eval.Report{}, fmt.Errorf("bench: build: %w", err)
	}
	return scoreArm(ctx, store, ds, Arm{Name: "default", Params: opt.Params}, opt).Report, nil
}
