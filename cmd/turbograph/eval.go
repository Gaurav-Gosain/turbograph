package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// cmdEval scores retrieval quality against a labeled suite. The suite is JSONL,
// one {"query":..., "relevant":[chunk ids]} per line, and the report gives
// recall, precision, MRR, NDCG, and context precision at the cut-off k. It is
// deterministic given a fixed store and embedder, so it is suitable for
// regression gating in CI.
func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	storePath := fs.String("store", "store.tg", "store path")
	suitePath := fs.String("suite", "", "JSONL suite: one {query, relevant:[ids]} per line")
	k := fs.Int("k", 10, "cut-off k for the metrics")
	mix := fs.Float64("mix", 0.2, "graph boost added on top of relevance (0 uses the default, negative disables the graph)")
	mmr := fs.Float64("mmr", 0, "MMR diversity lambda in (0,1); 0 disables")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "ollama embedding model")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	asJSON := fs.Bool("json", false, "emit the full report as JSON instead of a table")
	fs.Parse(args)

	if *suitePath == "" {
		return fmt.Errorf("--suite is required")
	}
	sf, err := os.Open(*suitePath)
	if err != nil {
		return err
	}
	defer sf.Close()
	cases, err := eval.LoadSuite(sf)
	if err != nil {
		return err
	}
	if len(cases) == 0 {
		return fmt.Errorf("suite %s has no cases", *suitePath)
	}

	f, err := os.Open(*storePath)
	if err != nil {
		return err
	}
	defer f.Close()

	client := ollama.New()
	client.SetEmbedModel(*embedModel)
	if *ollamaURL != "" {
		client.BaseURL = *ollamaURL
	}
	store, err := rag.Load(client, f)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	retrieve := func(query string) []string {
		res, rerr := store.Retrieve(ctx, query, rag.RetrieveParams{
			TopK: *k, GraphMix: float32(*mix), MMRLambda: float32(*mmr),
		})
		if rerr != nil {
			fmt.Fprintf(os.Stderr, "retrieve %q: %v\n", query, rerr)
			return nil
		}
		ids := make([]string, len(res))
		for i, r := range res {
			ids[i] = r.Chunk.ID
		}
		return ids
	}

	report := eval.Run(cases, *k, retrieve)
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}

	m := report.Mean
	fmt.Printf("suite: %s  cases: %d  k: %d\n\n", *suitePath, len(report.Cases), report.K)
	fmt.Printf("  recall@%d            %.3f\n", *k, m.RecallAtK)
	fmt.Printf("  precision@%d         %.3f\n", *k, m.PrecisionAtK)
	fmt.Printf("  MRR                  %.3f\n", m.MRR)
	fmt.Printf("  NDCG@%d              %.3f\n", *k, m.NDCGAtK)
	fmt.Printf("  context precision@%d %.3f\n", *k, m.ContextPrecisionAtK)
	return nil
}
