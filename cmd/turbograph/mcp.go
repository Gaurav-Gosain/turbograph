package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/Gaurav-Gosain/turbograph/mcp"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// cmdMCP serves a store to MCP clients (Claude Desktop, editors, agents) over
// stdio. It exposes two tools: "search" returns the top retrieved chunks as
// JSON, and "answer" synthesizes a grounded answer with the configured model.
// The transport is line-delimited JSON-RPC on stdin/stdout, so it plugs into any
// MCP host with a command entry.
func cmdMCP(args []string) error {
	fs := flag.NewFlagSet("mcp", flag.ExitOnError)
	storePath := fs.String("store", "store.tg", "store path to serve")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "ollama embedding model")
	genModel := fs.String("gen-model", "", "ollama model for the answer tool (omit to expose search only)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	topk := fs.Int("topk", 8, "default chunks to retrieve")
	mix := fs.Float64("mix", 0.6, "graph vs similarity blend in [0,1]")
	fs.Parse(args)

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

	srv := mcp.NewServer("turbograph", "1")

	type searchArgs struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}
	retrieve := func(ctx context.Context, query string, k int) ([]rag.Retrieved, error) {
		if k <= 0 {
			k = *topk
		}
		return store.Retrieve(ctx, query, rag.RetrieveParams{TopK: k, GraphMix: float32(*mix)})
	}

	searchSchema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "the search query"},
    "top_k": {"type": "integer", "description": "number of chunks to return"}
  },
  "required": ["query"]
}`)
	srv.Register("search", "Search the turbograph corpus and return the most relevant chunks as JSON.",
		searchSchema, func(ctx context.Context, raw json.RawMessage) (string, error) {
			var a searchArgs
			if err := json.Unmarshal(raw, &a); err != nil {
				return "", err
			}
			if a.Query == "" {
				return "", fmt.Errorf("query is required")
			}
			res, err := retrieve(ctx, a.Query, a.TopK)
			if err != nil {
				return "", err
			}
			out := make([]map[string]any, len(res))
			for i, r := range res {
				out[i] = map[string]any{
					"id": r.Chunk.ID, "doc_id": r.Chunk.DocID,
					"score": r.Score, "similarity": r.Similarity, "text": r.Chunk.Text,
				}
			}
			b, _ := json.Marshal(out)
			return string(b), nil
		})

	if *genModel != "" {
		srv.Register("answer", "Answer a question grounded in the turbograph corpus, citing the passages used.",
			searchSchema, func(ctx context.Context, raw json.RawMessage) (string, error) {
				var a searchArgs
				if err := json.Unmarshal(raw, &a); err != nil {
					return "", err
				}
				if a.Query == "" {
					return "", fmt.Errorf("query is required")
				}
				res, err := retrieve(ctx, a.Query, a.TopK)
				if err != nil {
					return "", err
				}
				return client.Generate(ctx, *genModel, ragSystemPrompt, buildPrompt(a.Query, res))
			})
	}

	fmt.Fprintf(os.Stderr, "turbograph mcp serving %s (%d chunks) on stdio\n", *storePath, store.Len())
	return srv.Serve(context.Background(), os.Stdin, os.Stdout)
}
