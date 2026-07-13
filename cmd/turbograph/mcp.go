package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

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
					// Why this chunk ranked: the additive dense / lexical / graph /
					// entity contributions. Lets an agent judge whether a hit is an
					// exact keyword match or a graph-associated one.
					"components": r.Components,
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

	registerFetchTools(srv, store)

	fmt.Fprintf(os.Stderr, "turbograph mcp serving %s (%d chunks) on stdio\n", *storePath, store.Len())
	return srv.Serve(context.Background(), os.Stdin, os.Stdout)
}

// registerFetchTools adds the retrieval-adjacent tools an agent actually needs
// after a search: pulling the source text back out. Search returns chunks, which
// are sized for embedding, not for reading; an agent usually wants to zoom out to
// the surrounding document, take a specific line range, or pull several documents
// at once without blowing its context window. "get" and "multi_get" cover those,
// with an explicit byte budget so the caller stays in control of how much text
// lands in its context.
func registerFetchTools(srv *mcp.Server, store *rag.Store) {
	getSchema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "doc": {"type": "string", "description": "document id to fetch (from a search result's doc_id)"},
    "chunk": {"type": "string", "description": "chunk id to fetch instead of a whole document (from a search result's id)"},
    "lines": {"type": "string", "description": "line range within the document, e.g. \"50:100\" or \"50\" for from-50-to-end (1-based, inclusive)"},
    "window": {"type": "integer", "description": "when fetching a chunk, also include this many neighbouring chunks on each side for context"},
    "max_bytes": {"type": "integer", "description": "truncate the returned text to this many bytes (default 20000)"}
  }
}`)
	srv.Register("get", "Fetch the source text of a document (optionally a line range) or of a chunk with its surrounding context. Use after search to read the full context of a hit.",
		getSchema, func(_ context.Context, raw json.RawMessage) (string, error) {
			var a struct {
				Doc      string `json:"doc"`
				Chunk    string `json:"chunk"`
				Lines    string `json:"lines"`
				Window   int    `json:"window"`
				MaxBytes int    `json:"max_bytes"`
			}
			if err := json.Unmarshal(raw, &a); err != nil {
				return "", err
			}
			if a.Doc == "" && a.Chunk == "" {
				return "", fmt.Errorf("one of doc or chunk is required")
			}
			text, docID, err := fetchText(store, a.Doc, a.Chunk, a.Lines, a.Window)
			if err != nil {
				return "", err
			}
			text, truncated := clipBytes(text, budget(a.MaxBytes))
			b, _ := json.Marshal(map[string]any{
				"doc_id": docID, "bytes": len(text), "truncated": truncated, "text": text,
			})
			return string(b), nil
		})

	multiSchema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "docs": {"type": "array", "items": {"type": "string"}, "description": "document ids to fetch"},
    "chunks": {"type": "array", "items": {"type": "string"}, "description": "chunk ids to fetch instead of documents"},
    "window": {"type": "integer", "description": "neighbouring chunks to include on each side when fetching chunks"},
    "max_bytes": {"type": "integer", "description": "total byte budget across all items (default 20000); the budget is split evenly and each item is truncated to its share"}
  }
}`)
	srv.Register("multi_get", "Fetch several documents or chunks at once under a total byte budget, so an agent can pull a set of sources without overflowing its context window.",
		multiSchema, func(_ context.Context, raw json.RawMessage) (string, error) {
			var a struct {
				Docs     []string `json:"docs"`
				Chunks   []string `json:"chunks"`
				Window   int      `json:"window"`
				MaxBytes int      `json:"max_bytes"`
			}
			if err := json.Unmarshal(raw, &a); err != nil {
				return "", err
			}
			ids := a.Docs
			asChunk := false
			if len(ids) == 0 {
				ids, asChunk = a.Chunks, true
			}
			if len(ids) == 0 {
				return "", fmt.Errorf("one of docs or chunks is required")
			}
			// Split the budget evenly so one large document cannot starve the rest.
			share := budget(a.MaxBytes) / len(ids)
			if share < 1 {
				share = 1
			}
			items := make([]map[string]any, 0, len(ids))
			for _, id := range ids {
				var text, docID string
				var err error
				if asChunk {
					text, docID, err = fetchText(store, "", id, "", a.Window)
				} else {
					text, docID, err = fetchText(store, id, "", "", 0)
				}
				if err != nil {
					items = append(items, map[string]any{"id": id, "error": err.Error()})
					continue
				}
				text, truncated := clipBytes(text, share)
				items = append(items, map[string]any{
					"id": id, "doc_id": docID, "bytes": len(text), "truncated": truncated, "text": text,
				})
			}
			b, _ := json.Marshal(map[string]any{"items": items})
			return string(b), nil
		})
}

// fetchText resolves a get request to source text: a chunk (optionally widened by
// a neighbour window) or a document (optionally sliced to a 1-based line range).
func fetchText(store *rag.Store, doc, chunk, lines string, window int) (text, docID string, err error) {
	if chunk != "" {
		t := store.ExpandWindow(chunk, window)
		if t == "" {
			return "", "", fmt.Errorf("no chunk %q", chunk)
		}
		return t, docIDOfChunk(store, chunk), nil
	}
	view, ok := store.DocumentView(doc)
	if !ok {
		return "", "", fmt.Errorf("no document %q (or its text was not retained)", doc)
	}
	return sliceLines(view.Text, lines), doc, nil
}

// docIDOfChunk reports the document a chunk belongs to, "" if unknown.
func docIDOfChunk(store *rag.Store, chunkID string) string {
	for _, d := range store.Documents() {
		if strings.HasPrefix(chunkID, d.ID+"#") {
			return d.ID
		}
	}
	return ""
}

// sliceLines returns the 1-based inclusive line range "start:end" of text, or
// "start" for start-to-end. An empty or malformed spec returns the whole text.
func sliceLines(text, spec string) string {
	if spec == "" {
		return text
	}
	all := strings.Split(text, "\n")
	start, end := 1, len(all)
	parts := strings.SplitN(spec, ":", 2)
	if n, err := strconv.Atoi(strings.TrimSpace(parts[0])); err == nil && n > 0 {
		start = n
	}
	if len(parts) == 2 {
		if n, err := strconv.Atoi(strings.TrimSpace(parts[1])); err == nil && n > 0 {
			end = n
		}
	}
	if start > len(all) {
		return ""
	}
	if end > len(all) {
		end = len(all)
	}
	if end < start {
		end = start
	}
	return strings.Join(all[start-1:end], "\n")
}

// budget returns the byte budget for a fetch, defaulting when unset.
func budget(n int) int {
	if n <= 0 {
		return 20000
	}
	return n
}

// clipBytes truncates s to at most n bytes on a rune boundary, reporting whether
// it was cut, so an agent can tell when it is seeing a partial document.
func clipBytes(s string, n int) (string, bool) {
	if len(s) <= n {
		return s, false
	}
	b := []byte(s)[:n]
	for len(b) > 0 && !utf8.Valid(b) {
		b = b[:len(b)-1]
	}
	return string(b), true
}
