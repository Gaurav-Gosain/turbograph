package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/Gaurav-Gosain/turbograph/mcp"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
	"github.com/Gaurav-Gosain/turbograph/redact"
)

// cmdMCP serves a store to MCP clients (Claude Desktop, editors, agents) over
// stdio. It exposes four tools: "search" returns the top retrieved chunks as
// JSON, "answer" synthesizes a grounded answer with the configured model,
// "add_document" indexes new text, and "forget_document" removes it.
// The transport is line-delimited JSON-RPC on stdin/stdout, so it plugs into any
// MCP host with a command entry.
func cmdMCP(args []string) error {
	fs := flag.NewFlagSet("mcp", flag.ExitOnError)
	storePath := fs.String("store", "store.tg", "store path to serve")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "embedding model (must match the one the store was built with)")
	embedDim := fs.Int("embed-dim", 0, "Matryoshka embedding dimension the store was built with (0 = full)")
	embedAPI := fs.String("embed-api", "ollama", "embedding backend: ollama or openai")
	embedURL := fs.String("embed-url", "", "base URL for an openai embedding backend")
	embedKey := fs.String("embed-key", "", "API key for an openai embedding backend (also $OPENAI_API_KEY)")
	genModel := fs.String("gen-model", "", "model for the answer tool and reranking (omit to expose search only)")
	genAPI := fs.String("llm-api", "ollama", "generation backend: ollama or openai")
	genURL := fs.String("llm-url", "", "base URL for an openai generation backend")
	genKey := fs.String("llm-key", "", "API key for an openai generation backend (also $OPENAI_API_KEY)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	// Retrieval defaults. Each is also a per-call argument on the search and answer
	// tools, so an operator sets the default and an agent tunes it per query.
	topk := fs.Int("topk", 8, "default chunks to retrieve")
	graphMix := fs.Float64("graph-mix", 0, "default graph boost in [0,1]; 0 is off")
	entityMix := fs.Float64("entity-mix", 0, "default entity-graph blend in [0,1]; 0 is off")
	entityLink := fs.String("entity-link", "fact", "how the query links to the entity graph: fact or node")
	mmr := fs.Float64("mmr", 0, "default MMR diversity in (0,1); 0 is off")
	minSim := fs.Float64("min-sim", 0, "default grounding floor; abstain when the top hit's cosine is below this")
	rerank := fs.Bool("rerank", false, "re-score candidates with the model by default")
	noRedact := fs.Bool("no-redact", false, "do not strip credentials from added documents (they would be indexed, and shipped with the store)")
	fs.Parse(args)

	f, err := os.Open(*storePath)
	if err != nil {
		return err
	}
	defer f.Close()

	embedder := buildEmbedder(cliEndpoint(*embedAPI, orStr(*embedURL, *ollamaURL), *embedKey), *embedModel, *embedDim)
	store, err := rag.Load(embedder, f)
	if err != nil {
		return err
	}
	// The generation backend answers and reranks. It may be Ollama or any
	// OpenAI-compatible endpoint, independent of the embedding backend.
	gen := buildBackend(cliEndpoint(*genAPI, orStr(*genURL, *ollamaURL), *genKey))

	srv := mcp.NewServer("turbograph", "1")

	// storeRef guards the in-memory store so write tools can swap in a freshly
	// persisted copy (loaded under the file lock) while search/answer keep reading a
	// consistent snapshot. MCP command dispatch is serial anyway, but the answer
	// tool spawns the model asynchronously while a write may be in flight.
	var storeMu sync.RWMutex
	getStore := func() *rag.Store {
		storeMu.RLock()
		defer storeMu.RUnlock()
		return store
	}

	// mutate runs fn against a fresh copy of the store loaded under the exclusive
	// file lock, persists it, and installs it as the live store. It is the same
	// read-modify-write cycle as the CLI's editStore, so an agent driving the MCP
	// cannot clobber a concurrent `turbograph add`. Returns the mutated store.
	mutate := func(fn func(st *rag.Store) error) (*rag.Store, error) {
		lock, err := rag.TryLockStore(*storePath, false)
		if err != nil {
			return nil, err
		}
		defer lock.Unlock()
		nf, err := os.Open(*storePath)
		if err != nil {
			return nil, err
		}
		defer nf.Close()
		st, err := rag.Load(embedder, nf)
		if err != nil {
			return nil, err
		}
		if err := fn(st); err != nil {
			return nil, err
		}
		if err := save(*storePath, st); err != nil {
			return nil, err
		}
		storeMu.Lock()
		store = st
		storeMu.Unlock()
		return st, nil
	}

	// tuneArgs are the retrieval knobs an agent may set per call. Each field is a pointer
	// so an omitted one falls back to the server default rather than to a zero that would
	// silently turn a feature off.
	type tuneArgs struct {
		Query      string   `json:"query"`
		TopK       int      `json:"top_k"`
		GraphMix   *float64 `json:"graph_mix"`
		EntityMix  *float64 `json:"entity_mix"`
		EntityLink string   `json:"entity_link"`
		MMR        *float64 `json:"mmr"`
		MinSim     *float64 `json:"min_sim"`
		Rerank     *bool    `json:"rerank"`
		MetaFilter []string `json:"meta_filter"`
	}
	f64 := func(p *float64, def float64) float32 {
		if p != nil {
			return float32(*p)
		}
		return float32(def)
	}
	retrieve := func(ctx context.Context, a tuneArgs) ([]rag.Retrieved, error) {
		k := a.TopK
		if k <= 0 {
			k = *topk
		}
		link := a.EntityLink
		if link == "" {
			link = *entityLink
		}
		doRerank := *rerank
		if a.Rerank != nil {
			doRerank = *a.Rerank
		}
		candK := k
		if doRerank && gen != nil && *genModel != "" {
			candK = k * 4
			if candK < 20 {
				candK = 20
			}
		}
		metaFilter, mfErr := rag.CompileMetaFilter(a.MetaFilter)
		if mfErr != nil {
			return nil, mfErr
		}
		res, err := getStore().Retrieve(ctx, a.Query, rag.RetrieveParams{
			TopK:       candK,
			GraphMix:   f64(a.GraphMix, *graphMix),
			EntityMix:  f64(a.EntityMix, *entityMix),
			EntityLink: link,
			MMRLambda:  f64(a.MMR, *mmr),
			MetaFilter: metaFilter,
		})
		if err != nil {
			return nil, err
		}
		floor := f64(a.MinSim, *minSim)
		if rag.ShouldAbstain(res, floor) {
			return nil, nil
		}
		if doRerank && gen != nil && *genModel != "" {
			res = rag.Rerank(ctx, cliGenerator{c: gen, model: *genModel}, a.Query, res, k)
		} else if len(res) > k {
			res = res[:k]
		}
		return res, nil
	}

	tuneSchema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "the search query"},
    "top_k": {"type": "integer", "description": "number of chunks to return"},
    "graph_mix": {"type": "number", "description": "similarity-graph boost in [0,1]; 0 is off. Lifts chunks associated with a strong hit; helps thematic queries, lowers precision on direct lookups"},
    "entity_mix": {"type": "number", "description": "entity knowledge-graph blend in [0,1]; 0 is off. Best on multi-hop questions whose evidence spans documents"},
    "entity_link": {"type": "string", "enum": ["fact", "node"], "description": "how the query links to the entity graph: fact (default, seed from matched relationships) or node (seed from matched entity names)"},
    "mmr": {"type": "number", "description": "MMR diversity in (0,1); 0 is off. Trades some relevance for less redundant results"},
    "min_sim": {"type": "number", "description": "grounding floor: return nothing when the best hit's cosine is below this, rather than surfacing weak matches"},
    "rerank": {"type": "boolean", "description": "re-score the candidates with the model for sharper ordering, at one extra model call"},
    "meta_filter": {"type": "array", "items": {"type": "string"}, "description": "document metadata predicates \"key=value\", AND-composed; only chunks whose whole document matches are returned"}
  },
  "required": ["query"]
}`)

	srv.Register("search", "Search the turbograph corpus and return the most relevant chunks as JSON. Accepts retrieval tuning: graph_mix, entity_mix, entity_link, mmr, min_sim, rerank.",
		tuneSchema, func(ctx context.Context, raw json.RawMessage) (string, error) {
			var a tuneArgs
			if err := json.Unmarshal(raw, &a); err != nil {
				return "", err
			}
			if a.Query == "" {
				return "", fmt.Errorf("query is required")
			}
			res, err := retrieve(ctx, a)
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

	if *genModel != "" && gen != nil {
		srv.Register("answer", "Answer a question grounded in the turbograph corpus, citing the passages used. Accepts the same retrieval tuning as search.",
			tuneSchema, func(ctx context.Context, raw json.RawMessage) (string, error) {
				var a tuneArgs
				if err := json.Unmarshal(raw, &a); err != nil {
					return "", err
				}
				if a.Query == "" {
					return "", fmt.Errorf("query is required")
				}
				res, err := retrieve(ctx, a)
				if err != nil {
					return "", err
				}
				if len(res) == 0 {
					return "Nothing in this corpus is relevant enough to answer that.", nil
				}
				return gen.Generate(ctx, *genModel, ragSystemPrompt, buildPrompt(a.Query, res))
			})
	}

	registerWriteTools(srv, getStore, mutate, *noRedact)

	registerFetchTools(srv, getStore)

	fmt.Fprintf(os.Stderr, "turbograph mcp serving %s (%d chunks, embed=%s, gen=%s) on stdio\n",
		*storePath, store.Len(), *embedModel, orStr(*genModel, "none"))
	return srv.Serve(context.Background(), os.Stdin, os.Stdout)
}

// registerFetchTools adds the retrieval-adjacent tools an agent actually needs
// after a search: pulling the source text back out. Search returns chunks, which
// are sized for embedding, not for reading; an agent usually wants to zoom out to
// the surrounding document, take a specific line range, or pull several documents
// at once without blowing its context window. "get" and "multi_get" cover those,
// with an explicit byte budget so the caller stays in control of how much text
// lands in its context.
func registerFetchTools(srv *mcp.Server, getStore func() *rag.Store) {
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
			text, docID, err := fetchText(getStore(), a.Doc, a.Chunk, a.Lines, a.Window)
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
					text, docID, err = fetchText(getStore(), "", id, "", a.Window)
				} else {
					text, docID, err = fetchText(getStore(), id, "", "", 0)
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

// registerWriteTools adds the tools that grow or shrink the corpus: add_document
// and forget_document. They are the agent-facing shape of the CLI's add/forget:
// the same flock + read-modify-write + atomic save as editStore, with redaction
// on by default. Writes swap the live store under the mutex, so a subsequent
// search immediately sees the new text.
func registerWriteTools(srv *mcp.Server, getStore func() *rag.Store, mutate func(func(*rag.Store) error) (*rag.Store, error), noRedact bool) {
	addSchema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "text": {"type": "string", "description": "the document text to index"},
    "id": {"type": "string", "description": "document id; the same id replaces the previous version (default: a hash of the text)"},
    "meta": {"type": "object", "description": "arbitrary JSON metadata to attach to the document, e.g. {\"source\":\"slack\",\"date\":\"2026-07-14\"}"}
  },
  "required": ["text"]
}`)
	srv.Register("add_document", "Add a text document to the corpus and index it, so later searches can find it. Re-adds the same id with different text as an update; identical text is a no-op. Credentials are stripped by default.",
		addSchema, func(_ context.Context, raw json.RawMessage) (string, error) {
			var a struct {
				Text string         `json:"text"`
				ID   string         `json:"id"`
				Meta map[string]any `json:"meta"`
			}
			if err := json.Unmarshal(raw, &a); err != nil {
				return "", err
			}
			a.Text = strings.TrimSpace(a.Text)
			if a.Text == "" {
				return "", fmt.Errorf("text is required")
			}
			docID := a.ID
			if docID == "" {
				docID = "note-" + shortHash(a.Text) + ".md"
			}
			st, err := mutate(func(st *rag.Store) error {
				st.SetRedaction(!noRedact)
				// Detect duplicate content using the redacted form, exactly like cmdAdd:
				// hash the text the same way it will be stored, then check if some other
				// document id already owns this content.
				stored := a.Text
				if !noRedact {
					stored, _ = redact.Text(a.Text)
				}
				owner, known := st.ContentOwner(sha256.Sum256([]byte(stored)))
				if known && owner != docID {
					return fmt.Errorf("duplicate: that text is already in the store under id %q", owner)
				}
				return st.AddDocuments(context.Background(), []rag.Document{{ID: docID, Text: a.Text, Meta: a.Meta}})
			})
			if err != nil {
				return "", err
			}
			out := map[string]any{"id": docID, "action": "indexed", "documents": st.DocCount(), "chunks": st.Len()}
			b, _ := json.Marshal(out)
			return string(b), nil
		})

	forgetSchema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "id": {"type": "string", "description": "document id to remove (list them with the docs of the search results, or via the CLI: turbograph docs)"}
  },
  "required": ["id"]
}`)
	srv.Register("forget_document", "Remove a document and all its chunks from the corpus. It is gone from retrieval and from the store file. Returns an error if the id is not present.",
		forgetSchema, func(_ context.Context, raw json.RawMessage) (string, error) {
			var a struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(raw, &a); err != nil {
				return "", err
			}
			if a.ID == "" {
				return "", fmt.Errorf("id is required")
			}
			st, err := mutate(func(st *rag.Store) error {
				if st.DeleteDocument(a.ID) == 0 {
					return fmt.Errorf("no document %q in the store", a.ID)
				}
				return nil
			})
			if err != nil {
				return "", err
			}
			out := map[string]any{"forgot": a.ID, "documents": st.DocCount(), "chunks": st.Len()}
			b, _ := json.Marshal(out)
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

// orStr returns a if it is non-empty, else b.
func orStr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
