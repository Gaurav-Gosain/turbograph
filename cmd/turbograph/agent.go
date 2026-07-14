package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
	"github.com/Gaurav-Gosain/turbograph/server"
)

// This file is the agent-facing command line: the subset of turbograph an agent can
// drive with nothing but a shell. Every agentic harness has a bash tool, so a good
// CLI reaches every one of them without an integration, a server, or a config file.
//
// The shape is deliberate. An agent building a knowledge base over time needs to
// append a note it just learned (add), find what it knew (search), get a grounded
// answer with citations (ask), see and remove what is in there (docs, forget), and
// hand the result to someone else (merge, and the .tg file itself). Every command
// takes --store, defaults it from $TURBOGRAPH_STORE, and speaks JSON on request,
// because an agent parses output rather than reading it.

// storeFlag registers the --store flag, defaulting to $TURBOGRAPH_STORE so an agent
// can export it once per session instead of repeating it on every call.
func storeFlag(fs *flag.FlagSet) *string {
	def := os.Getenv("TURBOGRAPH_STORE")
	if def == "" {
		def = "store.tg"
	}
	return fs.String("store", def, "store path (default $TURBOGRAPH_STORE, else store.tg)")
}

// embedFlags registers the embedding backend flags every store-touching command
// needs, since the store has to be re-embedded through the same model that built it.
type embedOpts struct {
	model, url, api, key *string
	dim                  *int
}

func embedFlags(fs *flag.FlagSet) embedOpts {
	return embedOpts{
		model: fs.String("embed-model", ollama.DefaultEmbedModel, "embedding model (must match the one the store was built with)"),
		dim:   fs.Int("embed-dim", 0, "Matryoshka dimension the store was built with (0 = full)"),
		url:   fs.String("embed-url", "", "base URL for an openai embedding backend"),
		api:   fs.String("embed-api", "ollama", "embedding backend: ollama or openai"),
		key:   fs.String("embed-key", "", "API key for an openai embedding backend (also $OPENAI_API_KEY)"),
	}
}

func (o embedOpts) embedder() rag.Embedder {
	return buildEmbedder(cliEndpoint(*o.api, *o.url, *o.key), *o.model, *o.dim)
}

// openStore loads a store, or returns an empty one when the file does not exist and
// create is set. Creating on first write is what lets an agent start a knowledge base
// with a single command instead of an init step it has to remember.
func openStore(path string, emb rag.Embedder, create bool) (*rag.Store, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) && create {
		return rag.New(emb, rag.Config{}), nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return rag.Load(emb, f)
}

// save writes a store back to its path atomically (see saveStore in main.go).
func save(path string, s *rag.Store) error { return saveStore(s, path, rag.VectorsExact) }

// emit prints v as JSON when asJSON, otherwise runs human for a readable summary.
func emit(asJSON bool, v any, human func()) {
	if asJSON {
		b, _ := json.MarshalIndent(v, "", "  ")
		fmt.Println(string(b))
		return
	}
	human()
}

// cmdAdd appends a note to the knowledge base, creating it if it does not exist.
// The text comes from --text, a file, or stdin, so an agent can pipe what it just
// learned straight in:
//
//	turbograph add --store kb.tg --id "auth-design" <<< "we chose JWT because ..."
func cmdAdd(args []string) error {
	fs := flag.NewFlagSet("add", flag.ExitOnError)
	storePath := storeFlag(fs)
	id := fs.String("id", "", "document id; the same id replaces the previous version (default: a hash of the text)")
	text := fs.String("text", "", "the text to add (default: read stdin)")
	file := fs.String("file", "", "read the text from this file instead of stdin")
	metaRaw := fs.String("meta", "", "arbitrary JSON metadata to attach, e.g. '{\"source\":\"slack\",\"date\":\"2026-07-14\"}'")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	eo := embedFlags(fs)
	fs.Parse(args)

	body := *text
	switch {
	case body != "":
	case *file != "":
		b, err := os.ReadFile(*file)
		if err != nil {
			return err
		}
		body = string(b)
	default:
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		body = string(b)
	}
	body = strings.TrimSpace(body)
	if body == "" {
		return fmt.Errorf("nothing to add: pass --text, --file, or pipe text on stdin")
	}

	docID := *id
	if docID == "" {
		docID = "note-" + shortHash(body) + ".md"
	}
	var meta map[string]any
	if *metaRaw != "" {
		if err := json.Unmarshal([]byte(*metaRaw), &meta); err != nil {
			return fmt.Errorf("--meta is not valid JSON: %w", err)
		}
	}

	store, err := openStore(*storePath, eo.embedder(), true)
	if err != nil {
		return err
	}
	// Work out what this call is actually doing BEFORE doing it. Inferring it afterwards
	// from the chunk count is wrong: replacing a one-chunk document with another
	// one-chunk document leaves the count unchanged, and reporting "nothing was added"
	// tells an agent its correction failed when in fact it landed.
	owner, contentKnown := store.ContentOwner(sha256.Sum256([]byte(body)))
	action := "added"
	switch {
	case contentKnown && owner == docID:
		action = "unchanged" // same id, same text: genuinely a no-op
	case contentKnown:
		action = "duplicate" // this exact text is already here under another id
	case store.HasDoc(docID):
		action = "updated" // same id, new text: a correction
	}

	before := store.Len()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := store.AddDocuments(ctx, []rag.Document{{ID: docID, Text: body, Meta: meta}}); err != nil {
		return err
	}
	delta := store.Len() - before
	if err := save(*storePath, store); err != nil {
		return err
	}

	res := map[string]any{
		"store": *storePath, "id": docID, "action": action, "chunk_delta": delta,
		"chunks": store.Len(), "documents": store.DocCount(),
	}
	if action == "duplicate" {
		res["duplicate_of"] = owner
	}
	emit(*asJSON, res, func() {
		switch action {
		case "unchanged":
			fmt.Printf("%s is already in %s, unchanged\n", docID, *storePath)
		case "duplicate":
			fmt.Printf("that text is already in %s under the id %q; nothing added\n", *storePath, owner)
		case "updated":
			fmt.Printf("updated %s in %s (now %d documents, %d chunks)\n",
				docID, *storePath, store.DocCount(), store.Len())
		default:
			fmt.Printf("added %s to %s (+%d chunks; now %d documents, %d chunks)\n",
				docID, *storePath, delta, store.DocCount(), store.Len())
		}
	})
	return nil
}

// cmdSearch retrieves passages and prints them as JSON by default, because the
// caller is a program.
func cmdSearch(args []string) error {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	storePath := storeFlag(fs)
	q := fs.String("q", "", "the query")
	topk := fs.Int("topk", 6, "passages to return")
	mix := fs.Float64("graph", 0, "graph boost in [0,1]; 0 is off")
	entityMix := fs.Float64("entity", 0, "entity-graph blend in [0,1]; 0 is off")
	mmr := fs.Float64("mmr", 0, "MMR diversity lambda in (0,1); 0 disables")
	maxBytes := fs.Int("max-bytes", 0, "truncate each passage to this many bytes (0 = full text)")
	text := fs.Bool("text", true, "include the passage text")
	human := fs.Bool("human", false, "print a readable summary instead of JSON")
	eo := embedFlags(fs)
	fs.Parse(args)

	if *q == "" {
		return fmt.Errorf("--q is required")
	}
	store, err := openStore(*storePath, eo.embedder(), false)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	res, err := store.Retrieve(ctx, *q, rag.RetrieveParams{
		TopK: *topk, GraphMix: float32(*mix), EntityMix: float32(*entityMix), MMRLambda: float32(*mmr),
	})
	if err != nil {
		return err
	}
	hits := make([]map[string]any, len(res))
	for i, r := range res {
		h := map[string]any{
			"id": r.Chunk.ID, "doc_id": r.Chunk.DocID,
			"score": round3(r.Score), "similarity": round3(r.Similarity),
		}
		if *text {
			t := r.Chunk.Text
			if *maxBytes > 0 {
				t, _ = clipBytes(t, *maxBytes)
			}
			h["text"] = t
		}
		hits[i] = h
	}
	emit(!*human, map[string]any{"query": *q, "hits": hits}, func() {
		if len(hits) == 0 {
			fmt.Println("no matches")
			return
		}
		for i, r := range res {
			fmt.Printf("[%d] %s  score=%.3f\n    %s\n", i+1, r.Chunk.ID, r.Score, truncate(r.Chunk.Text, 240))
		}
	})
	return nil
}

// cmdAsk answers a question from the knowledge base and says which documents the
// answer rests on, so the caller can verify it rather than trust it.
func cmdAsk(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	storePath := storeFlag(fs)
	q := fs.String("q", "", "the question")
	topk := fs.Int("topk", 6, "passages to ground the answer in")
	mix := fs.Float64("graph", 0, "graph boost in [0,1]; 0 is off")
	entityMix := fs.Float64("entity", 0, "entity-graph blend in [0,1]; 0 is off")
	genModel := fs.String("gen-model", envOr("TURBOGRAPH_MODEL", ""), "model for the answer (default $TURBOGRAPH_MODEL)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL")
	asJSON := fs.Bool("json", false, "print the answer and its sources as JSON")
	eo := embedFlags(fs)
	fs.Parse(args)

	if *q == "" {
		return fmt.Errorf("--q is required")
	}
	if *genModel == "" {
		return fmt.Errorf("--gen-model is required (or set $TURBOGRAPH_MODEL); use `turbograph search` for retrieval without a model")
	}
	store, err := openStore(*storePath, eo.embedder(), false)
	if err != nil {
		return err
	}
	client := ollama.New()
	if *ollamaURL != "" {
		client.BaseURL = *ollamaURL
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	res, err := store.Retrieve(ctx, *q, rag.RetrieveParams{
		TopK: *topk, GraphMix: float32(*mix), EntityMix: float32(*entityMix),
	})
	if err != nil {
		return err
	}
	if len(res) == 0 {
		emit(*asJSON, map[string]any{"question": *q, "answer": "", "sources": []any{},
			"note": "nothing in this store matches the question"}, func() {
			fmt.Println("nothing in this store matches the question")
		})
		return nil
	}
	answer, err := client.Generate(ctx, *genModel, ragSystemPrompt, buildPrompt(*q, res))
	if err != nil {
		return err
	}
	answer = strings.TrimSpace(answer)
	sources := make([]map[string]any, len(res))
	for i, r := range res {
		sources[i] = map[string]any{"id": r.Chunk.ID, "doc_id": r.Chunk.DocID, "score": round3(r.Score)}
	}
	emit(*asJSON, map[string]any{"question": *q, "answer": answer, "sources": sources}, func() {
		fmt.Println(answer)
		fmt.Fprintln(os.Stderr, "\nsources:")
		for _, s := range sources {
			fmt.Fprintf(os.Stderr, "  %s\n", s["id"])
		}
	})
	return nil
}

// cmdDocs lists what the knowledge base holds.
func cmdDocs(args []string) error {
	fs := flag.NewFlagSet("docs", flag.ExitOnError)
	storePath := storeFlag(fs)
	asJSON := fs.Bool("json", false, "print the list as JSON")
	fs.Parse(args)

	store, err := openStore(*storePath, nil, false)
	if err != nil {
		return err
	}
	docs := store.Documents()
	out := map[string]any{
		"store": *storePath, "documents": docs,
		"chunks": store.Len(), "entities": store.EntityCount(), "relations": store.RelationCount(),
	}
	emit(*asJSON, out, func() {
		for _, d := range docs {
			fmt.Printf("%-44s %4d chunks  %7d bytes\n", d.ID, d.Chunks, d.Bytes)
		}
		fmt.Printf("\n%d documents, %d chunks", len(docs), store.Len())
		if n := store.EntityCount(); n > 0 {
			fmt.Printf(", %d entities, %d relationships", n, store.RelationCount())
		}
		fmt.Println()
	})
	return nil
}

// cmdForget removes a document. Knowledge that turned out to be wrong has to be
// removable, or the store only ever accumulates and an agent cannot correct itself.
func cmdForget(args []string) error {
	fs := flag.NewFlagSet("forget", flag.ExitOnError)
	storePath := storeFlag(fs)
	id := fs.String("id", "", "document id to remove (see `turbograph docs`)")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	eo := embedFlags(fs)
	fs.Parse(args)

	if *id == "" {
		return fmt.Errorf("--id is required")
	}
	store, err := openStore(*storePath, eo.embedder(), false)
	if err != nil {
		return err
	}
	removed := store.DeleteDocument(*id)
	if removed == 0 {
		return fmt.Errorf("no document %q in %s", *id, *storePath)
	}
	if err := save(*storePath, store); err != nil {
		return err
	}
	emit(*asJSON, map[string]any{"store": *storePath, "id": *id, "chunks_removed": removed,
		"chunks": store.Len(), "documents": store.DocCount()}, func() {
		fmt.Printf("forgot %s (-%d chunks; now %d documents, %d chunks)\n",
			*id, removed, store.DocCount(), store.Len())
	})
	return nil
}

// cmdMerge folds other stores into one. This is what makes a .tg a unit of knowledge
// people and agents can exchange: index separately, swap files, merge.
func cmdMerge(args []string) error {
	fs := flag.NewFlagSet("merge", flag.ExitOnError)
	into := fs.String("into", "", "store to merge into; it is created if it does not exist (required)")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	eo := embedFlags(fs)
	fs.Parse(args)

	if *into == "" {
		return fmt.Errorf("--into is required")
	}
	srcs := fs.Args()
	if len(srcs) == 0 {
		return fmt.Errorf("name at least one store to merge, e.g. turbograph merge --into combined.tg a.tg b.tg")
	}
	emb := eo.embedder()
	dst, err := openStore(*into, emb, true)
	if err != nil {
		return err
	}
	total := rag.MergeStats{}
	per := make([]map[string]any, 0, len(srcs))
	for _, p := range srcs {
		src, err := openStore(p, emb, false)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		st, err := rag.Merge(dst, src)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		total.Documents += st.Documents
		total.Chunks += st.Chunks
		total.Skipped += st.Skipped
		total.Cached += st.Cached
		per = append(per, map[string]any{"store": p, "documents": st.Documents,
			"chunks": st.Chunks, "skipped": st.Skipped})
	}
	if err := save(*into, dst); err != nil {
		return err
	}
	out := map[string]any{"into": *into, "sources": per,
		"documents_added": total.Documents, "chunks_added": total.Chunks,
		"duplicates_skipped": total.Skipped, "extractions_carried": total.Cached,
		"documents": dst.DocCount(), "chunks": dst.Len()}
	if total.Cached > 0 {
		out["note"] = "run `turbograph entities` to rebuild the entity graph; the merged extraction cache makes it nearly free"
	}
	emit(*asJSON, out, func() {
		fmt.Printf("merged %d store(s) into %s: +%d documents, +%d chunks, %d duplicate(s) skipped\n",
			len(srcs), *into, total.Documents, total.Chunks, total.Skipped)
		fmt.Printf("%s now holds %d documents, %d chunks\n", *into, dst.DocCount(), dst.Len())
		if total.Cached > 0 {
			fmt.Printf("carried %d cached extractions; `turbograph entities --store %s` is nearly free\n", total.Cached, *into)
		}
	})
	return nil
}

// cmdEntities builds or refreshes the entity graph on an existing store. It is
// separate from ingest because it is the expensive pass, and because after a merge
// it is the one thing that has to be redone.
func cmdEntities(args []string) error {
	fs := flag.NewFlagSet("entities", flag.ExitOnError)
	storePath := storeFlag(fs)
	genModel := fs.String("gen-model", envOr("TURBOGRAPH_MODEL", ""), "model that reads each chunk (default $TURBOGRAPH_MODEL)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL")
	batch := fs.Int("batch", 2, "chunks per model call; more is faster and measurably lossier on a small model")
	refresh := fs.Bool("refresh", false, "ignore the extraction cache and re-read every chunk")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	eo := embedFlags(fs)
	fs.Parse(args)

	if *genModel == "" {
		return fmt.Errorf("--gen-model is required (or set $TURBOGRAPH_MODEL)")
	}
	store, err := openStore(*storePath, eo.embedder(), false)
	if err != nil {
		return err
	}
	client := ollama.New()
	if *ollamaURL != "" {
		client.BaseURL = *ollamaURL
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
	defer cancel()

	start := time.Now()
	// Redraw the progress line in place only on a terminal. Piped into a file or a
	// program, carriage returns are not erasures and the output becomes one long
	// unreadable smear, so there it prints nothing at all.
	tty := isTerminal(os.Stderr)
	var cached int
	err = buildEntitiesOn(ctx, store, client, *genModel, *batch, *refresh, func(done, total, c int) {
		cached = c
		if tty && !*asJSON {
			fmt.Fprintf(os.Stderr, "\rextracting %d/%d", done, total)
		}
	})
	if tty && !*asJSON {
		fmt.Fprintln(os.Stderr)
	}
	if err != nil {
		return err
	}
	if err := save(*storePath, store); err != nil {
		return err
	}
	emit(*asJSON, map[string]any{"store": *storePath, "entities": store.EntityCount(),
		"relations": store.RelationCount(), "chunks_from_cache": cached,
		"seconds": round3(float32(time.Since(start).Seconds()))}, func() {
		fmt.Printf("entity graph: %d entities, %d relationships in %s",
			store.EntityCount(), store.RelationCount(), time.Since(start).Round(time.Millisecond))
		if cached > 0 {
			fmt.Printf(" (%d chunks answered from cache)", cached)
		}
		fmt.Println()
	})
	return nil
}

// isTerminal reports whether f is a character device, which is the stdlib-only way to
// ask "is a human watching this".
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func round3(f float32) float64 {
	return float64(int(f*1000+0.5)) / 1000
}

// shortHash gives an unnamed note a stable id derived from its content, so adding
// the same note twice is idempotent rather than creating a second copy under a
// second generated name.
func shortHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:6])
}

// buildEntitiesOn runs the extraction pass, reporting progress. It sits here rather
// than in cmdEntities so `ingest --entities` and `entities` share one code path.
func buildEntitiesOn(ctx context.Context, store *rag.Store, gen server.Backend, model string,
	batch int, refresh bool, onProgress func(done, total, cached int),
) error {
	ex := entity.NewLLMExtractor(cliGenerator{c: gen, model: model})
	return store.BuildEntityGraph(ctx, ex, rag.EntityBuildOptions{
		BatchSize: batch,
		Model:     model,
		Refresh:   refresh,
		OnProgress: func(p rag.EntityProgress) {
			if onProgress != nil {
				onProgress(p.Done, p.Total, p.Cached)
			}
		},
	})
}
