// Command turbograph is a fast graph-RAG CLI: ingest a corpus into a quantized,
// graph-augmented store, then query it with vector seeds plus PageRank
// propagation, optionally synthesizing an answer with a local model.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/extract"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
	"github.com/Gaurav-Gosain/turbograph/server"
	"github.com/Gaurav-Gosain/turbograph/storage"
)

// splitCmd splits a command string into argv on whitespace. Templates use {in}
// and {out} placeholders, so simple splitting is sufficient.
func splitCmd(s string) []string { return strings.Fields(s) }

// version is set at build time via -ldflags "-X main.version=...". It falls back
// to the module's build info when installed with `go install`.
var version = "dev"

func resolvedVersion() string {
	if version != "dev" {
		return version
	}
	if bi, ok := debug.ReadBuildInfo(); ok && bi.Main.Version != "" && bi.Main.Version != "(devel)" {
		return bi.Main.Version
	}
	return version
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "version", "--version", "-v":
		fmt.Printf("turbograph %s (%s %s/%s)\n", resolvedVersion(), runtime.Version(), runtime.GOOS, runtime.GOARCH)
		return
	case "ingest":
		err = cmdIngest(os.Args[2:])
	case "query":
		err = cmdQuery(os.Args[2:])
	case "stats":
		err = cmdStats(os.Args[2:])
	case "serve":
		err = cmdServe(os.Args[2:])
	case "eval":
		err = cmdEval(os.Args[2:])
	case "mcp":
		err = cmdMCP(os.Args[2:])
	case "quant":
		err = cmdQuant(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `turbograph - fast graph RAG over TurboQuant

usage:
  turbograph ingest --src <dir|file> --out <store> [flags]
  turbograph query  --store <store> --q "<question>" [flags]
  turbograph serve  --data <dir> --addr :8080 [flags]
  turbograph stats  --store <store>
  turbograph eval   --store <store> --suite <suite.jsonl> [flags]
  turbograph mcp    --store <store> [--gen-model M]
  turbograph quant  bench [flags]                             # benchmark the codec
  turbograph version

run a subcommand with -h for its flags.
`)
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	data := fs.String("data", "turbograph-data", "directory of buckets; each <name>.tg is a separate corpus")
	addr := fs.String("addr", ":8080", "listen address")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "ollama embedding model")
	embedDim := fs.Int("embed-dim", 0, "truncate embeddings to this Matryoshka dimension for new buckets (0 = full; e.g. 256 or 512 for embeddinggemma)")
	genModel := fs.String("gen-model", "", "default ollama model for chat (UI can override)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	bits := fs.Int("bits", 4, "quantization bits per coordinate for new buckets")
	knn := fs.Int("knn", 12, "similarity neighbors per chunk for new buckets")
	pdfCmd := fs.String("pdf-cmd", "", "override the pdf extraction command, {in} for the input path (default: pdftotext if present)")
	ocrCmd := fs.String("ocr-cmd", "", "extraction command for scanned PDFs/images via OCR, e.g. a PaddleOCR PP-OCRv6 wrapper")
	s3Bucket := fs.String("s3-bucket", "", "store buckets in an S3-compatible bucket instead of --data")
	s3Endpoint := fs.String("s3-endpoint", "", "S3 endpoint, e.g. https://s3.us-east-1.amazonaws.com or http://localhost:9000")
	s3Region := fs.String("s3-region", "us-east-1", "S3 region")
	s3Prefix := fs.String("s3-prefix", "", "optional key prefix within the bucket")
	apiKey := fs.String("api-key", "", "require this key on all requests (Authorization: Bearer, X-API-Key, or ?api_key=); also via $TURBOGRAPH_API_KEY")
	cors := fs.String("cors", "", "Access-Control-Allow-Origin to send (e.g. * or https://app.example.com); empty disables CORS")
	metrics := fs.Bool("metrics", false, "expose expvar metrics at /debug/vars")
	pprofOn := fs.Bool("pprof", false, "expose the runtime profiler at /debug/pprof/ (behind --api-key when set)")
	maxBody := fs.Int64("max-body", server.DefaultMaxBodyBytes, "max JSON/query request body in bytes (0 uses the default, negative disables)")
	maxUpload := fs.Int64("max-upload", server.DefaultMaxUploadBytes, "max file-upload request body in bytes (0 uses the default, negative disables)")
	fs.Parse(args)

	if *apiKey == "" {
		*apiKey = os.Getenv("TURBOGRAPH_API_KEY")
	}

	client := ollama.New()
	client.SetEmbedModel(*embedModel)
	client.EmbedDim = *embedDim
	if *ollamaURL != "" {
		client.BaseURL = *ollamaURL
	}

	cfg := rag.Config{Bits: *bits, GraphKNN: *knn}
	var mgr *rag.Manager
	var err error
	if *s3Bucket != "" {
		// Credentials come from the environment to keep them off the command line.
		blob, berr := storage.NewS3(storage.S3Config{
			Endpoint:  *s3Endpoint,
			Bucket:    *s3Bucket,
			Region:    *s3Region,
			Prefix:    *s3Prefix,
			AccessKey: os.Getenv("AWS_ACCESS_KEY_ID"),
			SecretKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		})
		if berr != nil {
			return berr
		}
		mgr, err = rag.NewManagerBlob(blob, client, cfg)
	} else {
		mgr, err = rag.NewManager(*data, client, cfg)
	}
	if err != nil {
		return err
	}
	// Ensure the default bucket exists so the UI works immediately.
	if _, err := mgr.GetOrCreate(server.DefaultBucket); err != nil {
		return err
	}
	source := *data
	if *s3Bucket != "" {
		source = "s3://" + *s3Bucket
	}
	fmt.Fprintf(os.Stderr, "serving %d bucket(s) from %s\n", len(mgr.List()), source)

	srv := server.NewManager(mgr)
	srv.SetGenerator(client, *genModel)

	reg := buildRegistry(*pdfCmd, *ocrCmd)
	srv.SetExtractor(reg)
	if reg.Has("pdf") {
		fmt.Fprintln(os.Stderr, "pdf extraction enabled")
	}

	handler := srv.HandlerWithOptions(server.Options{
		APIKey:         *apiKey,
		CORSOrigin:     *cors,
		Metrics:        *metrics,
		Pprof:          *pprofOn,
		MaxBodyBytes:   *maxBody,
		MaxUploadBytes: *maxUpload,
		Version:        resolvedVersion(),
	})
	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		// No write timeout: chat and ingest stream for a long time. Idle and
		// read-header timeouts still bound slow-loris style abuse.
		IdleTimeout: 120 * time.Second,
	}
	if *apiKey != "" {
		fmt.Fprintln(os.Stderr, "API key authentication enabled")
	}
	fmt.Fprintf(os.Stderr, "turbograph UI on %s\n", uiURL(*addr))

	// Serve until interrupted, then drain in-flight requests before exiting.
	errCh := make(chan error, 1)
	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	select {
	case err := <-errCh:
		return err
	case <-sig:
		fmt.Fprintln(os.Stderr, "\nshutting down, draining in-flight requests...")
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		return httpSrv.Shutdown(ctx)
	}
}

func uiURL(a string) string {
	if strings.HasPrefix(a, ":") {
		return "http://localhost" + a
	}
	if !strings.Contains(a, "://") {
		return "http://" + a
	}
	return a
}

// batchingEmbedder splits large embed requests into bounded batches so a big
// corpus does not become one oversized request.
type batchingEmbedder struct {
	c     *ollama.Client
	batch int
}

func (b *batchingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return b.batched(ctx, texts, b.c.Embed)
}

// EmbedQuery forwards to the client's query path so a store created or queried
// through this wrapper keeps the asymmetric query/document embedding.
func (b *batchingEmbedder) EmbedQuery(ctx context.Context, texts []string) ([][]float32, error) {
	return b.batched(ctx, texts, b.c.EmbedQuery)
}

func (b *batchingEmbedder) batched(ctx context.Context, texts []string, embed func(context.Context, []string) ([][]float32, error)) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += b.batch {
		end := min(i+b.batch, len(texts))
		vecs, err := embed(ctx, texts[i:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func cmdIngest(args []string) error {
	fs := flag.NewFlagSet("ingest", flag.ExitOnError)
	src := fs.String("src", "", "source file or directory to ingest")
	out := fs.String("out", "store.tg", "store path; loaded and extended if it already exists")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "ollama embedding model")
	embedDim := fs.Int("embed-dim", 0, "truncate embeddings to this Matryoshka dimension (0 = full; e.g. 256 or 512 for embeddinggemma)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	bits := fs.Int("bits", 4, "quantization bits per coordinate (1-8)")
	residual := fs.Int("residual", 32, "QJL residual projections (0-64)")
	knn := fs.Int("knn", 12, "similarity neighbors per chunk in the graph")
	target := fs.Int("chunk-words", 120, "target chunk size in words")
	overlap := fs.Int("chunk-overlap", 24, "chunk overlap in words")
	minSim := fs.Float64("min-sim", 0.5, "minimum cosine similarity for a graph edge")
	batch := fs.Int("batch", 64, "embedding request batch size per document")
	workers := fs.Int("workers", 0, "documents embedded concurrently (0 = number of CPUs)")
	checkpoint := fs.Int("checkpoint", 200, "save the store every N documents for crash recovery (0 = only at end)")
	pdfCmd := fs.String("pdf-cmd", "", "override the pdf extraction command, {in} for input path")
	ocrCmd := fs.String("ocr-cmd", "", "OCR command for scanned pdfs and images, e.g. a PaddleOCR PP-OCRv6 wrapper")
	entities := fs.Bool("entities", false, "after indexing, extract an entity-relationship knowledge graph (GraphRAG style)")
	entModel := fs.String("gen-model", "", "model used to extract entities when --entities is set")
	fs.Parse(args)

	if *src == "" {
		return fmt.Errorf("--src is required")
	}

	client := ollama.New()
	client.SetEmbedModel(*embedModel)
	client.EmbedDim = *embedDim
	if *ollamaURL != "" {
		client.BaseURL = *ollamaURL
	}
	emb := &batchingEmbedder{c: client, batch: *batch}

	// Cancel on the first interrupt so ingestion stops cleanly after checkpointing.
	// A second interrupt aborts immediately.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sig := make(chan os.Signal, 2)
	signal.Notify(sig, os.Interrupt)
	go func() {
		<-sig
		fmt.Fprintln(os.Stderr, "\ninterrupt received, finishing current work and checkpointing...")
		cancel()
		<-sig
		os.Exit(130)
	}()

	pingCtx, pingCancel := context.WithTimeout(ctx, 5*time.Second)
	err := client.Ping(pingCtx)
	pingCancel()
	if err != nil {
		return fmt.Errorf("ollama unreachable at %s: %w", client.BaseURL, err)
	}

	reg := buildRegistry(*pdfCmd, *ocrCmd)

	// Load an existing store to resume and extend it, or create a fresh one.
	var store *rag.Store
	if f, e := os.Open(*out); e == nil {
		store, err = rag.Load(emb, f)
		f.Close()
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "resuming store %s with %d documents\n", *out, store.DocCount())
	} else {
		store = rag.New(emb, rag.Config{
			Bits: *bits, ResidualDims: *residual, GraphKNN: *knn,
			MinSimilarity: float32(*minSim),
			Chunk:         rag.ChunkConfig{TargetWords: *target, OverlapWords: *overlap},
		})
	}

	journal, err := rag.OpenJournal(*out + ".journal")
	if err != nil {
		return err
	}
	defer journal.Close()

	docs := streamDocs(ctx, *src, reg, journal, store)
	start := time.Now()
	prog, ingErr := store.Ingest(ctx, docs, 0, rag.IngestOptions{
		Workers:         *workers,
		Journal:         journal,
		Save:            func() error { return saveStore(store, *out) },
		CheckpointEvery: *checkpoint,
		OnProgress: func(p rag.Progress) {
			fmt.Fprintf(os.Stderr, "\rdone %d  failed %d  skipped %d  chunks %d", p.Done, p.Failed, p.Skipped, p.Chunks)
		},
	})
	fmt.Fprintln(os.Stderr)

	if err := saveStore(store, *out); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "indexed %d new documents (%d chunks) in %s; store has %d documents, %d chunks\n",
		prog.Done, prog.Chunks, time.Since(start).Round(time.Millisecond), store.DocCount(), store.Len())
	fmt.Fprintf(os.Stderr, "saved to %s\n", *out)

	if ingErr == context.Canceled {
		fmt.Fprintf(os.Stderr, "paused. re-run the same command to resume from %s.journal\n", *out)
		return nil
	}
	if ingErr != nil {
		return ingErr
	}

	if *entities {
		if *entModel == "" {
			return fmt.Errorf("--entities requires --gen-model for extraction")
		}
		fmt.Fprintf(os.Stderr, "extracting entity graph with %s ...\n", *entModel)
		ex := entity.NewLLMExtractor(cliGenerator{c: client, model: *entModel})
		eerr := store.BuildEntityGraph(ctx, ex, rag.EntityBuildOptions{
			OnProgress: func(p rag.EntityProgress) {
				fmt.Fprintf(os.Stderr, "\rextracting %d/%d  entities %d  relations %d", p.Done, p.Total, p.Entities, p.Relations)
			},
		})
		fmt.Fprintln(os.Stderr)
		if eerr != nil {
			return eerr
		}
		if err := saveStore(store, *out); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "entity graph: %d entities, saved to %s\n", store.EntityCount(), *out)
	}
	return nil
}

// cliGenerator binds a model to the Ollama client for entity extraction.
type cliGenerator struct {
	c     *ollama.Client
	model string
}

func (g cliGenerator) Generate(ctx context.Context, system, prompt string) (string, error) {
	return g.c.Generate(ctx, g.model, system, prompt)
}

// buildRegistry assembles the extractor registry with optional command overrides.
func buildRegistry(pdfCmd, ocrCmd string) *extract.Registry {
	reg := extract.DefaultRegistry()
	if pdfCmd != "" {
		reg.Register("pdf", extract.CommandFromTemplate(splitCmd(pdfCmd)))
	}
	if ocrCmd != "" {
		reg.Register("pdf", extract.CommandFromTemplate(splitCmd(ocrCmd)))
		for _, ext := range []string{"png", "jpg", "jpeg", "tiff", "webp"} {
			reg.Register(ext, extract.CommandFromTemplate(splitCmd(ocrCmd)))
		}
	}
	return reg
}

// streamDocs walks src and emits a Document per supported file, extracting text
// through the registry. Files already ingested are skipped without being read,
// and extraction failures are logged and skipped rather than aborting the run.
func streamDocs(ctx context.Context, src string, reg *extract.Registry, j *rag.Journal, store *rag.Store) <-chan rag.Document {
	ch := make(chan rag.Document)
	go func() {
		defer close(ch)
		root := src
		info, err := os.Stat(src)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return
		}
		if !info.IsDir() {
			root = filepath.Dir(src)
		}
		walkFn := func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(path), "."))
			if !reg.Has(ext) {
				return nil
			}
			id, rerr := filepath.Rel(root, path)
			if rerr != nil {
				id = filepath.Base(path)
			}
			if j.Done(id) || store.HasDoc(id) {
				return nil
			}
			data, rerr := os.ReadFile(path)
			if rerr != nil {
				fmt.Fprintf(os.Stderr, "\nskip %s: %v\n", id, rerr)
				return nil
			}
			text, rerr := reg.Extract(ctx, path, data)
			if rerr != nil {
				fmt.Fprintf(os.Stderr, "\nextract %s: %v\n", id, rerr)
				return nil
			}
			select {
			case ch <- rag.Document{ID: id, Text: text}:
			case <-ctx.Done():
				return filepath.SkipAll
			}
			return nil
		}
		if info.IsDir() {
			filepath.WalkDir(src, walkFn)
		} else {
			walkFn(src, dirEntry{info}, nil)
		}
	}()
	return ch
}

// dirEntry adapts a FileInfo to a DirEntry for the single-file ingestion path.
type dirEntry struct{ fi os.FileInfo }

func (d dirEntry) Name() string               { return d.fi.Name() }
func (d dirEntry) IsDir() bool                { return d.fi.IsDir() }
func (d dirEntry) Type() os.FileMode          { return d.fi.Mode().Type() }
func (d dirEntry) Info() (os.FileInfo, error) { return d.fi, nil }

func saveStore(store *rag.Store, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return store.Save(f)
}

func cmdQuery(args []string) error {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	storePath := fs.String("store", "store.tg", "store path")
	q := fs.String("q", "", "query text")
	topk := fs.Int("topk", 8, "number of chunks to retrieve")
	mix := fs.Float64("mix", 0, "graph boost added on top of relevance; 0 is off (graph is opt-in)")
	lexWeight := fs.Float64("lexical-weight", 0, "BM25 weight added to dense relevance; 0 uses the default, negative forces pure dense")
	mmr := fs.Float64("mmr", 0, "MMR diversity lambda in (0,1); 0 disables")
	prf := fs.Int("prf", 0, "pseudo-relevance feedback documents to expand the query with (0 disables)")
	prfWeight := fs.Float64("prf-weight", 0.5, "how strongly PRF feedback is mixed into the query")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "ollama embedding model")
	embedDim := fs.Int("embed-dim", 0, "Matryoshka dimension the store was built with (must match ingest; 0 = full)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	genModel := fs.String("gen-model", "", "ollama model for answer synthesis (empty to only list context)")
	showText := fs.Bool("show", true, "print retrieved chunk text")
	fs.Parse(args)

	if *q == "" {
		return fmt.Errorf("--q is required")
	}
	f, err := os.Open(*storePath)
	if err != nil {
		return err
	}
	defer f.Close()

	client := ollama.New()
	client.SetEmbedModel(*embedModel)
	client.EmbedDim = *embedDim
	if *ollamaURL != "" {
		client.BaseURL = *ollamaURL
	}
	store, err := rag.Load(client, f)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	start := time.Now()
	res, err := store.Retrieve(ctx, *q, rag.RetrieveParams{
		TopK: *topk, GraphMix: float32(*mix), MMRLambda: float32(*mmr),
		PRF: *prf, PRFWeight: float32(*prfWeight), LexicalWeight: float32(*lexWeight),
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "retrieved %d chunks in %s\n\n", len(res), time.Since(start).Round(time.Microsecond))

	for i, r := range res {
		fmt.Printf("[%d] %s  score=%.3f sim=%.3f\n", i+1, r.Chunk.ID, r.Score, r.Similarity)
		if *showText {
			fmt.Printf("    %s\n", truncate(r.Chunk.Text, 280))
		}
	}

	if *genModel != "" {
		fmt.Fprintln(os.Stderr, "\nsynthesizing answer...")
		answer, err := client.Generate(ctx, *genModel, ragSystemPrompt, buildPrompt(*q, res))
		if err != nil {
			return err
		}
		fmt.Printf("\n%s\n", strings.TrimSpace(answer))
	}
	return nil
}

func cmdStats(args []string) error {
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	storePath := fs.String("store", "store.tg", "store path")
	fs.Parse(args)
	f, err := os.Open(*storePath)
	if err != nil {
		return err
	}
	defer f.Close()
	store, err := rag.Load(nil, f)
	if err != nil {
		return err
	}
	fmt.Printf("chunks: %d\n", store.Len())
	return nil
}

const ragSystemPrompt = "You are a precise assistant. Answer the question using only the provided context. " +
	"If the context does not contain the answer, say so. Cite the chunk ids you used."

func buildPrompt(q string, res []rag.Retrieved) string {
	var sb strings.Builder
	sb.WriteString("Context:\n")
	for _, r := range res {
		fmt.Fprintf(&sb, "[%s] %s\n", r.Chunk.ID, r.Chunk.Text)
	}
	sb.WriteString("\nQuestion: ")
	sb.WriteString(q)
	sb.WriteString("\nAnswer:")
	return sb.String()
}

func truncate(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
