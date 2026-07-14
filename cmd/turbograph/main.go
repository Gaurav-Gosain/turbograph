// Command turbograph is a fast graph-RAG CLI: ingest a corpus into a quantized,
// graph-augmented store, then query it with vector seeds plus PageRank
// propagation, optionally synthesizing an answer with a local model.
package main

import (
	"context"
	"encoding/json"
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

	"github.com/Gaurav-Gosain/turbograph/bench"
	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/extract"
	"github.com/Gaurav-Gosain/turbograph/oai"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
	"github.com/Gaurav-Gosain/turbograph/script"
	"github.com/Gaurav-Gosain/turbograph/server"
	"github.com/Gaurav-Gosain/turbograph/storage"
)

// orEnv returns v, or the named environment variable when v is empty.
func orEnv(v, env string) string {
	if v != "" {
		return v
	}
	return os.Getenv(env)
}

// buildEmbedder builds the document/query embedder for a resolved endpoint.
// "openai" targets any OpenAI-compatible /v1/embeddings endpoint; anything else
// uses Ollama (the base URL overrides its default when set). Both satisfy
// rag.Embedder and the asymmetric QueryEmbedder.
func buildEmbedder(ep server.Endpoint, model string, dim int) rag.Embedder {
	if ep.API == "openai" {
		c := oai.New(ep.BaseURL, ep.APIKey, model)
		c.Headers = ep.Headers
		c.EmbedDim = dim
		return c
	}
	c := ollama.New()
	c.SetEmbedModel(model)
	c.EmbedDim = dim
	if ep.BaseURL != "" {
		c.BaseURL = ep.BaseURL
	}
	return c
}

// pingBackend checks reachability when the backend supports it (both clients do).
func pingBackend(ctx context.Context, e any) error {
	if p, ok := e.(interface {
		Ping(context.Context) error
	}); ok {
		return p.Ping(ctx)
	}
	return nil
}

// cliEndpoint resolves the flag-driven backend selection to an endpoint. The
// command line has no provider list, so it is the inline "ollama"/"openai" pair.
func cliEndpoint(api, baseURL, key string) server.Endpoint {
	if api == "openai" {
		return server.Endpoint{API: "openai", BaseURL: baseURL, APIKey: orEnv(key, "OPENAI_API_KEY")}
	}
	return server.Endpoint{API: "ollama", BaseURL: baseURL}
}

// buildBackend builds the generation backend for a resolved endpoint.
func buildBackend(ep server.Endpoint) server.Backend {
	if ep.API == "openai" {
		c := oai.New(ep.BaseURL, ep.APIKey, "")
		c.Headers = ep.Headers
		return c
	}
	c := ollama.New()
	if ep.BaseURL != "" {
		c.BaseURL = ep.BaseURL
	}
	return c
}

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
	case "add":
		err = cmdAdd(os.Args[2:])
	case "search":
		err = cmdSearch(os.Args[2:])
	case "ask":
		err = cmdAsk(os.Args[2:])
	case "docs":
		err = cmdDocs(os.Args[2:])
	case "forget":
		err = cmdForget(os.Args[2:])
	case "merge":
		err = cmdMerge(os.Args[2:])
	case "entities":
		err = cmdEntities(os.Args[2:])
	case "skill":
		err = cmdSkill(os.Args[2:])
	case "ingest":
		err = cmdIngest(os.Args[2:])
	case "query":
		err = cmdQuery(os.Args[2:])
	case "stats":
		err = cmdStats(os.Args[2:])
	case "export":
		err = cmdExport(os.Args[2:])
	case "bench":
		err = cmdBench(os.Args[2:])
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

a knowledge base is a single .tg file. build one up over time, then share it.

knowledge base (these are what an agent drives from a shell):
  turbograph add      --store kb.tg [--id ID] [--text T | --file F | < stdin]
  turbograph search   --store kb.tg --q "<query>" [--topk N]        # JSON passages
  turbograph ask      --store kb.tg --q "<question>" --gen-model M  # grounded answer + sources
  turbograph docs     --store kb.tg                                 # what is in it
  turbograph forget   --store kb.tg --id ID                         # remove a document
  turbograph merge    --into combined.tg a.tg b.tg                  # combine shared stores
  turbograph entities --store kb.tg --gen-model M                   # build the entity graph
  turbograph skill                                                  # print the agent skill

  $TURBOGRAPH_STORE and $TURBOGRAPH_MODEL supply the defaults for --store and --gen-model.

bulk and serving:
  turbograph ingest --src <dir|file> --out <store> [flags]
  turbograph query  --store <store> --q "<question>" [flags]
  turbograph serve  --data <dir> --addr :8080 [flags]
  turbograph stats  --store <store>
  turbograph export --store <store> [--out <file.json>] [--no-vectors]   # JSON view for interop
  turbograph eval   --store <store> --suite <suite.jsonl> [flags]
  turbograph bench  --format beir --corpus c.jsonl --queries q.jsonl --qrels q.tsv  # reproduce metrics
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
	genModel := fs.String("gen-model", "", "default chat model (UI can override)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	embedAPI := fs.String("embed-api", "ollama", "embedding backend: ollama or openai (any OpenAI-compatible endpoint)")
	embedURL := fs.String("embed-url", "", "base URL for an openai embedding backend (e.g. https://api.openai.com)")
	embedKey := fs.String("embed-key", "", "API key for an openai embedding backend (also via $OPENAI_API_KEY)")
	llmAPI := fs.String("llm-api", "ollama", "generation backend: ollama or openai (any OpenAI-compatible endpoint)")
	llmURL := fs.String("llm-url", "", "base URL for an openai generation backend")
	llmKey := fs.String("llm-key", "", "API key for an openai generation backend (also via $OPENAI_API_KEY)")
	bits := fs.Int("bits", 4, "quantization bits per coordinate for new buckets")
	knn := fs.Int("knn", 12, "similarity neighbors per chunk for new buckets")
	chunkStrategy := fs.String("chunk-strategy", rag.StrategyRecursive, "chunking strategy for new buckets: recursive, word, markdown, or sentence")
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
	configPath := fs.String("config", "", "persisted settings JSON, editable in the UI (default <data>/config.json)")
	lean := fs.String("lean", "exact", "vector storage when buckets are saved: exact (float32), codes (compact TurboQuant, ~40% size, ~98% recall), or text (no vectors, re-embed on load, smallest and lossless)")
	scriptsDir := fs.String("scripts", "", "directory of executable transform scripts callers may run over documents at ingest, by name (SECURITY: every program in it becomes runnable by anyone who can reach the API; leave unset to disable)")
	scriptTimeout := fs.Duration("script-timeout", script.DefaultTimeout, "how long a single transform script may run before it is killed")
	fs.Parse(args)

	if *apiKey == "" {
		*apiKey = os.Getenv("TURBOGRAPH_API_KEY")
	}

	// Assemble the runtime config from flags, then let a saved config file (the
	// UI's persisted edits) take precedence so settings survive restarts.
	rc := server.RuntimeConfig{
		GenAPI: *llmAPI, GenURL: *llmURL, GenKey: orEnv(*llmKey, "OPENAI_API_KEY"), GenModel: *genModel,
		EmbedAPI: *embedAPI, EmbedURL: *embedURL, EmbedKey: orEnv(*embedKey, "OPENAI_API_KEY"), EmbedModel: *embedModel, EmbedDim: *embedDim,
		OllamaURL:     *ollamaURL,
		ChunkStrategy: *chunkStrategy,
		S3Endpoint:    *s3Endpoint, S3Bucket: *s3Bucket, S3Region: *s3Region, S3Prefix: *s3Prefix,
	}
	cfgFile := *configPath
	if cfgFile == "" && *data != "" {
		cfgFile = filepath.Join(*data, "config.json")
	}
	if loaded, ok, lerr := server.LoadConfig(cfgFile); lerr != nil {
		return lerr
	} else if ok {
		rc = loaded
		rc.GenKey = orEnv(rc.GenKey, "OPENAI_API_KEY")
		rc.EmbedKey = orEnv(rc.EmbedKey, "OPENAI_API_KEY")
		fmt.Fprintf(os.Stderr, "loaded settings from %s\n", cfgFile)
	}

	embedder := buildEmbedder(rc.EmbedEndpoint(), rc.EmbedModel, rc.EmbedDim)
	backend := buildBackend(rc.GenEndpoint())

	cfg := rag.Config{Bits: *bits, GraphKNN: *knn, Chunk: rag.ChunkConfig{
		Strategy: rc.ChunkStrategy, TargetWords: rc.ChunkWords, OverlapWords: rc.ChunkOverlap,
	}}
	var mgr *rag.Manager
	var err error
	if rc.S3Bucket != "" {
		// Credentials come from the environment to keep them off the command line.
		blob, berr := storage.NewS3(storage.S3Config{
			Endpoint:  rc.S3Endpoint,
			Bucket:    rc.S3Bucket,
			Region:    rc.S3Region,
			Prefix:    rc.S3Prefix,
			AccessKey: os.Getenv("AWS_ACCESS_KEY_ID"),
			SecretKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		})
		if berr != nil {
			return berr
		}
		mgr, err = rag.NewManagerBlob(blob, embedder, cfg)
	} else {
		mgr, err = rag.NewManager(*data, embedder, cfg)
	}
	if err != nil {
		return err
	}
	if *lean != "exact" {
		mode, merr := parseVectorMode(*lean)
		if merr != nil {
			return merr
		}
		mgr.SetVectorMode(mode)
	}
	// Ensure the default bucket exists so the UI works immediately.
	if _, err := mgr.GetOrCreate(server.DefaultBucket); err != nil {
		return err
	}
	source := *data
	if rc.S3Bucket != "" {
		source = "s3://" + rc.S3Bucket
	}
	fmt.Fprintf(os.Stderr, "serving %d bucket(s) from %s\n", len(mgr.List()), source)

	srv := server.NewManager(mgr)
	srv.SetGenerator(backend, rc.GenModel, rc.EmbedModel)
	// Transform scripts are registered by the operator here, at startup. Requests
	// may only name what is in this directory, never supply a command of their own.
	scripts, serr := script.Load(*scriptsDir, *scriptTimeout)
	if serr != nil {
		return serr
	}
	srv.SetScripts(scripts)
	if n := scripts.Len(); n > 0 {
		fmt.Fprintf(os.Stderr, "transform scripts: %d from %s (%s)\n", n, *scriptsDir, strings.Join(scripts.Names(), ", "))
	}
	// Image assets live beside the buckets so the multimodal path (caption then
	// embed) can store and serve figures; a local data dir is required for it.
	if *data != "" {
		if err := srv.EnableAssets(filepath.Join(*data, "assets")); err != nil {
			fmt.Fprintln(os.Stderr, "warning: image ingestion disabled:", err)
		}
	}
	srv.EnableConfig(rc, cfgFile, server.Factories{Backend: buildBackend, Embedder: buildEmbedder})

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
	inner rag.Embedder // an ollama.Client or oai.Client
	batch int
}

func (b *batchingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return b.batched(ctx, texts, b.inner.Embed)
}

// EmbedQuery forwards to the inner query path (when the embedder distinguishes
// queries) so a store created or queried through this wrapper keeps asymmetric
// embedding.
func (b *batchingEmbedder) EmbedQuery(ctx context.Context, texts []string) ([][]float32, error) {
	if qe, ok := b.inner.(rag.QueryEmbedder); ok {
		return b.batched(ctx, texts, qe.EmbedQuery)
	}
	return b.batched(ctx, texts, b.inner.Embed)
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
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "embedding model")
	embedDim := fs.Int("embed-dim", 0, "truncate embeddings to this Matryoshka dimension (0 = full; e.g. 256 or 512 for embeddinggemma)")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
	embedAPI := fs.String("embed-api", "ollama", "embedding backend: ollama or openai (any OpenAI-compatible endpoint)")
	embedURL := fs.String("embed-url", "", "base URL for an openai embedding backend")
	embedKey := fs.String("embed-key", "", "API key for an openai embedding backend (also via $OPENAI_API_KEY)")
	bits := fs.Int("bits", 4, "quantization bits per coordinate (1-8)")
	residual := fs.Int("residual", 32, "QJL residual projections (0-64)")
	knn := fs.Int("knn", 12, "similarity neighbors per chunk in the graph")
	target := fs.Int("chunk-words", 120, "target chunk size in words")
	overlap := fs.Int("chunk-overlap", 24, "chunk overlap in words")
	chunkStrategy := fs.String("chunk-strategy", rag.StrategyRecursive, "chunking strategy: recursive, word, markdown, or sentence")
	minSim := fs.Float64("min-sim", 0.5, "minimum cosine similarity for a graph edge")
	batch := fs.Int("batch", 64, "embedding request batch size per document")
	workers := fs.Int("workers", 0, "documents embedded concurrently (0 = number of CPUs)")
	checkpoint := fs.Int("checkpoint", 200, "save the store every N documents for crash recovery (0 = only at end)")
	pdfCmd := fs.String("pdf-cmd", "", "override the pdf extraction command, {in} for input path")
	ocrCmd := fs.String("ocr-cmd", "", "OCR command for scanned pdfs and images, e.g. a PaddleOCR PP-OCRv6 wrapper")
	entities := fs.Bool("entities", false, "after indexing, extract an entity-relationship knowledge graph (GraphRAG style)")
	entModel := fs.String("gen-model", "", "model used to extract entities when --entities is set")
	entBatch := fs.Int("entity-batch", 4, "chunks per model call during entity extraction (1 = one call per chunk; higher is faster but can dilute a small model)")
	lean := fs.String("lean", "exact", "vector storage in the .tg file: exact (float32), codes (compact TurboQuant, ~40% size, ~98% recall), or text (no vectors, re-embed on load, smallest and lossless)")
	fs.Parse(args)

	if *src == "" {
		return fmt.Errorf("--src is required")
	}
	vmode, verr := parseVectorMode(*lean)
	if verr != nil {
		return verr
	}

	embedBase := *ollamaURL
	if *embedAPI == "openai" {
		embedBase = *embedURL
	}
	embedder := buildEmbedder(cliEndpoint(*embedAPI, embedBase, *embedKey), *embedModel, *embedDim)
	emb := &batchingEmbedder{inner: embedder, batch: *batch}

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
	err := pingBackend(pingCtx, embedder)
	pingCancel()
	if err != nil {
		return fmt.Errorf("embedding backend unreachable: %w", err)
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
			Chunk:         rag.ChunkConfig{Strategy: *chunkStrategy, TargetWords: *target, OverlapWords: *overlap},
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
		Save:            func() error { return saveStore(store, *out, vmode) },
		CheckpointEvery: *checkpoint,
		OnProgress: func(p rag.Progress) {
			fmt.Fprintf(os.Stderr, "\rdone %d  failed %d  skipped %d  chunks %d", p.Done, p.Failed, p.Skipped, p.Chunks)
		},
	})
	fmt.Fprintln(os.Stderr)

	if err := saveStore(store, *out, vmode); err != nil {
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
		gen := buildBackend(server.Endpoint{API: "ollama", BaseURL: *ollamaURL})
		ex := entity.NewLLMExtractor(cliGenerator{c: gen, model: *entModel})
		eerr := store.BuildEntityGraph(ctx, ex, rag.EntityBuildOptions{
			BatchSize: *entBatch,
			OnProgress: func(p rag.EntityProgress) {
				fmt.Fprintf(os.Stderr, "\rextracting %d/%d  entities %d  relations %d", p.Done, p.Total, p.Entities, p.Relations)
			},
		})
		fmt.Fprintln(os.Stderr)
		if eerr != nil {
			return eerr
		}
		if err := saveStore(store, *out, vmode); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "entity graph: %d entities, saved to %s\n", store.EntityCount(), *out)
	}
	return nil
}

// cliGenerator binds a model to the Ollama client for entity extraction.
type cliGenerator struct {
	c     server.Backend
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

// saveStore writes the store through a temporary file and renames it into place.
// os.Create truncates the target first, so a crash or a full disk part-way through a
// write left a corrupt .tg where the corpus used to be, which is precisely the case
// ingest checkpointing exists to survive. A rename is atomic, so the old store stands
// until the new one is complete.
func saveStore(store *rag.Store, path string, mode rag.VectorMode) error {
	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp, err := os.CreateTemp(dir, ".tg-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op once the rename succeeds
	if err := store.SaveLean(tmp, mode); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// parseVectorMode maps the --lean flag to a VectorMode. "exact" (or "" / "full")
// stores raw float32; "codes" stores compact TurboQuant codes (about 40% the
// size, decoded on load, ~98% recall); "text"/"none" stores no vectors and
// re-embeds from text on load (smallest and lossless, but the load re-embeds the
// whole corpus with the same model).
func parseVectorMode(s string) (rag.VectorMode, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "exact", "full", "float32":
		return rag.VectorsExact, nil
	case "codes", "quant", "quantized":
		return rag.VectorsCodes, nil
	case "text", "none", "recompute":
		return rag.VectorsNone, nil
	default:
		return rag.VectorsExact, fmt.Errorf("unknown --lean mode %q (use exact, codes, or text)", s)
	}
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

// cmdExport transcodes a .tg store (Go gob) to JSON, the interop format other
// languages and tools can read. It streams straight from the file without
// rebuilding any indexes, so no embedder is needed.
func cmdExport(args []string) error {
	fs := flag.NewFlagSet("export", flag.ExitOnError)
	storePath := fs.String("store", "store.tg", "store path")
	out := fs.String("out", "", "output file (default: stdout)")
	noVectors := fs.Bool("no-vectors", false, "omit embeddings (much smaller)")
	fs.Parse(args)
	f, err := os.Open(*storePath)
	if err != nil {
		return err
	}
	defer f.Close()
	w := os.Stdout
	if *out != "" {
		of, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer of.Close()
		w = of
	}
	if err := rag.ExportJSON(f, w, !*noVectors); err != nil {
		return err
	}
	if *out != "" {
		fmt.Fprintf(os.Stderr, "exported %s -> %s\n", *storePath, *out)
	}
	return nil
}

// cmdBench reproduces the retrieval benchmark numbers on a labeled dataset using
// a real embedder. It ingests the corpus, scores every labeled query, and prints
// the metric table, so the headline numbers in docs/benchmarks.md can be
// regenerated by anyone with a model server.
func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	format := fs.String("format", "beir", "dataset format: beir or multihop")
	corpus := fs.String("corpus", "", "corpus.jsonl path (BEIR)")
	queries := fs.String("queries", "", "queries.jsonl path (BEIR)")
	qrels := fs.String("qrels", "", "qrels TSV path (BEIR)")
	k := fs.Int("k", 10, "cutoff for the @k metrics")
	graphMix := fs.Float64("graph-mix", 0, "Personalized PageRank weight (0 = off, the default)")
	ablate := fs.Bool("ablate", false, "score every retrieval configuration against one index, so the report says which part is doing the work")
	entityMix := fs.Float64("entity-mix", 0, "entity-graph weight")
	mmr := fs.Float64("mmr", 0, "MMR diversification lambda (0 = off)")
	jsonOut := fs.String("json", "", "also write the report as JSON to this path")
	embedModel := fs.String("embed-model", ollama.DefaultEmbedModel, "embedding model")
	embedDim := fs.Int("embed-dim", 0, "Matryoshka embedding dimension (0 = full)")
	embedAPI := fs.String("embed-api", "ollama", "embedding backend: ollama or openai")
	embedURL := fs.String("embed-url", "", "base URL for the embedding backend")
	embedKey := fs.String("embed-key", "", "API key for an openai embedding backend (also $OPENAI_API_KEY)")
	knn := fs.Int("knn", 12, "similarity neighbors per chunk in the graph")
	chunkWords := fs.Int("chunk-words", 120, "target chunk size in words")
	chunkStrategy := fs.String("chunk-strategy", rag.StrategyRecursive, "chunking strategy: recursive, word, markdown, sentence")
	buildStore := fs.String("build-store", "", "ingest the dataset, build the entity graph, save a .tg here, and exit (needs --gen-model)")
	storePath := fs.String("store", "", "score against a prebuilt .tg instead of ingesting the corpus (built with --build-store)")
	genModel := fs.String("gen-model", "", "model for entity extraction when building a store")
	ollamaURL := fs.String("ollama-url", "", "Ollama base URL")
	entBatch := fs.Int("ent-batch", 2, "chunks per model call during entity extraction")
	fs.Parse(args)

	var ds *bench.Dataset
	var err error
	switch *format {
	case "beir":
		if *corpus == "" || *queries == "" || *qrels == "" {
			return fmt.Errorf("--corpus, --queries, and --qrels are required for BEIR")
		}
		ds, err = bench.LoadBEIR(*corpus, *queries, *qrels)
	case "multihop":
		if *corpus == "" || *queries == "" {
			return fmt.Errorf("--corpus and --queries are required for multihop")
		}
		ds, err = bench.LoadMultiHopRAG(*corpus, *queries)
	default:
		return fmt.Errorf("unknown --format %q (supported: beir, multihop)", *format)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "loaded %d documents, %d labeled queries\n", len(ds.Docs), len(ds.Cases))

	embedder := buildEmbedder(cliEndpoint(*embedAPI, *embedURL, *embedKey), *embedModel, *embedDim)
	cfg := rag.Config{Seed: 1, GraphKNN: *knn, MinSimilarity: 0.5,
		Chunk: rag.ChunkConfig{Strategy: *chunkStrategy, TargetWords: *chunkWords, OverlapWords: *chunkWords / 5}}

	// Build a store once, with the entity graph, and save it. The entity pass is an
	// hours-long LLM run on a real corpus, so it is done once and the A/B arms score
	// against the saved store rather than rebuilding it per arm.
	if *buildStore != "" {
		if *genModel == "" {
			return fmt.Errorf("--build-store needs --gen-model for entity extraction")
		}
		return benchBuildStore(ds, embedder, cfg, *buildStore, *genModel, *ollamaURL, *entBatch)
	}
	opt := bench.Options{K: *k, DocLevel: true,
		Params:     rag.RetrieveParams{GraphMix: float32(*graphMix), EntityMix: float32(*entityMix), MMRLambda: float32(*mmr)},
		OnProgress: func(done, total int) { fmt.Fprintf(os.Stderr, "\rscoring %d/%d", done, total) },
		OnArm: func(i, n int, name string) {
			fmt.Fprintf(os.Stderr, "\r\033[Karm %d/%d: %s\n", i+1, n, name)
		}}

	if *ablate {
		// The arms answer the only question worth asking: which part is doing the work.
		// Relevance is dense + w*bm25, so sweeping w from "off" through the default to
		// lexical-dominant shows exactly what the embedder contributes and what BM25 adds
		// on top of it. The graph, entity and MMR arms then have to justify themselves
		// against that baseline rather than being assumed to help.
		arms := []bench.Arm{
			{Name: "dense only (bm25 off)", Params: rag.RetrieveParams{LexicalWeight: -1}},
			{Name: "hybrid w=0.25 (default)", Params: rag.RetrieveParams{}},
			{Name: "hybrid w=0.50", Params: rag.RetrieveParams{LexicalWeight: 0.5}},
			{Name: "hybrid w=1.0", Params: rag.RetrieveParams{LexicalWeight: 1.0}},
			{Name: "lexical-dominant w=8", Params: rag.RetrieveParams{LexicalWeight: 8}},
			{Name: "+ graph 0.2", Params: rag.RetrieveParams{GraphMix: 0.2}},
			{Name: "+ entity 0.3", Params: rag.RetrieveParams{EntityMix: 0.3}},
			{Name: "+ entity 0.5", Params: rag.RetrieveParams{EntityMix: 0.5}},
			{Name: "+ MMR 0.5", Params: rag.RetrieveParams{MMRLambda: 0.5}},
		}
		var res []bench.ArmResult
		var store *rag.Store
		if *storePath != "" {
			// Score against the prebuilt store, which carries the entity graph. This is how
			// the entity arms become real: without a built graph, EntityMix is a silent no-op.
			f, oerr := os.Open(*storePath)
			if oerr != nil {
				return oerr
			}
			store, err = rag.Load(embedder, f)
			f.Close()
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "loaded %s: %d chunks, %d entities, %d relationships\n",
				*storePath, store.Len(), store.EntityCount(), store.RelationCount())
			res = bench.AblateStore(context.Background(), store, ds, arms, opt)
		} else {
			res, store, err = bench.Ablate(context.Background(), embedder, cfg, ds, arms, opt)
			if err != nil {
				return err
			}
		}
		fmt.Fprintln(os.Stderr)
		fmt.Printf("%-24s %8s %8s %8s %10s %10s\n", "arm", "ndcg@"+itoa(*k), "recall", "mrr", "p50", "p95")
		base := res[1].Report.Mean.NDCGAtK // the hybrid default, for the delta column
		for _, r := range res {
			m := r.Report.Mean
			d := ""
			if m.NDCGAtK != base {
				d = fmt.Sprintf("  %+.4f", m.NDCGAtK-base)
			}
			fmt.Printf("%-24s %8.4f %8.4f %8.4f %10s %10s%s\n", r.Name,
				m.NDCGAtK, m.RecallAtK, m.MRR,
				r.QueryP50.Round(time.Microsecond), r.QueryP95.Round(time.Microsecond), d)
		}
		fmt.Fprintf(os.Stderr, "\n%d documents, %d chunks, %d queries\n", len(ds.Docs), store.Len(), len(ds.Cases))
		if *jsonOut != "" {
			b, _ := json.MarshalIndent(res, "", "  ")
			if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "wrote %s\n", *jsonOut)
		}
		return nil
	}

	rep, err := bench.Evaluate(context.Background(), embedder, cfg, ds, opt)
	if err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr)
	m := rep.Mean
	fmt.Printf("metric          @%d\n", *k)
	fmt.Printf("recall          %.4f\n", m.RecallAtK)
	fmt.Printf("precision       %.4f\n", m.PrecisionAtK)
	fmt.Printf("ndcg            %.4f\n", m.NDCGAtK)
	fmt.Printf("mrr             %.4f\n", m.MRR)
	fmt.Printf("context_prec    %.4f\n", m.ContextPrecisionAtK)
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *jsonOut)
	}
	return nil
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

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

// benchBuildStore ingests a benchmark corpus, builds the entity graph, and saves the
// store. It is separated from scoring because the entity pass is the expensive part: on
// a real corpus it is an hours-long LLM run, done once here so the A/B arms can score
// the saved store repeatedly without paying for it again.
func benchBuildStore(ds *bench.Dataset, embedder rag.Embedder, cfg rag.Config, out, genModel, ollamaURL string, batch int) error {
	store := rag.New(embedder, cfg)
	fmt.Fprintf(os.Stderr, "ingesting %d documents...\n", len(ds.Docs))
	ctx := context.Background()
	start := time.Now()
	if err := store.Build(ctx, ds.Docs); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}
	fmt.Fprintf(os.Stderr, "ingested %d chunks in %s; extracting the entity graph with %s...\n",
		store.Len(), time.Since(start).Round(time.Second), genModel)

	client := ollama.New()
	if ollamaURL != "" {
		client.BaseURL = ollamaURL
	}
	extStart := time.Now()
	last := time.Now()
	err := buildEntitiesOn(ctx, store, client, genModel, batch, false, func(done, total, cached int) {
		if time.Since(last) > 5*time.Second {
			last = time.Now()
			el := time.Since(extStart)
			rate := float64(done) / el.Seconds()
			eta := time.Duration(float64(total-done)/max(rate, 0.01)) * time.Second
			fmt.Fprintf(os.Stderr, "\rextracting %d/%d chunks (%.1f/s, eta %s)      ",
				done, total, rate, eta.Round(time.Minute))
		}
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return fmt.Errorf("entity extraction: %w", err)
	}
	fmt.Fprintf(os.Stderr, "entity graph: %d entities, %d relationships in %s\n",
		store.EntityCount(), store.RelationCount(), time.Since(extStart).Round(time.Second))

	if err := saveStore(store, out, rag.VectorsExact); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "saved %s\n", out)
	return nil
}
