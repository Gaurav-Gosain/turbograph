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

// buildEmbedder builds the document/query embedder for the chosen backend.
// "openai" targets any OpenAI-compatible /v1/embeddings endpoint at baseURL;
// anything else uses Ollama (baseURL overrides its default when set). Both satisfy
// rag.Embedder and the asymmetric QueryEmbedder.
func buildEmbedder(api, baseURL, key, model string, dim int) rag.Embedder {
	if api == "openai" {
		c := oai.New(baseURL, key, model)
		c.EmbedDim = dim
		return c
	}
	c := ollama.New()
	c.SetEmbedModel(model)
	c.EmbedDim = dim
	if baseURL != "" {
		c.BaseURL = baseURL
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

// buildBackend builds the generation backend for the chosen provider at baseURL.
func buildBackend(api, baseURL, key string) server.Backend {
	if api == "openai" {
		return oai.New(baseURL, key, "")
	}
	c := ollama.New()
	if baseURL != "" {
		c.BaseURL = baseURL
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

usage:
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

	embedBase, genBase := rc.OllamaURL, rc.OllamaURL
	if rc.EmbedAPI == "openai" {
		embedBase = rc.EmbedURL
	}
	if rc.GenAPI == "openai" {
		genBase = rc.GenURL
	}
	embedder := buildEmbedder(rc.EmbedAPI, embedBase, rc.EmbedKey, rc.EmbedModel, rc.EmbedDim)
	backend := buildBackend(rc.GenAPI, genBase, rc.GenKey)

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
	// Image assets live beside the buckets so the multimodal path (caption then
	// embed) can store and serve figures; a local data dir is required for it.
	if *data != "" {
		if err := srv.EnableAssets(filepath.Join(*data, "assets")); err != nil {
			fmt.Fprintln(os.Stderr, "warning: image ingestion disabled:", err)
		}
	}
	srv.EnableConfig(rc, cfgFile, server.Factories{
		Backend: func(api, url, key string) server.Backend { return buildBackend(api, url, key) },
		Embedder: func(api, url, key, model string, dim int) rag.Embedder {
			return buildEmbedder(api, url, key, model, dim)
		},
	})

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
	fs.Parse(args)

	if *src == "" {
		return fmt.Errorf("--src is required")
	}

	embedBase := *ollamaURL
	if *embedAPI == "openai" {
		embedBase = *embedURL
	}
	embedder := buildEmbedder(*embedAPI, embedBase, orEnv(*embedKey, "OPENAI_API_KEY"), *embedModel, *embedDim)
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
		gen := buildBackend("ollama", *ollamaURL, "")
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
		if err := saveStore(store, *out); err != nil {
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
	format := fs.String("format", "beir", "dataset format: beir")
	corpus := fs.String("corpus", "", "corpus.jsonl path (BEIR)")
	queries := fs.String("queries", "", "queries.jsonl path (BEIR)")
	qrels := fs.String("qrels", "", "qrels TSV path (BEIR)")
	k := fs.Int("k", 10, "cutoff for the @k metrics")
	graphMix := fs.Float64("graph-mix", 0, "Personalized PageRank weight (0 = off, the default)")
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
	fs.Parse(args)

	if *format != "beir" {
		return fmt.Errorf("unknown --format %q (supported: beir)", *format)
	}
	if *corpus == "" || *queries == "" || *qrels == "" {
		return fmt.Errorf("--corpus, --queries, and --qrels are required for BEIR")
	}
	ds, err := bench.LoadBEIR(*corpus, *queries, *qrels)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "loaded %d documents, %d labeled queries\n", len(ds.Docs), len(ds.Cases))

	embedder := buildEmbedder(*embedAPI, *embedURL, orEnv(*embedKey, "OPENAI_API_KEY"), *embedModel, *embedDim)
	cfg := rag.Config{Seed: 1, GraphKNN: *knn, MinSimilarity: 0.5,
		Chunk: rag.ChunkConfig{Strategy: rag.StrategyRecursive, TargetWords: *chunkWords, OverlapWords: *chunkWords / 5}}
	opt := bench.Options{K: *k, DocLevel: true,
		Params:     rag.RetrieveParams{GraphMix: float32(*graphMix), EntityMix: float32(*entityMix), MMRLambda: float32(*mmr)},
		OnProgress: func(done, total int) { fmt.Fprintf(os.Stderr, "\rscoring %d/%d", done, total) }}
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
