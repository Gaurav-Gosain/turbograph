// Package server exposes turbograph over HTTP with a small JSON API and an
// embedded web UI. It is dependency-free (standard net/http) so it embeds cleanly
// into any service or runs as a standalone daemon. Requests operate on a named
// bucket (an isolated corpus); the bucket is selected with a "bucket" query
// parameter and defaults to "default".
package server

import (
	"context"
	"encoding/json"
	"expvar"
	"log"
	"net/http"
	"net/http/pprof"
	"time"

	"github.com/Gaurav-Gosain/turbograph/extract"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// DefaultBucket is used when a request does not name one.
const DefaultBucket = "default"

// Backend is the generation/model surface the server needs. Both ollama.Client
// and oai.Client (any OpenAI-compatible endpoint) satisfy it, so the server is
// not tied to a single provider.
type Backend interface {
	Generate(ctx context.Context, model, system, prompt string) (string, error)
	GenerateStream(ctx context.Context, model, system, prompt string, onToken func(string) error) error
	ListModels(ctx context.Context) ([]string, error)
	Ping(ctx context.Context) error
}

// Puller is the optional model-download surface; only the Ollama backend provides
// it, so pull-related UI is offered only when the backend implements this.
type Puller interface {
	Pull(ctx context.Context, model string, onProgress func(ollama.PullProgress) error) error
}

// Server serves a set of buckets managed by a rag.Manager.
type Server struct {
	mgr        *rag.Manager
	gen        Backend
	genModel   string
	embedModel string
	version    string
	extract    *extract.Registry

	// Runtime configuration, editable over /api/config when EnableConfig was called.
	cfg       RuntimeConfig
	cfgPath   string
	factories Factories

	// assets stores ingested image bytes for the multimodal path; nil disables it.
	assets *assetStore
}

// New returns a server backed by a single store, exposed as the "default" bucket.
// It is the simple, in-memory entry point; use NewManager for multitenancy.
func New(store *rag.Store) *Server {
	mgr, _ := rag.NewManager("", store.Embedder(), rag.Config{})
	mgr.Put(DefaultBucket, store)
	return &Server{mgr: mgr}
}

// NewManager returns a server backed by a bucket manager.
func NewManager(mgr *rag.Manager) *Server { return &Server{mgr: mgr} }

// SetGenerator attaches an Ollama client and default generation model, enabling
// the chat and model-listing endpoints.
func (s *Server) SetGenerator(c Backend, defaultModel, embedModel string) {
	s.gen = c
	s.genModel = defaultModel
	s.embedModel = embedModel
}

// SetExtractor attaches a document extractor registry, enabling binary file
// ingestion (for example PDF) through POST /api/ingest/files.
func (s *Server) SetExtractor(r *extract.Registry) { s.extract = r }

// Handler returns the HTTP routes wrapped in the default hardening middleware
// (panic recovery, body-size limits, the access log). For auth, CORS, or metrics,
// use HandlerWithOptions.
func (s *Server) Handler() http.Handler {
	return s.HandlerWithOptions(Options{})
}

// HandlerWithOptions returns the routes wrapped in the configured middleware.
func (s *Server) HandlerWithOptions(opt Options) http.Handler {
	s.version = opt.Version
	if s.version == "" {
		s.version = "dev"
	}
	return chain(s.routes(opt), opt)
}

// routes builds the bare route table (no middleware).
func (s *Server) routes(opt Options) http.Handler {
	version := opt.Version
	if version == "" {
		version = "dev"
	}
	mux := http.NewServeMux()
	// Health captures the version at build time, so no mutable server state is
	// shared with concurrent requests.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "version": version})
	})
	mux.HandleFunc("GET /readyz", s.handleReady)
	if opt.Metrics {
		mux.Handle("GET /debug/vars", expvar.Handler())
	}
	if opt.Pprof {
		// Registered on our mux (not DefaultServeMux), so it stays behind auth.
		mux.HandleFunc("GET /debug/pprof/", pprof.Index)
		mux.HandleFunc("GET /debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("GET /debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("GET /debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("GET /debug/pprof/trace", pprof.Trace)
	}
	mux.HandleFunc("GET /api/status", s.handleStatus)
	// Core endpoints. The /api/* paths are canonical and match the rest of the
	// surface; the short paths are kept as aliases so existing clients keep working.
	mux.HandleFunc("GET /stats", s.handleStats)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("POST /ingest", s.handleIngest)
	mux.HandleFunc("POST /api/ingest", s.handleIngest)
	mux.HandleFunc("POST /query", s.handleQuery)
	mux.HandleFunc("POST /api/query", s.handleQuery)

	// OpenAI-compatible chat completions, retrieval-augmented.
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)

	// Web UI and its API.
	mux.HandleFunc("GET /", s.handleUI)
	mux.HandleFunc("GET /api/documents", s.handleDocuments)
	mux.HandleFunc("GET /api/document", s.handleDocument)
	mux.HandleFunc("DELETE /api/document", s.handleDocument)
	mux.HandleFunc("GET /api/versions", s.handleVersions)
	mux.HandleFunc("GET /api/version", s.handleVersionText)
	mux.HandleFunc("POST /api/restore", s.handleRestore)
	mux.HandleFunc("GET /api/graph", s.handleGraph)
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/save", s.handleSave)
	mux.HandleFunc("POST /api/ingest/files", s.handleIngestFiles)
	mux.HandleFunc("POST /api/ingest/image", s.handleIngestImage)
	mux.HandleFunc("GET /api/asset/{id}", s.handleAsset)
	mux.HandleFunc("POST /api/pull", s.handlePull)
	mux.HandleFunc("GET /api/entity-graph", s.handleEntityGraph)
	mux.HandleFunc("POST /api/build-entities", s.handleBuildEntities)
	mux.HandleFunc("GET /api/communities", s.handleCommunities)
	mux.HandleFunc("POST /api/build-communities", s.handleBuildCommunities)

	// Runtime configuration.
	mux.HandleFunc("GET /api/config", s.handleGetConfig)
	mux.HandleFunc("POST /api/config", s.handlePostConfig)

	// Bucket management.
	mux.HandleFunc("GET /api/buckets", s.handleBuckets)
	mux.HandleFunc("POST /api/buckets", s.handleCreateBucket)
	mux.HandleFunc("DELETE /api/buckets", s.handleDeleteBucket)
	return mux
}

// bucketOf returns the bucket named by the request, or the default.
func bucketOf(r *http.Request) string {
	if b := r.URL.Query().Get("bucket"); b != "" {
		return b
	}
	return DefaultBucket
}

// store resolves the request's bucket, creating it on first use.
func (s *Server) store(r *http.Request) (*rag.Store, error) {
	return s.mgr.GetOrCreate(bucketOf(r))
}

// handleReady is a readiness probe: the process is up (liveness) and its
// dependencies are reachable. When a generator is configured it pings Ollama, so
// an orchestrator can hold traffic until the model backend is actually available.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if s.gen != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()
		if err := s.gen.Ping(ctx); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready", "ollama": err.Error()})
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// handleStatus aggregates everything a transparency view needs in one call: the
// version, where data is stored, which model backends are configured and whether
// the generation backend is reachable, and the current bucket's stats.
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	storage := "local disk"
	if s.cfg.S3Bucket != "" {
		storage = "s3://" + s.cfg.S3Bucket
	}
	genReady := false
	if s.gen != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		genReady = s.gen.Ping(ctx) == nil
		cancel()
	}
	out := map[string]any{
		"version": s.version,
		"storage": map[string]any{
			"backend":  map[bool]string{true: "s3", false: "local"}[s.cfg.S3Bucket != ""],
			"location": storage, "endpoint": s.cfg.S3Endpoint,
		},
		"generation": map[string]any{"api": orStr(s.cfg.GenAPI, "ollama"), "model": s.genModel, "reachable": genReady},
		"embedding":  map[string]any{"api": orStr(s.cfg.EmbedAPI, "ollama"), "model": s.embedModel, "dim": s.cfg.EmbedDim},
		"buckets":    len(s.mgr.List()),
		"bucket":     bucketOf(r),
	}
	if st, err := s.store(r); err == nil {
		stats := map[string]any{"chunks": st.Len(), "documents": st.DocCount(), "entities": st.EntityCount()}
		if c := st.Communities(); c != nil {
			stats["communities"] = c.NumCommunities()
		}
		stats["chunk_strategy"] = st.Config().Chunk.Strategy
		out["stats"] = stats
	}
	writeJSON(w, http.StatusOK, out)
}

func orStr(v, d string) string {
	if v == "" {
		return d
	}
	return v
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	resp := map[string]any{"chunks": st.Len(), "documents": st.DocCount(), "entities": st.EntityCount()}
	if c := st.Communities(); c != nil {
		resp["communities"] = c.NumCommunities()
	}
	writeJSON(w, http.StatusOK, resp)
}

type ingestRequest struct {
	Documents []rag.Document `json:"documents"`
	// Replace rebuilds from scratch; otherwise documents are added incrementally.
	Replace bool `json:"replace"`
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var req ingestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if len(req.Documents) == 0 {
		writeErr(w, http.StatusBadRequest, errEmpty("documents"))
		return
	}
	if req.Replace {
		err = st.Build(r.Context(), req.Documents)
	} else {
		err = st.AddDocuments(r.Context(), req.Documents)
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	saved, saveErr := s.persist(bucketOf(r))
	writeJSON(w, http.StatusOK, map[string]any{"chunks": st.Len(), "saved": saved, "save_error": saveErr})
}

// persist saves a bucket after a mutation. It is a no-op success for an in-memory
// server. A save error is returned to the caller rather than failing the request,
// since the in-memory result is already correct and the caller can surface it.
func (s *Server) persist(bucket string) (bool, string) {
	if err := s.mgr.Save(bucket); err != nil {
		return false, err.Error()
	}
	return true, ""
}

type queryRequest struct {
	Query     string  `json:"query"`
	TopK      int     `json:"top_k"`
	GraphMix  float32 `json:"graph_mix"`
	MMRLambda float32 `json:"mmr_lambda"`
	EntityMix float32 `json:"entity_mix"`
}

type queryResult struct {
	ID         string          `json:"id"`
	DocID      string          `json:"doc_id"`
	Score      float32         `json:"score"`
	Similarity float32         `json:"similarity"`
	Text       string          `json:"text"`
	Start      int             `json:"start"` // rune offset of this chunk in its document
	End        int             `json:"end"`
	Meta       json.RawMessage `json:"meta,omitempty"`      // the source document's metadata
	Kind       string          `json:"kind,omitempty"`      // "image" for an image-derived chunk
	ImageRef   string          `json:"image_ref,omitempty"` // asset id of the source image
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var req queryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Query == "" {
		writeErr(w, http.StatusBadRequest, errEmpty("query"))
		return
	}
	res, err := st.Retrieve(r.Context(), req.Query, rag.RetrieveParams{
		TopK:      req.TopK,
		GraphMix:  req.GraphMix,
		MMRLambda: req.MMRLambda,
		EntityMix: req.EntityMix,
	})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": toQueryResults(res)})
}

// handleBuckets lists buckets with basic stats.
func (s *Server) handleBuckets(w http.ResponseWriter, _ *http.Request) {
	type bucketInfo struct {
		Name        string `json:"name"`
		Chunks      int    `json:"chunks"`
		Documents   int    `json:"documents"`
		Communities int    `json:"communities"`
	}
	names := s.mgr.List()
	out := make([]bucketInfo, 0, len(names))
	for _, n := range names {
		st, ok := s.mgr.Get(n)
		if !ok {
			continue
		}
		info := bucketInfo{Name: n, Chunks: st.Len(), Documents: st.DocCount()}
		if c := st.Communities(); c != nil {
			info.Communities = c.NumCommunities()
		}
		out = append(out, info)
	}
	writeJSON(w, http.StatusOK, map[string]any{"buckets": out, "default": DefaultBucket})
}

func (s *Server) handleCreateBucket(w http.ResponseWriter, r *http.Request) {
	name := bucketOf(r)
	if _, err := s.mgr.Create(name); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"created": name})
}

func (s *Server) handleDeleteBucket(w http.ResponseWriter, r *http.Request) {
	name := bucketOf(r)
	if name == DefaultBucket {
		writeErr(w, http.StatusBadRequest, errEmpty("a non-default bucket name"))
		return
	}
	if err := s.mgr.Delete(name); err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": name})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

type errEmpty string

func (e errEmpty) Error() string { return string(e) + " is required" }

// logging wraps a handler with a minimal access log: method, path, status, and
// latency, emitted after the response so it never interferes with headers.
func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := asStatus(w)
		next.ServeHTTP(sw, r)
		status := sw.status
		if status == 0 {
			status = http.StatusOK
		}
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, status, time.Since(start).Round(time.Microsecond))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

// asStatus returns w as a *statusWriter, wrapping it only if it is not already
// one. recoverPanic is the outermost middleware and installs the wrapper, so the
// inner layers (metrics, logging) reuse it instead of each allocating their own.
func asStatus(w http.ResponseWriter) *statusWriter {
	if s, ok := w.(*statusWriter); ok {
		return s
	}
	return &statusWriter{ResponseWriter: w}
}

func (s *statusWriter) WriteHeader(code int) {
	if s.status == 0 {
		s.status = code
	}
	s.ResponseWriter.WriteHeader(code)
}

// Write records that the response body has started (status defaults to 200) so
// the panic guard knows not to try writing a 500 over an in-flight response.
func (s *statusWriter) Write(b []byte) (int, error) {
	if s.status == 0 {
		s.status = http.StatusOK
	}
	return s.ResponseWriter.Write(b)
}

// Flush forwards to the underlying writer so server-sent events stream through
// the logging middleware instead of being buffered.
func (s *statusWriter) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
