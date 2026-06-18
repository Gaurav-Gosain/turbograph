// Package server exposes turbograph over HTTP with a small JSON API and an
// embedded web UI. It is dependency-free (standard net/http) so it embeds cleanly
// into any service or runs as a standalone daemon. Requests operate on a named
// bucket (an isolated corpus); the bucket is selected with a "bucket" query
// parameter and defaults to "default".
package server

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/Gaurav-Gosain/turbograph/extract"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// DefaultBucket is used when a request does not name one.
const DefaultBucket = "default"

// Server serves a set of buckets managed by a rag.Manager.
type Server struct {
	mgr      *rag.Manager
	oll      *ollama.Client
	genModel string
	extract  *extract.Registry
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
func (s *Server) SetGenerator(c *ollama.Client, defaultModel string) {
	s.oll = c
	s.genModel = defaultModel
}

// SetExtractor attaches a document extractor registry, enabling binary file
// ingestion (for example PDF) through POST /api/ingest/files.
func (s *Server) SetExtractor(r *extract.Registry) { s.extract = r }

// Handler returns the HTTP routes, including the embedded UI.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /stats", s.handleStats)
	mux.HandleFunc("POST /ingest", s.handleIngest)
	mux.HandleFunc("POST /query", s.handleQuery)

	// OpenAI-compatible chat completions, retrieval-augmented.
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)

	// Web UI and its API.
	mux.HandleFunc("GET /", s.handleUI)
	mux.HandleFunc("GET /api/graph", s.handleGraph)
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/save", s.handleSave)
	mux.HandleFunc("POST /api/ingest/files", s.handleIngestFiles)
	mux.HandleFunc("POST /api/pull", s.handlePull)
	mux.HandleFunc("GET /api/entity-graph", s.handleEntityGraph)
	mux.HandleFunc("POST /api/build-entities", s.handleBuildEntities)

	// Bucket management.
	mux.HandleFunc("GET /api/buckets", s.handleBuckets)
	mux.HandleFunc("POST /api/buckets", s.handleCreateBucket)
	mux.HandleFunc("DELETE /api/buckets", s.handleDeleteBucket)
	return logging(mux)
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

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
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
	ID         string  `json:"id"`
	DocID      string  `json:"doc_id"`
	Score      float32 `json:"score"`
	Similarity float32 `json:"similarity"`
	Text       string  `json:"text"`
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
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, sw.status, time.Since(start).Round(time.Microsecond))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

// Flush forwards to the underlying writer so server-sent events stream through
// the logging middleware instead of being buffered.
func (s *statusWriter) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
