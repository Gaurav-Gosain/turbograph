package server

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

//go:embed static/index.html
var indexHTML []byte

const chatSystemPrompt = "You are a precise assistant answering from the provided context. " +
	"Use only the context. If it does not contain the answer, say so plainly. " +
	"Cite the passages you rely on with bracketed numbers that match the numbered context, like [1] or [2]. " +
	"Do not use emojis or em dashes."

const rewriteSystem = "Rewrite the user's latest message into a single standalone search query " +
	"using the conversation so far to resolve pronouns and references. " +
	"Reply with only the rewritten query on one line, no quotes, no explanation."

func (s *Server) handleUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}

func (s *Server) handleGraph(w http.ResponseWriter, r *http.Request) {
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, st.GraphView())
}

func (s *Server) handleEntityGraph(w http.ResponseWriter, r *http.Request) {
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, st.EntityGraphView())
}

// genAdapter binds a model to the Ollama client so it satisfies entity.Generator.
type genAdapter struct {
	c     Backend
	model string
}

func (g genAdapter) Generate(ctx context.Context, system, prompt string) (string, error) {
	return g.c.Generate(ctx, g.model, system, prompt)
}

// handleBuildEntities extracts the entity-relationship graph with the language
// model, streaming progress over server-sent events. This is the GraphRAG-style
// indexing pass; it is on demand because it is much more expensive than the
// similarity graph.
func (s *Server) handleBuildEntities(w http.ResponseWriter, r *http.Request) {
	if s.gen == nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("no language model configured"))
		return
	}
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	model := r.URL.Query().Get("model")
	if model == "" {
		model = s.genModel
	}
	if model == "" {
		writeErr(w, http.StatusBadRequest, errEmpty("model"))
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	send := func(event string, v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}
	// Batch several chunks per model call (default 4) to cut round trips; ?batch=1
	// restores one call per chunk for maximum small-model fidelity.
	batch := 4
	if b := r.URL.Query().Get("batch"); b != "" {
		if n, err := strconv.Atoi(b); err == nil && n >= 1 {
			batch = n
		}
	}
	ex := entity.NewLLMExtractor(genAdapter{c: s.gen, model: model})
	err = st.BuildEntityGraph(r.Context(), ex, rag.EntityBuildOptions{
		BatchSize: batch,
		OnProgress: func(p rag.EntityProgress) {
			send("progress", map[string]int{"done": p.Done, "total": p.Total, "entities": p.Entities, "relations": p.Relations})
		},
	})
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}
	s.persist(bucketOf(r))
	send("done", map[string]int{"entities": st.EntityCount()})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	pdf := s.extract != nil && s.extract.Has("pdf")
	if s.gen == nil {
		writeJSON(w, http.StatusOK, map[string]any{"models": []string{}, "default": "", "pdf": pdf})
		return
	}
	models, err := s.gen.ListModels(r.Context())
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	embed := s.embedModel
	embedReady := false
	for _, m := range models {
		if m == embed || strings.HasPrefix(m, embed+":") {
			embedReady = true
			break
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"models":      models,
		"default":     s.genModel,
		"pdf":         pdf,
		"embed_model": embed,
		"embed_ready": embedReady,
	})
}

// handlePull streams the download of a model over server-sent events. It emits
// "progress" events with a status line and byte counts, then "done" or "error".
func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	puller, ok := s.gen.(Puller)
	if !ok {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("the configured backend does not support pulling models"))
		return
	}
	model := r.URL.Query().Get("model")
	if model == "" {
		writeErr(w, http.StatusBadRequest, errEmpty("model"))
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	send := func(event string, v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}
	err := puller.Pull(r.Context(), model, func(p ollama.PullProgress) error {
		send("progress", map[string]any{"status": p.Status, "completed": p.Completed, "total": p.Total})
		return nil
	})
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}
	send("done", map[string]bool{"done": true})
}

type ingestFile struct {
	ID  string `json:"id"`
	B64 string `json:"b64"`
}

type ingestFailure struct {
	ID    string `json:"id"`
	Error string `json:"error"`
}

// handleIngestFiles extracts text from uploaded binary files (for example PDF via
// the configured extractor) and indexes the results. Extraction is error
// tolerant: a file that fails to parse is reported in "failed" and the rest are
// still indexed.
func (s *Server) handleIngestFiles(w http.ResponseWriter, r *http.Request) {
	if s.extract == nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("file extraction is not configured on this server"))
		return
	}
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var req struct {
		Files []ingestFile `json:"files"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var docs []rag.Document
	var failed []ingestFailure
	for _, f := range req.Files {
		data, derr := base64.StdEncoding.DecodeString(f.B64)
		if derr != nil {
			failed = append(failed, ingestFailure{f.ID, "invalid encoding"})
			continue
		}
		text, xerr := s.extract.Extract(r.Context(), f.ID, data)
		if xerr != nil {
			failed = append(failed, ingestFailure{f.ID, xerr.Error()})
			continue
		}
		docs = append(docs, rag.Document{ID: f.ID, Text: text})
	}
	if len(docs) > 0 {
		if err := st.AddDocuments(r.Context(), docs); err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
	}
	saved, saveErr := s.persist(bucketOf(r))
	writeJSON(w, http.StatusOK, map[string]any{
		"chunks":     st.Len(),
		"indexed":    len(docs),
		"failed":     failed,
		"saved":      saved,
		"save_error": saveErr,
	})
}

// handleSave persists the request's bucket to disk. It is a no-op success for an
// in-memory server.
func (s *Server) handleSave(w http.ResponseWriter, r *http.Request) {
	bucket := bucketOf(r)
	if err := s.mgr.Save(bucket); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"saved": true, "bucket": bucket, "path": s.mgr.Path(bucket)})
}

type chatTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Query     string     `json:"query"`
	TopK      int        `json:"top_k"`
	GraphMix  float32    `json:"graph_mix"`
	MMRLambda float32    `json:"mmr_lambda"`
	EntityMix float32    `json:"entity_mix"`
	MinSim    float32    `json:"min_sim"` // abstain if the top hit's cosine is below this
	Rerank    bool       `json:"rerank"`  // pointwise LLM reranking of candidates
	History   []chatTurn `json:"history"` // recent turns, for query rewriting
	Model     string     `json:"model"`
}

// handleChat retrieves context and streams a generated answer over server-sent
// events. It emits a "sources" event first (so the UI can highlight the graph),
// then "token" events as the model produces text, then "done" or "error".
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.Query == "" {
		writeErr(w, http.StatusBadRequest, errEmpty("query"))
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	send := func(event string, v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}

	st, err := s.store(r)
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}
	if req.TopK <= 0 {
		req.TopK = 6
	}
	model := req.Model
	if model == "" {
		model = s.genModel
	}

	res, abstain, err := s.retrieveForChat(r.Context(), st, req, model)
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}

	// Evidence-sufficiency gate: if the best hit is too weak, abstain rather than
	// answer from the model's parametric memory.
	if abstain {
		send("abstain", map[string]string{"message": "I could not find anything relevant in this corpus to answer that."})
		send("done", map[string]bool{"done": true})
		return
	}

	send("sources", map[string]any{"sources": toQueryResults(res)})

	if s.gen == nil || model == "" {
		send("error", map[string]string{"error": "no language model configured"})
		return
	}

	prompt := buildChatPrompt(req.Query, res)
	streamErr := s.gen.GenerateStream(r.Context(), model, chatSystemPrompt, prompt, func(tok string) error {
		send("token", map[string]string{"text": tok})
		return nil
	})
	if streamErr != nil {
		send("error", map[string]string{"error": streamErr.Error()})
		return
	}
	send("done", map[string]bool{"done": true})
}

// retrieveForChat runs the augmented retrieval pipeline shared by the web chat
// and the OpenAI-compatible endpoint: optional conversational query rewriting,
// over-retrieval when reranking, the evidence-sufficiency abstention gate, and
// optional pointwise LLM reranking. It returns the final ranked results, whether
// the gate fired, and any retrieval error. Each stage is independently optional,
// so the cheap default path is identical to plain retrieval.
func (s *Server) retrieveForChat(ctx context.Context, st *rag.Store, req chatRequest, model string) ([]rag.Retrieved, bool, error) {
	// Rewrite an elliptical follow-up into a standalone query for retrieval only;
	// the original question is still what the model answers.
	retrievalQuery := s.rewriteQuery(ctx, req.History, req.Query, model)

	// Over-retrieve when reranking so the reranker has candidates to reorder.
	candK := req.TopK
	if req.Rerank {
		candK = req.TopK * 4
		if candK < 20 {
			candK = 20
		}
	}
	res, err := st.Retrieve(ctx, retrievalQuery, rag.RetrieveParams{
		TopK:      candK,
		GraphMix:  req.GraphMix,
		MMRLambda: req.MMRLambda,
		EntityMix: req.EntityMix,
	})
	if err != nil {
		return nil, false, err
	}
	if rag.ShouldAbstain(res, req.MinSim) {
		return nil, true, nil
	}
	if req.Rerank && model != "" && s.gen != nil {
		res = rag.Rerank(ctx, genAdapter{c: s.gen, model: model}, retrievalQuery, res, req.TopK)
	} else if len(res) > req.TopK {
		res = res[:req.TopK]
	}
	return res, false, nil
}

func toQueryResults(res []rag.Retrieved) []queryResult {
	out := make([]queryResult, len(res))
	for i, r := range res {
		out[i] = queryResult{
			ID:         r.Chunk.ID,
			DocID:      r.Chunk.DocID,
			Score:      r.Score,
			Similarity: r.Similarity,
			Text:       r.Chunk.Text,
		}
	}
	return out
}

// buildChatPrompt numbers the passages [1..k] so the model can cite them with the
// same numbers the UI shows. The numbers, not chunk ids, are the citation tokens.
func buildChatPrompt(query string, res []rag.Retrieved) string {
	var sb strings.Builder
	sb.WriteString("Context:\n")
	for i, r := range res {
		fmt.Fprintf(&sb, "[%d] %s\n", i+1, r.Chunk.Text)
	}
	sb.WriteString("\nQuestion: ")
	sb.WriteString(query)
	sb.WriteString("\nAnswer:")
	return sb.String()
}

// rewriteQuery turns an elliptical follow-up into a standalone search query using
// the recent conversation. It only fires when there is history and the message
// looks dependent (short, or contains a pronoun or reference), and it falls back
// to the original on any weak or malformed rewrite, so it can only help.
func (s *Server) rewriteQuery(ctx context.Context, history []chatTurn, query string, model string) string {
	if s.gen == nil || model == "" || len(history) == 0 || !looksDependent(query) {
		return query
	}
	var sb strings.Builder
	sb.WriteString("Conversation:\n")
	for _, t := range history {
		fmt.Fprintf(&sb, "%s: %s\n", t.Role, t.Content)
	}
	fmt.Fprintf(&sb, "user: %s\n\nStandalone query:", query)
	out, err := s.gen.Generate(ctx, model, rewriteSystem, sb.String())
	if err != nil {
		return query
	}
	out = strings.TrimSpace(out)
	if i := strings.IndexByte(out, '\n'); i >= 0 {
		out = strings.TrimSpace(out[:i])
	}
	out = strings.Trim(out, `"`)
	// Reject empty, overlong, or degenerate rewrites.
	if out == "" || len(out) > 4*len(query)+120 {
		return query
	}
	return out
}

func looksDependent(q string) bool {
	if len(strings.Fields(q)) <= 4 {
		return true
	}
	ql := " " + strings.ToLower(q) + " "
	for _, p := range []string{" it ", " its ", " they ", " them ", " their ", " that ", " those ", " these ", " this ", " he ", " she ", " his ", " her ", " one ", " ones "} {
		if strings.Contains(ql, p) {
			return true
		}
	}
	return false
}
