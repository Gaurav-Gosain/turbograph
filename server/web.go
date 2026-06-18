package server

import (
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

//go:embed static/index.html
var indexHTML []byte

const chatSystemPrompt = "You are a precise assistant answering from the provided context. " +
	"Use only the context. If it does not contain the answer, say so plainly. " +
	"Do not use emojis or em dashes."

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

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	pdf := s.extract != nil && s.extract.Has("pdf")
	if s.oll == nil {
		writeJSON(w, http.StatusOK, map[string]any{"models": []string{}, "default": "", "pdf": pdf})
		return
	}
	models, err := s.oll.ListModels(r.Context())
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	embed := s.oll.EmbedModel
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
	if s.oll == nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("no ollama server configured"))
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
	err := s.oll.Pull(r.Context(), model, func(p ollama.PullProgress) error {
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

type chatRequest struct {
	Query     string  `json:"query"`
	TopK      int     `json:"top_k"`
	GraphMix  float32 `json:"graph_mix"`
	MMRLambda float32 `json:"mmr_lambda"`
	Model     string  `json:"model"`
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
	res, err := st.Retrieve(r.Context(), req.Query, rag.RetrieveParams{
		TopK:      req.TopK,
		GraphMix:  req.GraphMix,
		MMRLambda: req.MMRLambda,
	})
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}
	send("sources", map[string]any{"sources": toQueryResults(res)})

	if s.oll == nil {
		send("error", map[string]string{"error": "no language model configured"})
		return
	}
	model := req.Model
	if model == "" {
		model = s.genModel
	}
	if model == "" {
		send("error", map[string]string{"error": "no model selected"})
		return
	}

	prompt := buildChatPrompt(req.Query, res)
	streamErr := s.oll.GenerateStream(r.Context(), model, chatSystemPrompt, prompt, func(tok string) error {
		send("token", map[string]string{"text": tok})
		return nil
	})
	if streamErr != nil {
		send("error", map[string]string{"error": streamErr.Error()})
		return
	}
	send("done", map[string]bool{"done": true})
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

func buildChatPrompt(query string, res []rag.Retrieved) string {
	var sb strings.Builder
	sb.WriteString("Context:\n")
	for _, r := range res {
		fmt.Fprintf(&sb, "[%s] %s\n", r.Chunk.ID, r.Chunk.Text)
	}
	sb.WriteString("\nQuestion: ")
	sb.WriteString(query)
	sb.WriteString("\nAnswer:")
	return sb.String()
}
