package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// communitySystemPrompt instructs the model to write a retrieval-oriented summary
// of a cluster of related passages.
const communitySystemPrompt = "You write concise thematic summaries. Given a set of related passages, " +
	"summarize the shared themes, key entities, and notable facts in 3 to 5 sentences. " +
	"Write the summary only, no preamble."

// summarizerFor returns a rag.Summarizer backed by the configured model.
func (s *Server) summarizerFor(model string) rag.Summarizer {
	return func(ctx context.Context, passages []string) (string, error) {
		var sb strings.Builder
		for i, p := range passages {
			fmt.Fprintf(&sb, "[%d] %s\n", i+1, p)
		}
		return s.gen.Generate(ctx, model, communitySystemPrompt, sb.String())
	}
}

// handleBuildCommunities generates a thematic summary for every community of the
// chunk similarity graph, streaming progress. These summaries power global,
// corpus-wide questions (the GraphRAG community-report idea); building them is on
// demand because, unlike the similarity graph, it spends model calls.
func (s *Server) handleBuildCommunities(w http.ResponseWriter, r *http.Request) {
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
	max := 12
	if m := r.URL.Query().Get("max_passages"); m != "" {
		if n, err := strconv.Atoi(m); err == nil && n >= 1 {
			max = n
		}
	}
	err = st.BuildCommunitySummaries(r.Context(), s.summarizerFor(model), rag.CommunityOptions{
		MaxPassages: max,
		OnProgress:  func(done, total int) { send("progress", map[string]int{"done": done, "total": total}) },
	})
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}
	s.persist(bucketOf(r))
	send("done", map[string]int{"communities": len(st.CommunitySummaries())})
}

// handleCommunities lists the generated community summaries.
func (s *Server) handleCommunities(w http.ResponseWriter, r *http.Request) {
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"communities": st.CommunitySummaries()})
}

// chatGlobal answers a corpus-wide question from the community summaries. It
// picks the summaries most relevant to the query, streams an answer synthesized
// across them, and reports those communities as the sources. It shares the SSE
// send closure with handleChat.
func (s *Server) chatGlobal(w http.ResponseWriter, r *http.Request, st *rag.Store, req chatRequest, model string, send func(string, any)) {
	if !st.HasCommunitySummaries() {
		send("error", map[string]string{"error": "no community summaries yet; build them to answer global questions"})
		return
	}
	if s.gen == nil || model == "" {
		send("error", map[string]string{"error": "no language model configured"})
		return
	}
	k := req.TopK
	if k < 4 {
		k = 8 // global questions want a broader thematic base than a local top-k
	}
	comms, err := st.RelevantCommunities(r.Context(), req.Query, k)
	if err != nil {
		send("error", map[string]string{"error": err.Error()})
		return
	}
	if len(comms) == 0 {
		send("abstain", map[string]string{"message": "No themes matched that question."})
		send("done", map[string]bool{"done": true})
		return
	}
	// Report the communities as sources, reusing the source shape so the UI can
	// list them; one representative document id per community.
	srcs := make([]queryResult, len(comms))
	for i, c := range comms {
		doc := fmt.Sprintf("community %d", c.Label)
		if len(c.DocIDs) > 0 {
			doc = fmt.Sprintf("%d docs incl. %s", len(c.DocIDs), c.DocIDs[0])
		}
		srcs[i] = queryResult{ID: fmt.Sprintf("community-%d", c.Label), DocID: doc, Text: c.Summary, Start: -1, End: -1}
	}
	send("sources", map[string]any{"sources": srcs})

	prompt := buildGlobalPrompt(req.Query, comms)
	streamErr := s.gen.GenerateStream(r.Context(), model, communityAnswerSystem, prompt, func(tok string) error {
		send("token", map[string]string{"text": tok})
		return nil
	})
	if streamErr != nil {
		send("error", map[string]string{"error": streamErr.Error()})
		return
	}
	send("done", map[string]bool{"done": true})
}

const communityAnswerSystem = "You answer broad questions about a corpus using the provided section summaries. " +
	"Synthesize across sections, be comprehensive, and cite sections as [n]. If the summaries do not cover it, say so."

// buildGlobalPrompt assembles a map-reduce style prompt for a corpus-wide
// question: the relevant community summaries are the context, and the model
// synthesizes an answer across them rather than from individual passages.
func buildGlobalPrompt(query string, comms []rag.CommunitySummary) string {
	var sb strings.Builder
	sb.WriteString("You are answering a broad question about an entire corpus using thematic summaries of its sections.\n\n")
	sb.WriteString("Section summaries:\n")
	for i, c := range comms {
		fmt.Fprintf(&sb, "[%d] %s\n", i+1, c.Summary)
	}
	sb.WriteString("\nQuestion: ")
	sb.WriteString(query)
	sb.WriteString("\nSynthesize a comprehensive answer across the sections, citing them as [n]. Answer:")
	return sb.String()
}
