package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// This file implements a minimal OpenAI-compatible surface so existing chat
// clients and SDKs can talk to turbograph unchanged. Only chat completions are
// implemented, and every request is retrieval-augmented: the corpus is injected
// as numbered context before the model answers. The bucket is selected with the
// usual "bucket" query parameter. Embeddings and model listing are deliberately
// out of scope; this is a RAG chat endpoint wearing a familiar shape.

type oaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type oaiChatRequest struct {
	Model    string       `json:"model"`
	Messages []oaiMessage `json:"messages"`
	Stream   bool         `json:"stream"`
	// Retrieval knobs, accepted as extensions and ignored by stock clients.
	TopK      int     `json:"top_k"`
	GraphMix  float32 `json:"graph_mix"`
	MMRLambda float32 `json:"mmr_lambda"`
	EntityMix float32 `json:"entity_mix"`
	MinSim    float32 `json:"min_sim"`
	Rerank    bool    `json:"rerank"`
}

type oaiChoice struct {
	Index        int         `json:"index"`
	Message      *oaiMessage `json:"message,omitempty"`
	Delta        *oaiMessage `json:"delta,omitempty"`
	FinishReason *string     `json:"finish_reason"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type oaiResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []oaiChoice `json:"choices"`
	Usage   *oaiUsage   `json:"usage,omitempty"`
}

// toChatRequest maps an OpenAI request onto the internal chat request. The last
// user message is the query; the earlier messages become history for rewriting.
func (req oaiChatRequest) toChatRequest() (chatRequest, bool) {
	out := chatRequest{
		TopK:      req.TopK,
		GraphMix:  req.GraphMix,
		MMRLambda: req.MMRLambda,
		EntityMix: req.EntityMix,
		MinSim:    req.MinSim,
		Rerank:    req.Rerank,
		Model:     req.Model,
	}
	lastUser := -1
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			lastUser = i
			break
		}
	}
	if lastUser < 0 {
		return out, false
	}
	out.Query = req.Messages[lastUser].Content
	for _, m := range req.Messages[:lastUser] {
		if m.Role == "user" || m.Role == "assistant" {
			out.History = append(out.History, chatTurn(m))
		}
	}
	return out, out.Query != ""
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var oreq oaiChatRequest
	if err := json.NewDecoder(r.Body).Decode(&oreq); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	req, ok := oreq.toChatRequest()
	if !ok {
		writeErr(w, http.StatusBadRequest, errEmpty("a user message"))
		return
	}
	if req.TopK <= 0 {
		req.TopK = 6
	}
	model := req.Model
	if model == "" {
		model = s.genModel
	}
	if s.gen == nil || model == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("no language model configured"))
		return
	}

	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	res, abstain, err := s.retrieveForChat(r.Context(), st, req, model)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}

	created := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-%d", created)
	stop := "stop"

	if abstain {
		const msg = "I could not find anything relevant in this corpus to answer that."
		if oreq.Stream {
			s.streamCompletion(w, id, model, created, func(emit func(string)) error {
				emit(msg)
				return nil
			})
			return
		}
		writeJSON(w, http.StatusOK, oaiResponse{
			ID: id, Object: "chat.completion", Created: created, Model: model,
			Choices: []oaiChoice{{Index: 0, Message: &oaiMessage{Role: "assistant", Content: msg}, FinishReason: &stop}},
		})
		return
	}

	prompt := buildChatPrompt(req.Query, res, nil)
	if oreq.Stream {
		s.streamCompletion(w, id, model, created, func(emit func(string)) error {
			return s.gen.GenerateStream(r.Context(), model, chatSystemPrompt, prompt, func(tok string) error {
				emit(tok)
				return nil
			})
		})
		return
	}

	answer, err := s.gen.Generate(r.Context(), model, chatSystemPrompt, prompt)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, oaiResponse{
		ID: id, Object: "chat.completion", Created: created, Model: model,
		Choices: []oaiChoice{{Index: 0, Message: &oaiMessage{Role: "assistant", Content: answer}, FinishReason: &stop}},
		Usage:   &oaiUsage{},
	})
}

// streamCompletion writes an OpenAI-style server-sent event stream: a role
// preamble chunk, one chunk per token, a final stop chunk, and the [DONE]
// sentinel. produce drives token emission and reports any generation error.
func (s *Server) streamCompletion(w http.ResponseWriter, id, model string, created int64, produce func(emit func(string)) error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	chunk := func(c oaiChoice) {
		b, _ := json.Marshal(oaiResponse{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []oaiChoice{c},
		})
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	chunk(oaiChoice{Index: 0, Delta: &oaiMessage{Role: "assistant"}})
	genErr := produce(func(tok string) {
		chunk(oaiChoice{Index: 0, Delta: &oaiMessage{Content: tok}})
	})
	stop := "stop"
	if genErr != nil {
		stop = "error"
	}
	chunk(oaiChoice{Index: 0, Delta: &oaiMessage{}, FinishReason: &stop})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
