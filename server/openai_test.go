package server

import (
	"bytes"
	"net/http"
	"strings"
	"testing"
)

func TestToChatRequestMapping(t *testing.T) {
	oreq := oaiChatRequest{
		Model: "m",
		Messages: []oaiMessage{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "first question"},
			{Role: "assistant", Content: "first answer"},
			{Role: "user", Content: "follow up"},
		},
		Rerank: true,
		TopK:   4,
	}
	req, ok := oreq.toChatRequest()
	if !ok {
		t.Fatal("expected a usable request")
	}
	if req.Query != "follow up" {
		t.Fatalf("query = %q, want last user message", req.Query)
	}
	// History is the prior user/assistant turns, excluding the system message and
	// the final user message that became the query.
	if len(req.History) != 2 {
		t.Fatalf("history len = %d, want 2", len(req.History))
	}
	if req.History[0].Content != "first question" || req.History[1].Content != "first answer" {
		t.Fatalf("unexpected history: %+v", req.History)
	}
	if !req.Rerank || req.TopK != 4 || req.Model != "m" {
		t.Fatalf("knobs not carried through: %+v", req)
	}
}

func TestToChatRequestNoUser(t *testing.T) {
	oreq := oaiChatRequest{Messages: []oaiMessage{{Role: "system", Content: "x"}}}
	if _, ok := oreq.toChatRequest(); ok {
		t.Fatal("expected mapping to fail without a user message")
	}
}

// TestChatAbstains drives the abstention gate: an impossibly high grounding floor
// makes every result insufficient, so the stream emits an "abstain" event and
// never reaches the (absent) model.
func TestChatAbstains(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{"query":"graphs","top_k":3,"min_sim":2.0}`
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	out := buf.String()
	if !strings.Contains(out, "event: abstain") {
		t.Fatalf("expected an abstain event, got:\n%s", out)
	}
	if strings.Contains(out, "event: error") {
		t.Fatalf("did not expect an error event:\n%s", out)
	}
}

// TestChatCompletionsNeedsModel confirms the OpenAI endpoint rejects a request
// when no generation model is configured, rather than answering ungrounded.
func TestChatCompletionsNeedsModel(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"graphs"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 without a model", resp.StatusCode)
	}
}
