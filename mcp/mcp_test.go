package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
)

// drive runs the server over a single input string and returns the parsed
// response lines (skipping blanks). It uses a finite reader so Serve hits EOF and
// returns, which keeps tests from blocking.
func drive(t *testing.T, s *Server, input string) []response {
	t.Helper()
	var out bytes.Buffer
	if err := s.Serve(context.Background(), strings.NewReader(input), &out); err != nil {
		t.Fatalf("Serve returned error: %v", err)
	}
	return parseResponses(t, out.String())
}

func parseResponses(t *testing.T, raw string) []response {
	t.Helper()
	var resps []response
	for _, line := range strings.Split(raw, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r response
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("response line is not valid JSON: %q: %v", line, err)
		}
		if r.JSONRPC != "2.0" {
			t.Fatalf("response missing jsonrpc=2.0: %q", line)
		}
		resps = append(resps, r)
	}
	return resps
}

func newTestServer() *Server {
	s := NewServer("turbograph", "1.2.3")
	s.Register("echo", "echoes its input back",
		json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
		func(_ context.Context, args json.RawMessage) (string, error) {
			var p struct {
				Text string `json:"text"`
			}
			_ = json.Unmarshal(args, &p)
			return "echo: " + p.Text, nil
		})
	s.Register("boom", "always fails", nil,
		func(_ context.Context, _ json.RawMessage) (string, error) {
			return "", errors.New("kaboom")
		})
	return s
}

func TestInitialize(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"0"}}}`+"\n")

	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	var got initializeResult
	if err := json.Unmarshal(resps[0].Result, &got); err != nil {
		t.Fatalf("unmarshal initialize result: %v", err)
	}
	if got.ProtocolVersion != "2024-11-05" {
		t.Errorf("protocolVersion = %q, want 2024-11-05", got.ProtocolVersion)
	}
	if _, ok := got.Capabilities["tools"]; !ok {
		t.Errorf("capabilities missing tools key: %v", got.Capabilities)
	}
	if got.ServerInfo.Name != "turbograph" || got.ServerInfo.Version != "1.2.3" {
		t.Errorf("serverInfo = %+v, want {turbograph 1.2.3}", got.ServerInfo)
	}
	if string(resps[0].ID) != "1" {
		t.Errorf("id = %s, want 1", resps[0].ID)
	}
}

func TestToolsList(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`+"\n")
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	var got toolsListResult
	if err := json.Unmarshal(resps[0].Result, &got); err != nil {
		t.Fatalf("unmarshal tools/list: %v", err)
	}
	if len(got.Tools) != 2 {
		t.Fatalf("want 2 tools, got %d", len(got.Tools))
	}
	// Deterministic order: boom before echo.
	if got.Tools[0].Name != "boom" || got.Tools[1].Name != "echo" {
		t.Errorf("tool order = [%s %s], want [boom echo]", got.Tools[0].Name, got.Tools[1].Name)
	}
	// echo carries the schema it was registered with.
	var schema map[string]any
	if err := json.Unmarshal(got.Tools[1].InputSchema, &schema); err != nil {
		t.Fatalf("echo schema not valid JSON: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("echo schema type = %v, want object", schema["type"])
	}
	// boom was registered with a nil schema and should default to an object schema.
	if len(got.Tools[0].InputSchema) == 0 {
		t.Errorf("boom should have a default inputSchema, got empty")
	}
	if got.Tools[0].Description != "always fails" {
		t.Errorf("boom description = %q", got.Tools[0].Description)
	}
}

func TestToolsCallSuccess(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`+"\n")
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	if resps[0].Error != nil {
		t.Fatalf("unexpected error: %+v", resps[0].Error)
	}
	var got toolCallResult
	if err := json.Unmarshal(resps[0].Result, &got); err != nil {
		t.Fatalf("unmarshal tool result: %v", err)
	}
	if got.IsError {
		t.Errorf("isError = true, want false")
	}
	if len(got.Content) != 1 || got.Content[0].Type != "text" {
		t.Fatalf("content = %+v, want one text block", got.Content)
	}
	if got.Content[0].Text != "echo: hi" {
		t.Errorf("text = %q, want %q", got.Content[0].Text, "echo: hi")
	}
}

func TestToolsCallHandlerError(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"boom","arguments":{}}}`+"\n")
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	// A handler error must be a result with isError, not a JSON-RPC error.
	if resps[0].Error != nil {
		t.Fatalf("handler error should not be a protocol error: %+v", resps[0].Error)
	}
	var got toolCallResult
	if err := json.Unmarshal(resps[0].Result, &got); err != nil {
		t.Fatalf("unmarshal tool result: %v", err)
	}
	if !got.IsError {
		t.Errorf("isError = false, want true")
	}
	if len(got.Content) != 1 || got.Content[0].Text != "kaboom" {
		t.Errorf("content = %+v, want text kaboom", got.Content)
	}
}

func TestToolsCallUnknownTool(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nope","arguments":{}}}`+"\n")
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	if resps[0].Error == nil {
		t.Fatalf("want JSON-RPC error for unknown tool, got result %s", resps[0].Result)
	}
	// Documented choice: unknown tool -> -32602 (invalid params).
	if resps[0].Error.Code != codeInvalidParams {
		t.Errorf("error code = %d, want %d", resps[0].Error.Code, codeInvalidParams)
	}
}

func TestUnknownMethod(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":6,"method":"does/not/exist"}`+"\n")
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	if resps[0].Error == nil || resps[0].Error.Code != codeMethodNotFound {
		t.Fatalf("want method not found (-32601), got %+v", resps[0].Error)
	}
}

func TestNotificationProducesNoOutput(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","method":"notifications/initialized"}`+"\n")
	if len(resps) != 0 {
		t.Fatalf("notification produced %d responses, want 0", len(resps))
	}
}

func TestUnknownNotificationProducesNoOutput(t *testing.T) {
	s := newTestServer()
	// A method-less id means notification; an unknown notification is ignored.
	resps := drive(t, s, `{"jsonrpc":"2.0","method":"notifications/whatever"}`+"\n")
	if len(resps) != 0 {
		t.Fatalf("unknown notification produced %d responses, want 0", len(resps))
	}
}

func TestPing(t *testing.T) {
	s := newTestServer()
	resps := drive(t, s, `{"jsonrpc":"2.0","id":7,"method":"ping"}`+"\n")
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
	if resps[0].Error != nil {
		t.Fatalf("ping error: %+v", resps[0].Error)
	}
	if strings.TrimSpace(string(resps[0].Result)) != "{}" {
		t.Errorf("ping result = %s, want {}", resps[0].Result)
	}
}

func TestMultipleRequestsInOrder(t *testing.T) {
	s := newTestServer()
	input := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"ping"}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`, // no reply
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo","arguments":{"text":"a"}}}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/list"}`,
	}, "\n") + "\n"

	resps := drive(t, s, input)
	if len(resps) != 3 {
		t.Fatalf("want 3 responses (notification skipped), got %d", len(resps))
	}
	// Responses arrive in request order; the notification produced nothing.
	if string(resps[0].ID) != "1" || string(resps[1].ID) != "2" || string(resps[2].ID) != "3" {
		t.Errorf("ids = [%s %s %s], want [1 2 3]", resps[0].ID, resps[1].ID, resps[2].ID)
	}
}

func TestMalformedJSONDoesNotKillLoop(t *testing.T) {
	s := newTestServer()
	input := strings.Join([]string{
		`{not valid json`,
		`{"jsonrpc":"2.0","id":9,"method":"ping"}`,
	}, "\n") + "\n"

	resps := drive(t, s, input)
	if len(resps) != 2 {
		t.Fatalf("want 2 responses, got %d", len(resps))
	}
	// First is a parse error with null id.
	if resps[0].Error == nil || resps[0].Error.Code != codeParseError {
		t.Fatalf("want parse error (-32700), got %+v", resps[0].Error)
	}
	if strings.TrimSpace(string(resps[0].ID)) != "null" {
		t.Errorf("parse error id = %s, want null", resps[0].ID)
	}
	// The loop kept going and answered the next valid request.
	if string(resps[1].ID) != "9" || resps[1].Error != nil {
		t.Errorf("second response = %+v, want ping reply id 9", resps[1])
	}
}

func TestBlankLinesIgnored(t *testing.T) {
	s := newTestServer()
	input := "\n   \n" + `{"jsonrpc":"2.0","id":1,"method":"ping"}` + "\n\n"
	resps := drive(t, s, input)
	if len(resps) != 1 {
		t.Fatalf("want 1 response, got %d", len(resps))
	}
}

func TestContextCancellationStops(t *testing.T) {
	s := newTestServer()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before serving

	// A reader that never EOFs on its own; cancellation must break the loop. We
	// feed one full line so Scan returns, then the loop's ctx check fires.
	r := strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"ping"}` + "\n")
	var out bytes.Buffer
	err := s.Serve(ctx, r, &out)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestConcurrentWritesDoNotInterleave(t *testing.T) {
	// Register a tool whose handler launches concurrent writers indirectly is hard
	// to do through Serve, so instead we exercise the write serialization by
	// driving many requests and confirming every output line is independently
	// valid JSON (no interleaving corruption).
	s := NewServer("t", "1")
	s.Register("noop", "", nil, func(_ context.Context, _ json.RawMessage) (string, error) {
		return "ok", nil
	})

	var sb strings.Builder
	const n = 200
	for i := 0; i < n; i++ {
		sb.WriteString(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"noop","arguments":{}}}`)
		sb.WriteByte('\n')
	}
	resps := drive(t, s, sb.String())
	if len(resps) != n {
		t.Fatalf("want %d responses, got %d", n, len(resps))
	}
}

// TestServeReturnsWriteError verifies that a failing writer surfaces as an error
// rather than a panic or silent drop.
func TestServeReturnsWriteError(t *testing.T) {
	s := newTestServer()
	err := s.Serve(context.Background(),
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"ping"}`+"\n"),
		failingWriter{})
	if err == nil {
		t.Fatal("want write error, got nil")
	}
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

// TestRegisterConcurrent confirms the registry is safe under concurrent
// registration while serving.
func TestRegisterConcurrent(t *testing.T) {
	s := NewServer("t", "1")
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s.Register("tool", "", nil, func(_ context.Context, _ json.RawMessage) (string, error) {
				return "", nil
			})
		}(i)
	}
	// Concurrently serve an empty stream.
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = s.Serve(context.Background(), strings.NewReader(""), io.Discard)
	}()
	wg.Wait()
}
