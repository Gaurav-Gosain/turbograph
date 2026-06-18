// Package mcp implements a minimal Model Context Protocol (MCP) server over the
// stdio transport using only the Go standard library.
//
// Transport overview:
//
// MCP messages are JSON-RPC 2.0 objects exchanged as newline-delimited JSON: one
// complete JSON object per line, read from an io.Reader (typically stdin) and
// written to an io.Writer (typically stdout). Requests carry an "id" and expect a
// matching response; notifications carry no "id" and receive no response.
//
// The handshake is:
//
//  1. Client sends "initialize" with protocolVersion, capabilities and clientInfo.
//     The server replies with its protocolVersion, capabilities and serverInfo.
//  2. Client sends the "notifications/initialized" notification (no id, no reply).
//  3. Client may then call "tools/list" and "tools/call".
//
// Tools are registered by the host. turbograph registers its search/answer tools
// against a Server and then calls Serve over stdio so an MCP client (such as an
// editor or agent runtime) can discover and invoke them.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
)

// protocolVersion is the MCP revision this server speaks. Pinned to the value the
// task targets; the server echoes it back so clients can detect a mismatch.
const protocolVersion = "2024-11-05"

// JSON-RPC 2.0 error codes used by this server. The first three are the standard
// reserved codes; the rest are spec-defined application ranges we reuse.
const (
	codeParseError     = -32700 // invalid JSON was received
	codeInvalidRequest = -32600 // the JSON is not a valid Request object
	codeMethodNotFound = -32601 // the method does not exist
	codeInvalidParams  = -32602 // invalid method parameters
	codeInternalError  = -32603 // internal JSON-RPC error
)

// scannerBufferSize is the maximum length of a single line the server will read.
// MCP tool arguments and results can be large (whole documents), so we allow well
// beyond bufio.Scanner's 64KB default.
const scannerBufferSize = 4 * 1024 * 1024

// ToolHandler runs a registered tool. args is the raw JSON value passed by the
// client under params.arguments; it may be null or absent. The returned string is
// surfaced to the client as text content. A non-nil error is reported to the
// client as a tool result with isError true (the MCP convention) rather than as a
// JSON-RPC protocol error, so the model can read and react to the failure.
type ToolHandler func(ctx context.Context, args json.RawMessage) (string, error)

// Tool is the metadata advertised by tools/list. InputSchema must be a JSON Schema
// object describing the accepted arguments; clients use it to validate calls.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// registeredTool bundles a Tool's advertised metadata with the handler that backs
// it so the two cannot drift apart in the registry.
type registeredTool struct {
	tool    Tool
	handler ToolHandler
}

// Server is a registry of tools plus the stdio dispatch loop. The zero value is
// not usable; construct one with NewServer. A Server is safe for concurrent use:
// the registry is guarded by mu, and Serve serializes all writes to the output.
type Server struct {
	name    string
	version string

	mu    sync.RWMutex
	tools map[string]registeredTool
}

// NewServer returns a Server that will identify itself to clients with the given
// name and version in the initialize response's serverInfo.
func NewServer(name, version string) *Server {
	return &Server{
		name:    name,
		version: version,
		tools:   make(map[string]registeredTool),
	}
}

// Register adds (or replaces) a tool. It is safe to call before Serve or
// concurrently with a running Serve. inputSchema should be a JSON Schema object;
// nil is allowed and advertised as an empty schema so clients still see the tool.
func (s *Server) Register(name, description string, inputSchema json.RawMessage, h ToolHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[name] = registeredTool{
		tool: Tool{
			Name:        name,
			Description: description,
			InputSchema: inputSchema,
		},
		handler: h,
	}
}

// request is a parsed inbound JSON-RPC message. ID is kept as RawMessage because
// JSON-RPC permits string, number, or null ids and we must echo whatever we got
// back verbatim. A nil/absent ID marks a notification, which gets no response.
type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// isNotification reports whether the message must not be answered. Per JSON-RPC, a
// message without an id is a notification.
func (r *request) isNotification() bool {
	return len(r.ID) == 0
}

// response is an outbound JSON-RPC reply. Exactly one of Result or Error is set.
// ID mirrors the request id (or null for errors raised before an id was parsed).
type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// rpcError is the JSON-RPC error object.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Serve reads newline-delimited JSON-RPC messages from r and writes responses to
// w until r reaches EOF or ctx is cancelled. It returns nil on a clean EOF and a
// non-nil error if the scanner fails or ctx is done.
//
// Each message is dispatched on the calling goroutine in arrival order, so output
// preserves request order. Writes are guarded by an internal mutex so that if a
// handler spawns concurrent work that also writes, lines are never interleaved.
func (s *Server) Serve(ctx context.Context, r io.Reader, w io.Writer) error {
	scanner := bufio.NewScanner(r)
	// Grow the scanner's buffer so large single-line messages are not truncated
	// into a bufio.ErrTooLong, which would otherwise kill the loop.
	scanner.Buffer(make([]byte, 0, 64*1024), scannerBufferSize)

	bw := bufio.NewWriter(w)
	var writeMu sync.Mutex

	// writeResponse marshals and emits a single response line, then flushes. It is
	// the only path that touches the writer, so the mutex here fully serializes
	// output across goroutines.
	writeResponse := func(resp *response) error {
		data, err := json.Marshal(resp)
		if err != nil {
			// Marshalling our own response should never fail; if it does, fall back
			// to a generic internal error so the client still gets a valid line.
			data = internalErrorLine(resp.ID)
		}
		writeMu.Lock()
		defer writeMu.Unlock()
		if _, err := bw.Write(data); err != nil {
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			return err
		}
		return bw.Flush()
	}

	for scanner.Scan() {
		// Honor cancellation between messages so a stuck client cannot pin us open
		// indefinitely once the host wants to shut down.
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Bytes()
		// Skip blank lines: some clients emit them as keep-alives and they are not
		// valid JSON-RPC messages.
		if len(trimSpace(line)) == 0 {
			continue
		}

		// Copy the line: Scanner reuses its buffer on the next Scan, and we hold
		// references (Params, ID) past this iteration.
		buf := make([]byte, len(line))
		copy(buf, line)

		var req request
		if err := json.Unmarshal(buf, &req); err != nil {
			// A parse error has no associated id (we could not read one), so the id
			// in the error response is null per JSON-RPC.
			if werr := writeResponse(errorResponse(nil, codeParseError, "parse error")); werr != nil {
				return werr
			}
			continue
		}

		resp := s.dispatch(ctx, &req)
		if resp == nil {
			// Notification: no reply.
			continue
		}
		if err := writeResponse(resp); err != nil {
			return err
		}
	}

	return scanner.Err()
}

// dispatch routes a single parsed request to its handler and returns the response
// to send, or nil if the message is a notification that must not be answered.
func (s *Server) dispatch(ctx context.Context, req *request) *response {
	switch req.Method {
	case "initialize":
		return s.handleInitialize(req)
	case "notifications/initialized":
		// Spec notification acknowledging the handshake; never answered.
		return nil
	case "ping":
		// Health check: reply with an empty result object.
		return resultResponse(req.ID, json.RawMessage(`{}`))
	case "tools/list":
		return s.handleToolsList(req)
	case "tools/call":
		return s.handleToolsCall(ctx, req)
	default:
		if req.isNotification() {
			// Unknown notifications are silently ignored, as JSON-RPC requires no
			// reply to any notification.
			return nil
		}
		return errorResponse(req.ID, codeMethodNotFound, fmt.Sprintf("method not found: %s", req.Method))
	}
}

// initializeResult is the payload of the initialize response.
type initializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      serverInfo     `json:"serverInfo"`
}

type serverInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

func (s *Server) handleInitialize(req *request) *response {
	result := initializeResult{
		ProtocolVersion: protocolVersion,
		// Advertise tools support. The empty object means "supported, no
		// sub-capabilities", matching the spec shape {"tools":{}}.
		Capabilities: map[string]any{
			"tools": map[string]any{},
		},
		ServerInfo: serverInfo{Name: s.name, Version: s.version},
	}
	return marshalResult(req.ID, result)
}

// toolsListResult is the payload of tools/list.
type toolsListResult struct {
	Tools []toolDescriptor `json:"tools"`
}

// toolDescriptor is the wire shape of an advertised tool. inputSchema is always
// present (defaulting to an empty object schema) because clients expect it.
type toolDescriptor struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

func (s *Server) handleToolsList(req *request) *response {
	s.mu.RLock()
	descriptors := make([]toolDescriptor, 0, len(s.tools))
	for _, rt := range s.tools {
		schema := rt.tool.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		descriptors = append(descriptors, toolDescriptor{
			Name:        rt.tool.Name,
			Description: rt.tool.Description,
			InputSchema: schema,
		})
	}
	s.mu.RUnlock()

	// Sort by name so tools/list output is deterministic across calls; map
	// iteration order is otherwise random and would make clients and tests flaky.
	sortDescriptors(descriptors)

	return marshalResult(req.ID, toolsListResult{Tools: descriptors})
}

// toolCallParams is the params shape of tools/call.
type toolCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// contentBlock is one item of a tool result's content array. Only text blocks are
// produced here.
type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// toolCallResult is the payload of a tools/call response.
type toolCallResult struct {
	Content []contentBlock `json:"content"`
	IsError bool           `json:"isError"`
}

func (s *Server) handleToolsCall(ctx context.Context, req *request) *response {
	var params toolCallParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return errorResponse(req.ID, codeInvalidParams, "invalid tools/call params")
	}

	s.mu.RLock()
	rt, ok := s.tools[params.Name]
	s.mu.RUnlock()
	if !ok {
		// Unknown tool is reported as a JSON-RPC error with code -32602
		// (invalid params): the requested tool name is not a valid parameter
		// value for this server. (The spec also permits -32601; we standardize on
		// -32602 because the method tools/call itself exists.)
		return errorResponse(req.ID, codeInvalidParams, fmt.Sprintf("unknown tool: %s", params.Name))
	}

	text, err := rt.handler(ctx, params.Arguments)
	if err != nil {
		// Handler failures are surfaced as tool results with isError true rather
		// than protocol errors, so the model receives the message and can adapt.
		return marshalResult(req.ID, toolCallResult{
			Content: []contentBlock{{Type: "text", Text: err.Error()}},
			IsError: true,
		})
	}

	return marshalResult(req.ID, toolCallResult{
		Content: []contentBlock{{Type: "text", Text: text}},
		IsError: false,
	})
}

// marshalResult builds a success response whose result is the JSON encoding of v.
// If encoding fails (which should not happen for our own types), it degrades to an
// internal error response so the client still receives a valid reply.
func marshalResult(id json.RawMessage, v any) *response {
	data, err := json.Marshal(v)
	if err != nil {
		return errorResponse(id, codeInternalError, "internal error")
	}
	return resultResponse(id, data)
}

// resultResponse wraps an already-encoded result payload.
func resultResponse(id json.RawMessage, result json.RawMessage) *response {
	return &response{JSONRPC: "2.0", ID: normalizeID(id), Result: result}
}

// errorResponse builds an error response.
func errorResponse(id json.RawMessage, code int, message string) *response {
	return &response{
		JSONRPC: "2.0",
		ID:      normalizeID(id),
		Error:   &rpcError{Code: code, Message: message},
	}
}

// normalizeID ensures the response id field is valid JSON. JSON-RPC requires the
// id to be present in responses; when we have none (e.g. a parse error) it must be
// the JSON literal null rather than an empty/invalid value.
func normalizeID(id json.RawMessage) json.RawMessage {
	if len(id) == 0 {
		return json.RawMessage("null")
	}
	return id
}

// internalErrorLine is a last-resort, hand-built error line used only if marshalling
// a response somehow fails. It avoids json.Marshal so it cannot itself fail.
func internalErrorLine(id json.RawMessage) []byte {
	if len(id) == 0 {
		id = json.RawMessage("null")
	}
	return []byte(fmt.Sprintf(
		`{"jsonrpc":"2.0","id":%s,"error":{"code":%d,"message":"internal error"}}`,
		id, codeInternalError,
	))
}

// trimSpace strips ASCII whitespace from both ends without allocating, used to
// detect blank/keep-alive lines.
func trimSpace(b []byte) []byte {
	start := 0
	for start < len(b) && isSpace(b[start]) {
		start++
	}
	end := len(b)
	for end > start && isSpace(b[end-1]) {
		end--
	}
	return b[start:end]
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

// sortDescriptors orders tools by name with a simple insertion sort. The tool
// count is small, so this avoids pulling in the sort package for clarity.
func sortDescriptors(d []toolDescriptor) {
	for i := 1; i < len(d); i++ {
		for j := i; j > 0 && d[j-1].Name > d[j].Name; j-- {
			d[j-1], d[j] = d[j], d[j-1]
		}
	}
}
