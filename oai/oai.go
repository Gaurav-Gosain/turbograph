// Package oai is a minimal client for any OpenAI-compatible HTTP API: the
// /v1/embeddings, /v1/chat/completions, and /v1/models endpoints. It lets
// turbograph use OpenAI, OpenRouter, Together, vLLM, LM Studio, llama.cpp, or any
// other server that speaks the OpenAI wire format, in place of Ollama. The method
// surface (Embed, EmbedQuery, Generate, GenerateStream, ListModels, Ping)
// deliberately mirrors ollama.Client so either satisfies the same interfaces.
package oai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

// Client talks to an OpenAI-compatible server.
type Client struct {
	BaseURL    string // e.g. https://api.openai.com or http://localhost:1234 (no trailing /v1)
	APIKey     string // sent as "Authorization: Bearer <key>"; may be empty for local servers
	EmbedModel string
	HTTP       *http.Client

	// QueryPrefix/DocPrefix and EmbedDim mirror ollama.Client for parity: most
	// hosted models do not need prefixes, but the fields exist so the same wiring
	// works for instruction-tuned embedders, and EmbedDim truncates (Matryoshka).
	QueryPrefix string
	DocPrefix   string
	EmbedDim    int
}

// New returns a client for baseURL with the given API key. baseURL should be the
// host root (the "/v1" path is appended by each call).
func New(baseURL, apiKey, embedModel string) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		EmbedModel: embedModel,
		HTTP:       &http.Client{Timeout: 5 * time.Minute},
	}
}

func (c *Client) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, r)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	return c.HTTP.Do(req)
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// Embed returns one embedding per input, encoded as documents (DocPrefix applied).
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return c.embed(ctx, texts, c.DocPrefix)
}

// EmbedQuery embeds search queries (QueryPrefix applied).
func (c *Client) EmbedQuery(ctx context.Context, texts []string) ([][]float32, error) {
	return c.embed(ctx, texts, c.QueryPrefix)
}

func (c *Client) embed(ctx context.Context, texts []string, prefix string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	input := texts
	if prefix != "" {
		input = make([]string, len(texts))
		for i, t := range texts {
			input[i] = prefix + t
		}
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/embeddings", embedRequest{Model: c.EmbedModel, Input: input})
	if err != nil {
		return nil, fmt.Errorf("oai embed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, statusError("embed", resp)
	}
	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("oai embed decode: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("oai embed: got %d embeddings for %d inputs", len(out.Data), len(texts))
	}
	// The API may return data out of order; place each by its index.
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("oai embed: index %d out of range", d.Index)
		}
		vecs[d.Index] = truncateNormalize(d.Embedding, c.EmbedDim)
	}
	return vecs, nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

func messages(system, prompt string) []chatMessage {
	m := make([]chatMessage, 0, 2)
	if system != "" {
		m = append(m, chatMessage{Role: "system", Content: system})
	}
	return append(m, chatMessage{Role: "user", Content: prompt})
}

// Generate runs a non-streaming chat completion.
func (c *Client) Generate(ctx context.Context, model, system, prompt string) (string, error) {
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", chatRequest{
		Model: model, Messages: messages(system, prompt),
	})
	if err != nil {
		return "", fmt.Errorf("oai generate: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", statusError("generate", resp)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("oai generate decode: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", nil
	}
	return out.Choices[0].Message.Content, nil
}

// GenerateStream runs a streaming chat completion, invoking onToken per delta.
func (c *Client) GenerateStream(ctx context.Context, model, system, prompt string, onToken func(string) error) error {
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", chatRequest{
		Model: model, Messages: messages(system, prompt), Stream: true,
	})
	if err != nil {
		return fmt.Errorf("oai generate: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return statusError("generate", resp)
	}
	// Server-sent events: lines of "data: {json}" terminated by "data: [DONE]".
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return nil
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				if err := onToken(ch.Delta.Content); err != nil {
					return err
				}
			}
		}
	}
	return sc.Err()
}

// ListModels returns the ids advertised by /v1/models.
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	resp, err := c.do(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, statusError("models", resp)
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	names := make([]string, len(out.Data))
	for i, m := range out.Data {
		names[i] = m.ID
	}
	return names, nil
}

// Ping checks the endpoint is reachable by listing models.
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.ListModels(ctx)
	return err
}

func statusError(op string, resp *http.Response) error {
	msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<12))
	return fmt.Errorf("oai %s: status %d: %s", op, resp.StatusCode, strings.TrimSpace(string(msg)))
}

// truncateNormalize keeps the first dim coordinates and rescales to unit length
// (Matryoshka), or returns v unchanged when dim is 0 or not smaller.
func truncateNormalize(v []float32, dim int) []float32 {
	if dim <= 0 || dim >= len(v) {
		return v
	}
	out := v[:dim]
	var n float64
	for _, x := range out {
		n += float64(x) * float64(x)
	}
	if n > 0 {
		inv := float32(1 / math.Sqrt(n))
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}
