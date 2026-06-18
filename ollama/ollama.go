// Package ollama is a minimal client for the local Ollama server, covering the
// two endpoints the RAG pipeline needs: batch embeddings and text generation.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// DefaultEmbedModel is a compact, high-quality embedding model.
const DefaultEmbedModel = "nomic-embed-text"

// Client talks to an Ollama server.
type Client struct {
	BaseURL    string
	EmbedModel string
	HTTP       *http.Client
}

// New returns a client honoring the OLLAMA_HOST environment variable, falling
// back to the local default.
func New() *Client {
	base := os.Getenv("OLLAMA_HOST")
	if base == "" {
		base = "http://127.0.0.1:11434"
	}
	return &Client{
		BaseURL:    base,
		EmbedModel: DefaultEmbedModel,
		HTTP:       &http.Client{Timeout: 5 * time.Minute},
	}
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed returns one embedding per input string in order. It uses the batch
// endpoint so a whole chunk set is embedded in a single request.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(embedRequest{Model: c.EmbedModel, Input: texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<12))
		return nil, fmt.Errorf("ollama embed: status %d: %s", resp.StatusCode, msg)
	}
	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("ollama embed decode: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embed: got %d embeddings for %d inputs", len(out.Embeddings), len(texts))
	}
	return out.Embeddings, nil
}

// EmbedOne embeds a single string.
func (c *Client) EmbedOne(ctx context.Context, text string) ([]float32, error) {
	v, err := c.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}

type generateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	System string `json:"system,omitempty"`
	Stream bool   `json:"stream"`
	Think  bool   `json:"think"`
}

type generateResponse struct {
	Response string `json:"response"`
	Thinking string `json:"thinking"`
	Done     bool   `json:"done"`
}

// Generate runs a non-streaming completion with the given model. Thinking is
// disabled so reasoning models answer directly rather than spending the output
// budget on a hidden chain of thought.
func (c *Client) Generate(ctx context.Context, model, system, prompt string) (string, error) {
	body, err := json.Marshal(generateRequest{Model: model, Prompt: prompt, System: system, Stream: false, Think: false})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama generate: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<12))
		return "", fmt.Errorf("ollama generate: status %d: %s", resp.StatusCode, msg)
	}
	var out generateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("ollama generate decode: %w", err)
	}
	// Fall back to the thinking channel if a model ignored think=false and left
	// the response field empty.
	if out.Response == "" && out.Thinking != "" {
		return out.Thinking, nil
	}
	return out.Response, nil
}

// GenerateStream runs a streaming completion, invoking onToken for each token as
// it arrives. Thinking is disabled so reasoning models answer directly. onToken
// returning an error stops the stream early.
func (c *Client) GenerateStream(ctx context.Context, model, system, prompt string, onToken func(string) error) error {
	body, err := json.Marshal(generateRequest{Model: model, Prompt: prompt, System: system, Stream: true, Think: false})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("ollama generate: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<12))
		return fmt.Errorf("ollama generate: status %d: %s", resp.StatusCode, msg)
	}
	dec := json.NewDecoder(resp.Body)
	for {
		var chunk generateResponse
		if err := dec.Decode(&chunk); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if chunk.Response != "" {
			if err := onToken(chunk.Response); err != nil {
				return err
			}
		}
		if chunk.Done {
			return nil
		}
	}
}

// ListModels returns the names of locally available models.
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama tags: status %d", resp.StatusCode)
	}
	var out struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	names := make([]string, len(out.Models))
	for i, m := range out.Models {
		names[i] = m.Name
	}
	return names, nil
}

// Ping reports whether the server is reachable.
func (c *Client) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/api/tags", nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}
