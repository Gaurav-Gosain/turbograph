package server

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// RuntimeConfig is the subset of settings the web UI can view and edit and that
// is persisted to a JSON file, so the configuration survives restarts and can be
// changed without touching the command line. Secret fields are write-only over
// the API: send them to set, but they are never read back (booleans report
// whether one is set).
type RuntimeConfig struct {
	// Generation backend (hot-swappable).
	GenAPI   string `json:"gen_api"` // "ollama" or "openai"
	GenURL   string `json:"gen_url"`
	GenKey   string `json:"gen_key,omitempty"`
	GenModel string `json:"gen_model"`

	// Embedding backend (applies to new buckets; existing buckets keep theirs).
	EmbedAPI   string `json:"embed_api"`
	EmbedURL   string `json:"embed_url"`
	EmbedKey   string `json:"embed_key,omitempty"`
	EmbedModel string `json:"embed_model"`
	EmbedDim   int    `json:"embed_dim"`

	OllamaURL string `json:"ollama_url"`

	// Chunking for new buckets.
	ChunkStrategy string `json:"chunk_strategy"`
	ChunkWords    int    `json:"chunk_words"`
	ChunkOverlap  int    `json:"chunk_overlap"`

	// Storage (restart to apply; keys come from the environment, never stored).
	S3Endpoint string `json:"s3_endpoint"`
	S3Bucket   string `json:"s3_bucket"`
	S3Region   string `json:"s3_region"`
	S3Prefix   string `json:"s3_prefix"`
}

// Factories build backends from configuration. The cmd layer injects these so the
// server can rebuild the generation/embedding backends when the config changes,
// without the server package importing the concrete provider clients.
type Factories struct {
	Backend  func(api, url, key string) Backend
	Embedder func(api, url, key, model string, dim int) rag.Embedder
}

// EnableConfig turns on the /api/config endpoints, persisting edits to path and
// applying hot-swappable changes (the generation backend, default chunking) at
// runtime through the provided factories.
func (s *Server) EnableConfig(cfg RuntimeConfig, path string, f Factories) {
	s.cfg = cfg
	s.cfgPath = path
	s.factories = f
}

// LoadConfig reads a persisted RuntimeConfig from path. A missing file is not an
// error (ok is false); the caller then seeds from flags.
func LoadConfig(path string) (RuntimeConfig, bool, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return RuntimeConfig{}, false, nil
	}
	if err != nil {
		return RuntimeConfig{}, false, err
	}
	var c RuntimeConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return RuntimeConfig{}, false, fmt.Errorf("config %s: %w", path, err)
	}
	return c, true, nil
}

func saveConfig(path string, c RuntimeConfig) error {
	if path == "" {
		return nil
	}
	if dir := filepath.Dir(path); dir != "" {
		os.MkdirAll(dir, 0o755)
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600)
}

// handleGetConfig returns the current configuration with secrets redacted.
func (s *Server) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	c := s.cfg
	resp := map[string]any{
		"gen_api": c.GenAPI, "gen_url": c.GenURL, "gen_model": c.GenModel,
		"gen_key_set": c.GenKey != "",
		"embed_api":   c.EmbedAPI, "embed_url": c.EmbedURL, "embed_model": c.EmbedModel, "embed_dim": c.EmbedDim,
		"embed_key_set":  c.EmbedKey != "",
		"ollama_url":     c.OllamaURL,
		"chunk_strategy": c.ChunkStrategy, "chunk_words": c.ChunkWords, "chunk_overlap": c.ChunkOverlap,
		"s3_endpoint": c.S3Endpoint, "s3_bucket": c.S3Bucket, "s3_region": c.S3Region, "s3_prefix": c.S3Prefix,
		"strategies": []string{rag.StrategyRecursive, rag.StrategyWord, rag.StrategyMarkdown, rag.StrategySentence},
		"editable":   s.cfgPath != "",
	}
	writeJSON(w, http.StatusOK, resp)
}

// handlePostConfig applies and persists a configuration update. The generation
// backend and default chunking apply immediately; embedding-backend and storage
// changes are saved and take effect on the next start (swapping them live would
// risk mixing incompatible vectors or losing in-flight buckets).
func (s *Server) handlePostConfig(w http.ResponseWriter, r *http.Request) {
	if s.cfgPath == "" {
		writeErr(w, http.StatusForbidden, fmt.Errorf("configuration editing is disabled on this server"))
		return
	}
	var in RuntimeConfig
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	prev := s.cfg
	// Keep existing secrets when the client sends a blank (it never receives them).
	if in.GenKey == "" {
		in.GenKey = prev.GenKey
	}
	if in.EmbedKey == "" {
		in.EmbedKey = prev.EmbedKey
	}
	if err := validateConfig(in); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}

	// Hot-swap the generation backend.
	if s.factories.Backend != nil {
		s.gen = s.factories.Backend(in.GenAPI, genBaseURL(in), in.GenKey)
		s.genModel = in.GenModel
		s.embedModel = in.EmbedModel
	}
	// Apply chunking (and the embedder for new buckets) to the manager.
	mgrCfg := s.mgr.Config()
	mgrCfg.Chunk = rag.ChunkConfig{Strategy: in.ChunkStrategy, TargetWords: in.ChunkWords, OverlapWords: in.ChunkOverlap}
	s.mgr.SetConfig(mgrCfg)
	if s.factories.Embedder != nil {
		s.mgr.SetEmbedder(s.factories.Embedder(in.EmbedAPI, embedBaseURL(in), in.EmbedKey, in.EmbedModel, in.EmbedDim))
	}

	restart := prev.S3Endpoint != in.S3Endpoint || prev.S3Bucket != in.S3Bucket ||
		prev.S3Region != in.S3Region || prev.S3Prefix != in.S3Prefix
	s.cfg = in
	if err := saveConfig(s.cfgPath, in); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"saved": true, "restart_required": restart})
}

func validateConfig(c RuntimeConfig) error {
	for _, api := range []string{c.GenAPI, c.EmbedAPI} {
		if api != "" && api != "ollama" && api != "openai" {
			return fmt.Errorf("backend api must be ollama or openai, got %q", api)
		}
	}
	if c.EmbedAPI == "openai" && c.EmbedURL == "" {
		return fmt.Errorf("openai embedding backend requires a base URL")
	}
	if c.GenAPI == "openai" && c.GenURL == "" {
		return fmt.Errorf("openai generation backend requires a base URL")
	}
	for _, u := range []string{c.GenURL, c.EmbedURL, c.OllamaURL} {
		if err := validateBackendURL(u); err != nil {
			return err
		}
	}
	switch c.ChunkStrategy {
	case "", rag.StrategyRecursive, rag.StrategyWord, rag.StrategyMarkdown, rag.StrategySentence:
	default:
		return fmt.Errorf("unknown chunk strategy %q", c.ChunkStrategy)
	}
	return nil
}

// validateBackendURL guards the configurable backend URLs against the obvious
// SSRF footguns: the config endpoint controls where the server makes outbound
// requests, so a configured URL must be a plain http(s) endpoint and must not
// target the cloud instance-metadata address, which is never a legitimate model
// backend and is the classic SSRF target. Loopback and private addresses are
// allowed on purpose: a local Ollama at 127.0.0.1 is the default. Protect the
// config endpoint with --api-key when the server is exposed.
func validateBackendURL(raw string) error {
	if raw == "" {
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid backend URL %q: %w", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("backend URL must be http or https, got %q", raw)
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("backend URL must include a host: %q", raw)
	}
	// Block the link-local instance-metadata endpoints (AWS, GCP, Azure all use
	// 169.254.169.254; its IPv6 form is fd00:ec2::254).
	if host == "169.254.169.254" || host == "fd00:ec2::254" || strings.EqualFold(host, "metadata.google.internal") {
		return fmt.Errorf("backend URL targets the instance metadata endpoint, which is not allowed")
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLinkLocalUnicast() {
		return fmt.Errorf("backend URL targets a link-local address, which is not allowed")
	}
	return nil
}

// genBaseURL / embedBaseURL pick the right base URL for the chosen api.
func genBaseURL(c RuntimeConfig) string {
	if c.GenAPI == "openai" {
		return c.GenURL
	}
	return c.OllamaURL
}

func embedBaseURL(c RuntimeConfig) string {
	if c.EmbedAPI == "openai" {
		return c.EmbedURL
	}
	return c.OllamaURL
}
