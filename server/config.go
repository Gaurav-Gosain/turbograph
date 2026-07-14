package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// RuntimeConfig is the subset of settings the web UI can view and edit and that
// is persisted to a JSON file, so the configuration survives restarts and can be
// changed without touching the command line. Secret fields are write-only over
// the API: send them to set, but they are never read back (booleans report
// whether one is set).
type RuntimeConfig struct {
	// Providers are named OpenAI-compatible endpoints the operator has added. GenAPI
	// and EmbedAPI may name one, so generation and embedding can sit on different
	// services without either being hard-coded.
	Providers []Provider `json:"providers,omitempty"`

	// Generation backend (hot-swappable).
	GenAPI   string `json:"gen_api"` // "ollama", "openai", or a provider name
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

// Provider is a named OpenAI-compatible endpoint. Everything that differs between
// one such service and another lives here, which is why arbitrary headers are part
// of it: being OpenAI-compatible on the request body says nothing about what a
// service wants in the request headers.
type Provider struct {
	Name    string            `json:"name"`              // unique; what GenAPI/EmbedAPI reference
	BaseURL string            `json:"base_url"`          // host root, no /v1
	APIKey  string            `json:"api_key,omitempty"` // write-only over the API
	Headers map[string]string `json:"headers,omitempty"` // sent on every request
	Note    string            `json:"note,omitempty"`    // operator's own label
}

// Endpoint is a resolved backend target: which wire protocol, where, and with what
// credentials and headers. The server resolves a provider name to one of these and
// hands it to a factory, so adding a provider never means touching the factories.
type Endpoint struct {
	API     string // "ollama" or "openai"
	BaseURL string
	APIKey  string
	Headers map[string]string
}

// Factories build backends from configuration. The cmd layer injects these so the
// server can rebuild the generation/embedding backends when the config changes,
// without the server package importing the concrete provider clients.
type Factories struct {
	Backend  func(Endpoint) Backend
	Embedder func(ep Endpoint, model string, dim int) rag.Embedder
}

// endpoint resolves a provider to the target to talk to, dereferencing any
// environment-variable references in its key and headers.
func (p Provider) endpoint() Endpoint {
	var h map[string]string
	if len(p.Headers) > 0 {
		h = make(map[string]string, len(p.Headers))
		for k, v := range p.Headers {
			h[k] = envRef(v)
		}
	}
	return Endpoint{API: "openai", BaseURL: p.BaseURL, APIKey: envRef(p.APIKey), Headers: h}
}

// provider returns the named provider, or false. The lookup is linear because the
// list is operator-sized (a handful), not user-sized.
func (c RuntimeConfig) provider(name string) (Provider, bool) {
	for _, p := range c.Providers {
		if p.Name == name {
			return p, true
		}
	}
	return Provider{}, false
}

// endpoint resolves a backend selector to the target to talk to. The selector is
// "ollama", the legacy inline "openai" (from the command-line flags), or the name
// of an added provider.
func (c RuntimeConfig) endpoint(api, inlineURL, inlineKey string) Endpoint {
	switch api {
	case "", "ollama":
		return Endpoint{API: "ollama", BaseURL: c.OllamaURL}
	case "openai":
		return Endpoint{API: "openai", BaseURL: inlineURL, APIKey: inlineKey}
	}
	if p, ok := c.provider(api); ok {
		return p.endpoint()
	}
	// A selector naming a provider that no longer exists: fall back to Ollama rather
	// than silently building a client that points nowhere.
	return Endpoint{API: "ollama", BaseURL: c.OllamaURL}
}

// envRef resolves a value that is entirely an environment-variable reference
// ("$GROQ_API_KEY" or "${GROQ_API_KEY}") to that variable, and returns anything
// else unchanged. It lets an operator point a provider at a secret without the
// secret being written to the config file. The match is deliberately all-or-nothing:
// a key that merely contains a dollar sign is a literal key, not a reference.
func envRef(v string) string {
	name, ok := strings.CutPrefix(v, "$")
	if !ok {
		return v
	}
	if inner, braced := strings.CutPrefix(name, "{"); braced {
		if inner, closed := strings.CutSuffix(inner, "}"); closed {
			name = inner
		} else {
			return v
		}
	}
	if name == "" || strings.ContainsAny(name, "${} \t") {
		return v
	}
	if got, set := os.LookupEnv(name); set {
		return got
	}
	return v
}

// GenEndpoint and EmbedEndpoint resolve the configured backends.
func (c RuntimeConfig) GenEndpoint() Endpoint {
	return c.endpoint(c.GenAPI, c.GenURL, c.GenKey)
}

func (c RuntimeConfig) EmbedEndpoint() Endpoint {
	return c.endpoint(c.EmbedAPI, c.EmbedURL, c.EmbedKey)
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
	// Providers go out without their keys: the client is told only whether one is
	// set, and sends a blank back to leave it alone.
	provs := make([]map[string]any, 0, len(c.Providers))
	for _, p := range c.Providers {
		provs = append(provs, map[string]any{
			"name": p.Name, "base_url": p.BaseURL, "headers": p.Headers,
			"note": p.Note, "key_set": p.APIKey != "",
		})
	}
	resp := map[string]any{
		"providers": provs,
		"gen_api":   c.GenAPI, "gen_url": c.GenURL, "gen_model": c.GenModel,
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
	for i, p := range in.Providers {
		if p.APIKey == "" {
			if old, ok := prev.provider(p.Name); ok {
				in.Providers[i].APIKey = old.APIKey
			}
		}
	}
	if err := validateConfig(in); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}

	// Hot-swap the generation backend.
	if s.factories.Backend != nil {
		s.gen = s.factories.Backend(in.GenEndpoint())
		s.genModel = in.GenModel
		s.embedModel = in.EmbedModel
	}
	// Apply chunking (and the embedder for new buckets) to the manager.
	mgrCfg := s.mgr.Config()
	mgrCfg.Chunk = rag.ChunkConfig{Strategy: in.ChunkStrategy, TargetWords: in.ChunkWords, OverlapWords: in.ChunkOverlap}
	s.mgr.SetConfig(mgrCfg)
	if s.factories.Embedder != nil {
		s.mgr.SetEmbedder(s.factories.Embedder(in.EmbedEndpoint(), in.EmbedModel, in.EmbedDim))
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

// handleTestProvider probes a provider without saving it: it builds a throwaway
// client from the posted definition and lists the models the endpoint advertises,
// so the UI can tell the operator whether the URL, key, and headers actually work
// before the configuration is committed. A blank key means "use the saved one",
// matching the config endpoints. It is gated on config editing being enabled,
// because it makes the server issue an outbound request the caller controls.
func (s *Server) handleTestProvider(w http.ResponseWriter, r *http.Request) {
	if s.cfgPath == "" {
		writeErr(w, http.StatusForbidden, fmt.Errorf("configuration editing is disabled on this server"))
		return
	}
	if s.factories.Backend == nil {
		writeErr(w, http.StatusForbidden, fmt.Errorf("this server cannot swap backends"))
		return
	}
	var p Provider
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if p.BaseURL == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("a base URL is required"))
		return
	}
	if err := validateBackendURL(p.BaseURL); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if err := validateHeaders(p.Headers); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if p.APIKey == "" {
		if old, ok := s.cfg.provider(p.Name); ok {
			p.APIKey = old.APIKey
		}
	}
	back := s.factories.Backend(p.endpoint())

	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()
	models, err := back.ListModels(ctx)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "models": models})
}

func validateConfig(c RuntimeConfig) error {
	seen := make(map[string]bool, len(c.Providers))
	for _, p := range c.Providers {
		switch {
		case strings.TrimSpace(p.Name) == "":
			return fmt.Errorf("every provider needs a name")
		case p.Name == "ollama" || p.Name == "openai":
			return fmt.Errorf("provider name %q is reserved", p.Name)
		case seen[p.Name]:
			return fmt.Errorf("duplicate provider name %q", p.Name)
		case p.BaseURL == "":
			return fmt.Errorf("provider %q needs a base URL", p.Name)
		}
		seen[p.Name] = true
		if err := validateBackendURL(p.BaseURL); err != nil {
			return err
		}
		if err := validateHeaders(p.Headers); err != nil {
			return fmt.Errorf("provider %q: %w", p.Name, err)
		}
	}
	for _, api := range []string{c.GenAPI, c.EmbedAPI} {
		if api != "" && api != "ollama" && api != "openai" && !seen[api] {
			return fmt.Errorf("backend %q is neither ollama, openai, nor an added provider", api)
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

// validateHeaders rejects header names and values that are not well-formed. A
// value carrying a newline would let a configured header inject further headers
// into every outbound request, so it is refused rather than silently sanitized.
func validateHeaders(h map[string]string) error {
	for k, v := range h {
		if k == "" {
			return fmt.Errorf("a header name cannot be empty")
		}
		for i := 0; i < len(k); i++ {
			if !isTokenChar(k[i]) {
				return fmt.Errorf("%q is not a valid header name", k)
			}
		}
		for i := 0; i < len(v); i++ {
			// Visible ASCII, space, tab, and obs-text (>= 0x80) only: anything else is a
			// control character, and CR or LF would be header injection.
			if c := v[i]; c < 0x20 && c != '\t' || c == 0x7f {
				return fmt.Errorf("header %q has a value containing control characters", k)
			}
		}
	}
	return nil
}

// isTokenChar reports whether c may appear in an HTTP field name (RFC 9110 token).
func isTokenChar(c byte) bool {
	switch {
	case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		return true
	}
	return strings.IndexByte("!#$%&'*+-.^_`|~", c) >= 0
}
