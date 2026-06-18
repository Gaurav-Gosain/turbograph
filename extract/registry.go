package extract

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// Registry maps a file extension to the Extractor that handles it. It is the
// dispatch layer between the ingestion pipeline and the concrete extractors, so
// callers route a file with one call instead of switching on extension themselves.
//
// A Registry is not safe for concurrent Register and Extract calls. Build it fully
// during setup, then treat it as read-only during ingestion.
type Registry struct {
	// byExt is keyed by lowercased extension without a leading dot.
	byExt map[string]Extractor
}

// textExtensions are formats whose bytes are already plain text.
var textExtensions = []string{"txt", "md", "markdown", "text"}

// NewRegistry returns a Registry with TextExtractor pre-registered for common
// plain-text extensions.
func NewRegistry() *Registry {
	r := &Registry{byExt: make(map[string]Extractor)}
	for _, ext := range textExtensions {
		r.Register(ext, TextExtractor{})
	}
	return r
}

// normalizeExt lowercases an extension and strips any leading dot, so callers may
// pass "PDF", ".pdf", or "pdf" interchangeably.
func normalizeExt(ext string) string {
	return strings.ToLower(strings.TrimPrefix(ext, "."))
}

// Register associates an extension with an Extractor, replacing any prior entry.
func (r *Registry) Register(ext string, e Extractor) {
	r.byExt[normalizeExt(ext)] = e
}

// Has reports whether an extractor is registered for the extension.
func (r *Registry) Has(ext string) bool {
	_, ok := r.byExt[normalizeExt(ext)]
	return ok
}

// Extensions returns the registered extensions, sorted, for diagnostics and UIs.
func (r *Registry) Extensions() []string {
	exts := make([]string, 0, len(r.byExt))
	for ext := range r.byExt {
		exts = append(exts, ext)
	}
	sort.Strings(exts)
	return exts
}

// Extract routes filename to its registered Extractor based on the file extension
// and returns the extracted text. An unregistered extension yields an error that
// lists what is registered, so the caller can tell what tooling is missing.
func (r *Registry) Extract(ctx context.Context, filename string, data []byte) (string, error) {
	ext := normalizeExt(filepath.Ext(filename))
	e, ok := r.byExt[ext]
	if !ok {
		if ext == "" {
			return "", fmt.Errorf("extract: no extension on %q; registered: %s",
				filename, strings.Join(r.Extensions(), ", "))
		}
		return "", fmt.Errorf("extract: no extractor for extension %q; registered: %s",
			ext, strings.Join(r.Extensions(), ", "))
	}
	return e.Extract(ctx, filename, data)
}

// DefaultRegistry returns a Registry wired with sensible defaults discovered from
// the host. Text extractors are always present. If pdftotext is on PATH, "pdf" is
// registered to use it; otherwise pdf is left unregistered so callers can detect
// the gap via Has("pdf") rather than failing at construction time.
func DefaultRegistry() *Registry {
	r := NewRegistry()
	if _, err := exec.LookPath("pdftotext"); err == nil {
		r.Register("pdf", PDFViaPdftotext())
	}
	return r
}
