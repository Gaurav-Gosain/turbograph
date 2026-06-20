package server

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// VisionCaptioner is the optional vision surface of a Backend: it describes an
// image so the description can be embedded and indexed like any other text. The
// Ollama backend implements it; image ingestion is offered only when the active
// backend does. This is the "describe then embed" path that makes figures and
// tables retrievable in the same vector space as text, with no second index.
type VisionCaptioner interface {
	CaptionImage(ctx context.Context, model, prompt string, image []byte) (string, error)
}

// assetStore keeps the raw bytes of ingested images on disk, content-addressed by
// a hash of their bytes, so an image chunk can reference its source for display
// without bloating the .tg snapshot (which stays text and vectors only).
type assetStore struct{ dir string }

var assetID = regexp.MustCompile(`^[a-f0-9]{16}\.[a-z0-9]{2,5}$`)

func newAssetStore(dir string) (*assetStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &assetStore{dir: dir}, nil
}

// Put stores image bytes and returns their asset id ("<hash>.<ext>"). Identical
// bytes map to the same id, so re-ingesting an image is free.
func (a *assetStore) Put(data []byte, ext string) (string, error) {
	ext = strings.ToLower(strings.TrimPrefix(ext, "."))
	if ext == "" || len(ext) > 5 {
		ext = "png"
	}
	sum := sha256.Sum256(data)
	id := hex.EncodeToString(sum[:8]) + "." + ext
	if !assetID.MatchString(id) {
		return "", fmt.Errorf("invalid asset extension %q", ext)
	}
	return id, os.WriteFile(filepath.Join(a.dir, id), data, 0o644)
}

// Open returns the bytes and content type of an asset by id, rejecting any id
// that is not a bare content-addressed name (no path traversal).
func (a *assetStore) Open(id string) ([]byte, string, error) {
	if !assetID.MatchString(id) {
		return nil, "", fmt.Errorf("invalid asset id")
	}
	data, err := os.ReadFile(filepath.Join(a.dir, id))
	if err != nil {
		return nil, "", err
	}
	ct := mime.TypeByExtension(filepath.Ext(id))
	if ct == "" {
		ct = "application/octet-stream"
	}
	return data, ct, nil
}

// EnableAssets gives the server an on-disk asset directory, turning on image
// ingestion and serving. Without it the image endpoints report unconfigured.
func (s *Server) EnableAssets(dir string) error {
	store, err := newAssetStore(dir)
	if err != nil {
		return err
	}
	s.assets = store
	return nil
}

// handleAsset serves a stored image by id (GET /api/asset/<id>).
func (s *Server) handleAsset(w http.ResponseWriter, r *http.Request) {
	if s.assets == nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("assets are not configured"))
		return
	}
	data, ct, err := s.assets.Open(r.PathValue("id"))
	if err != nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("asset not found"))
		return
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	w.Write(data)
}

type ingestImageRequest struct {
	ID     string         `json:"id"`     // document id for the image
	B64    string         `json:"b64"`    // the image bytes, base64
	Ext    string         `json:"ext"`    // file extension (png, jpg, ...)
	Model  string         `json:"model"`  // vision model to caption with
	Prompt string         `json:"prompt"` // optional captioning instruction
	Meta   map[string]any `json:"meta"`   // optional document metadata
}

// handleIngestImage stores an image, captions it with a vision model, and indexes
// the caption as an image chunk that references the stored asset. The image then
// retrieves by its description in the same hybrid search as text.
func (s *Server) handleIngestImage(w http.ResponseWriter, r *http.Request) {
	if s.assets == nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("image ingestion is not configured on this server"))
		return
	}
	vc, ok := s.gen.(VisionCaptioner)
	if !ok {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("the configured backend cannot caption images"))
		return
	}
	st, err := s.store(r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var req ingestImageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.ID == "" || req.B64 == "" {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("id and b64 are required"))
		return
	}
	model := req.Model
	if model == "" {
		model = s.genModel
	}
	if model == "" {
		writeErr(w, http.StatusBadRequest, errEmpty("model"))
		return
	}
	data, err := base64.StdEncoding.DecodeString(req.B64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid base64"))
		return
	}
	ref, err := s.assets.Put(data, req.Ext)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	caption, err := vc.CaptionImage(r.Context(), model, req.Prompt, data)
	if err != nil {
		writeErr(w, http.StatusBadGateway, fmt.Errorf("caption: %w", err))
		return
	}
	caption = strings.TrimSpace(caption)
	if caption == "" {
		writeErr(w, http.StatusBadGateway, fmt.Errorf("the model returned an empty caption"))
		return
	}
	doc := rag.Document{ID: req.ID, Text: caption, Meta: req.Meta, Kind: "image", ImageRef: ref}
	if err := st.AddDocuments(r.Context(), []rag.Document{doc}); err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	s.persist(bucketOf(r))
	writeJSON(w, http.StatusOK, map[string]any{"id": req.ID, "image_ref": ref, "caption": caption})
}
