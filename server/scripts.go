package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/Gaurav-Gosain/turbograph/rag"
	"github.com/Gaurav-Gosain/turbograph/script"
)

// SetScripts attaches the operator's script registry, enabling the transform stage
// of ingestion. Callers may then name these scripts (and only these) on an ingest
// request. With no registry, or an empty one, the feature is off.
func (s *Server) SetScripts(r *script.Registry) { s.scripts = r }

// handleScripts lists the transform scripts the operator has made available, so
// the UI can offer them. It never reveals their paths, only their names.
func (s *Server) handleScripts(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"scripts": s.scripts.Names()})
}

// validateTransforms rejects any name that is not a registered script, so an
// unknown or crafted name fails the request instead of reaching the exec path.
func (s *Server) validateTransforms(names []string) error {
	if len(names) == 0 {
		return nil
	}
	if s.scripts == nil || s.scripts.Len() == 0 {
		return fmt.Errorf("no transform scripts are configured on this server (start it with --scripts <dir>)")
	}
	for _, n := range names {
		if !s.scripts.Has(n) {
			return fmt.Errorf("unknown transform script %q", n)
		}
	}
	return nil
}

// transformDocs runs the named scripts, in order, over every document. Each
// script's output feeds the next, so transforms compose into a pipeline.
//
// A document that a script fails on is reported and skipped rather than failing
// the whole ingest, matching how file extraction already isolates a bad file; a
// document a script drops is reported separately, because that is a decision, not
// an error. The returned documents are the ones that survived.
func (s *Server) transformDocs(ctx context.Context, names []string, docs []rag.Document) (kept []rag.Document, dropped []string, failed []ingestFailure) {
	if len(names) == 0 {
		return docs, nil, nil
	}
	for _, d := range docs {
		out, drop, err := s.runTransforms(ctx, names, d)
		switch {
		case err != nil:
			failed = append(failed, ingestFailure{d.ID, err.Error()})
		case drop:
			dropped = append(dropped, d.ID)
		default:
			kept = append(kept, out)
		}
	}
	return kept, dropped, failed
}

// runTransforms pipes one document through the named scripts in order.
func (s *Server) runTransforms(ctx context.Context, names []string, d rag.Document) (rag.Document, bool, error) {
	cur := d
	for _, name := range names {
		in := script.Doc{ID: cur.ID, Text: cur.Text}
		if cur.Meta != nil {
			b, err := json.Marshal(cur.Meta)
			if err != nil {
				return d, false, fmt.Errorf("encode metadata for %q: %w", name, err)
			}
			in.Meta = b
		}
		out, err := s.scripts.Run(ctx, name, in)
		if err != nil {
			return d, false, err
		}
		if out.Drop {
			return d, true, nil
		}
		cur.Text = out.Text
		// A script that returns no metadata leaves the document's metadata alone,
		// so a transform that only rewrites text cannot silently erase it.
		if len(out.Meta) > 0 {
			var m map[string]any
			if err := json.Unmarshal(out.Meta, &m); err != nil {
				return d, false, fmt.Errorf("script %s: metadata is not a JSON object: %w", name, err)
			}
			cur.Meta = m
		}
	}
	return cur, false, nil
}
