package rag

import (
	"github.com/Gaurav-Gosain/turbograph/redact"
)

// Redaction reports credentials removed from a document at ingest.
type Redaction struct {
	DocID string           `json:"doc_id"`
	Found []redact.Finding `json:"found"`
}

// SetRedaction turns credential stripping on or off for this store. It is on by
// default in every entry point turbograph ships, because a .tg is a file you hand to
// someone else and an agent-grown store is fed from tool output: shell sessions, env
// dumps, config files, CI logs. A key that reaches the store is not merely stored, it
// is packaged up and shared, and it persists in the version history even after the
// document is corrected.
//
// It is a store-level policy rather than a caller-level one so that every ingest path
// gets it: the CLI, the HTTP API, the web UI, and the MCP server alike.
func (s *Store) SetRedaction(on bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.redact = on
}

// redactDocs strips credentials from documents before anything else sees them. It runs
// ahead of chunking, embedding and version recording, which is the only point at which
// removing a secret actually removes it: after chunking it is in the index, and after
// recordVersion it is in the history that Merge copies into every shared store.
func (s *Store) redactDocs(docs []Document) ([]Document, []Redaction) {
	s.mu.RLock()
	on := s.redact
	s.mu.RUnlock()
	if !on || len(docs) == 0 {
		return docs, nil
	}
	var found []Redaction
	out := make([]Document, len(docs))
	copy(out, docs)
	for i := range out {
		clean, fs := redact.Text(out[i].Text)
		if len(fs) == 0 {
			continue
		}
		out[i].Text = clean
		found = append(found, Redaction{DocID: out[i].ID, Found: fs})
	}
	return out, found
}

// LastRedactions returns what the most recent ingest stripped out, so a caller can
// tell the user rather than silently altering their document.
func (s *Store) LastRedactions() []Redaction {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastRedactions
}
