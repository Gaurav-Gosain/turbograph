package rag

import (
	"encoding/hex"
	"time"
)

// docVersion is one immutable snapshot of a document's content, recorded each
// time the document is ingested or its content changes. The full text is kept so
// the UI can diff two versions and restore an earlier one without the original
// file. Embeddings are not snapshotted: a restore re-ingests the text through the
// normal update path, which reuses existing embeddings for unchanged chunks.
type docVersion struct {
	Hash   [32]byte
	Time   int64
	Text   string
	Chunks int
}

// DocVersion is a single version's metadata for listing, without the full text.
type DocVersion struct {
	N       int    `json:"n"`       // 1-based version number, oldest is 1
	Hash    string `json:"hash"`    // short hex content hash
	Time    int64  `json:"time"`    // unix seconds when recorded
	Bytes   int    `json:"bytes"`   // document size at this version
	Chunks  int    `json:"chunks"`  // chunks this version produced
	Current bool   `json:"current"` // whether this is the live version
}

// nowUnix returns the current unix time. It is a variable so tests can make
// version timestamps deterministic.
var nowUnix = func() int64 { return time.Now().Unix() }

// recordVersionLocked appends a content snapshot to a document's history. A
// repeat of the most recent hash is ignored, so re-ingesting identical content
// never pads the log. The caller must hold the write lock.
func (s *Store) recordVersionLocked(id string, h [32]byte, text string, chunks int) {
	if s.versions == nil {
		s.versions = make(map[string][]docVersion)
	}
	v := s.versions[id]
	if n := len(v); n > 0 && v[n-1].Hash == h {
		return
	}
	s.versions[id] = append(v, docVersion{Hash: h, Time: nowUnix(), Text: text, Chunks: chunks})
}

// DocVersions returns the version history of a document, oldest first, marking
// the newest as the current (live) version. It returns nil for an unknown
// document or one ingested before version tracking existed.
func (s *Store) DocVersions(id string) []DocVersion {
	s.mu.RLock()
	defer s.mu.RUnlock()
	vs := s.versions[id]
	if len(vs) == 0 {
		return nil
	}
	out := make([]DocVersion, len(vs))
	for i, v := range vs {
		out[i] = DocVersion{
			N:       i + 1,
			Hash:    hex.EncodeToString(v.Hash[:4]),
			Time:    v.Time,
			Bytes:   len(v.Text),
			Chunks:  v.Chunks,
			Current: i == len(vs)-1,
		}
	}
	return out
}

// DocVersionText returns the stored text of version n (1-based) of a document.
func (s *Store) DocVersionText(id string, n int) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	vs := s.versions[id]
	if n < 1 || n > len(vs) {
		return "", false
	}
	return vs[n-1].Text, true
}
