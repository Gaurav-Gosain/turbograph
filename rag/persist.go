package rag

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"

	"github.com/Gaurav-Gosain/turbograph/entity"
)

// snapshot is the on-disk representation of a Store. Only the inputs that are
// expensive to recompute are stored: the chunks and their embeddings. The vector
// index, lexical index, similarity graph, and communities are deterministically
// rebuilt on load, which keeps the format small, forward-compatible, and immune
// to index-internal layout changes.
type snapshot struct {
	Cfg    Config      `json:"config"`
	Dim    int         `json:"dim"`
	Chunks []Chunk     `json:"chunks"`
	Embeds [][]float32 `json:"embeds"`
	// Hashes maps document id to content hash, persisted so content-level dedup
	// survives a reload. Absent in older snapshots, in which case dedup falls back
	// to ids until the documents are seen again.
	Hashes map[string][32]byte `json:"hashes"`
	// Entities and Relations persist the entity-relationship graph, which is
	// expensive to extract (it uses an LLM), so it is not rebuilt on load.
	Entities  []entity.Entity   `json:"entities"`
	Relations []entity.Relation `json:"relations"`
	// Versions persists each document's content history. Absent in older
	// snapshots, in which case a document has no recorded history until its next
	// update.
	Versions map[string][]docVersion `json:"versions"`
	// DocMeta persists arbitrary per-document metadata as raw JSON. Absent in
	// older snapshots.
	DocMeta map[string]json.RawMessage `json:"doc_meta"`
	// CommSummary persists per-community thematic summaries for global queries.
	// Absent in older snapshots and when summaries were never built.
	CommSummary map[int]string `json:"community_summaries"`
}

// Save serializes the store to w.
func (s *Store) Save(w io.Writer) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.hnsw == nil {
		return fmt.Errorf("rag: cannot save an empty store")
	}
	snap := snapshot{Cfg: s.cfg, Dim: s.dim, Chunks: s.chunks, Embeds: s.embeds, Hashes: s.idHash, Versions: s.versions, DocMeta: s.docMeta, CommSummary: s.commSummary}
	snap.Cfg.Chunker = nil // a custom chunker is not gob-persistable; Strategy is
	if s.eg != nil {
		snap.Entities = s.eg.Entities()
		snap.Relations = s.eg.Relations()
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// ExportJSON reads a gob-encoded .tg snapshot from r and writes an equivalent,
// indented JSON document to w. The on-disk format is Go gob, which is
// Go-specific; this is the supported interop path for other languages and tools,
// producing a plain JSON view of the same data (config, chunks with their
// document offsets, embeddings, per-document metadata, version history, and the
// entity graph). Set includeVectors to false to omit the embeddings, which
// dominate the size, when only the text and structure are needed.
func ExportJSON(r io.Reader, w io.Writer, includeVectors bool) error {
	var snap snapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return fmt.Errorf("rag: decode: %w", err)
	}
	if !includeVectors {
		snap.Embeds = nil
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(snap); err != nil {
		return fmt.Errorf("rag: encode json: %w", err)
	}
	return nil
}

// Load reconstructs a store from r, attaching the given embedder for queries.
// Indexes are rebuilt from the stored embeddings, so loading is as fast as
// indexing minus the embedding step (the expensive part is already done).
func Load(embedder Embedder, r io.Reader) (*Store, error) {
	var snap snapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return nil, fmt.Errorf("rag: decode: %w", err)
	}
	snap.Cfg.withDefaults()
	s := &Store{cfg: snap.Cfg, embedder: embedder, dim: snap.Dim}
	if len(snap.Chunks) == 0 {
		return s, nil
	}
	s.initIndexes()
	s.docSet = make(map[string]struct{})
	s.hashes = make(map[[32]byte]string)
	s.idHash = make(map[string][32]byte)
	if err := s.appendChunksLocked(snap.Chunks, snap.Embeds); err != nil {
		return nil, err
	}
	for id, h := range snap.Hashes {
		s.recordHashLocked(id, h)
	}
	s.versions = snap.Versions
	s.docMeta = snap.DocMeta
	s.commSummary = snap.CommSummary
	if len(snap.Entities) > 0 {
		s.eg = entity.Restore(snap.Entities, snap.Relations)
		s.rebuildEntityLocked()
	}
	s.reindexLocked()
	return s, nil
}
