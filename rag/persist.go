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
	Cfg    Config
	Dim    int
	Chunks []Chunk
	Embeds [][]float32
	// Hashes maps document id to content hash, persisted so content-level dedup
	// survives a reload. Absent in older snapshots, in which case dedup falls back
	// to ids until the documents are seen again.
	Hashes map[string][32]byte
	// Entities and Relations persist the entity-relationship graph, which is
	// expensive to extract (it uses an LLM), so it is not rebuilt on load.
	Entities  []entity.Entity
	Relations []entity.Relation
	// Versions persists each document's content history. Absent in older
	// snapshots, in which case a document has no recorded history until its next
	// update.
	Versions map[string][]docVersion
	// DocMeta persists arbitrary per-document metadata as raw JSON. Absent in
	// older snapshots.
	DocMeta map[string]json.RawMessage
}

// Save serializes the store to w.
func (s *Store) Save(w io.Writer) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.hnsw == nil {
		return fmt.Errorf("rag: cannot save an empty store")
	}
	snap := snapshot{Cfg: s.cfg, Dim: s.dim, Chunks: s.chunks, Embeds: s.embeds, Hashes: s.idHash, Versions: s.versions, DocMeta: s.docMeta}
	snap.Cfg.Chunker = nil // a custom chunker is not gob-persistable; Strategy is
	if s.eg != nil {
		snap.Entities = s.eg.Entities()
		snap.Relations = s.eg.Relations()
	}
	return gob.NewEncoder(w).Encode(&snap)
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
	if len(snap.Entities) > 0 {
		s.eg = entity.Restore(snap.Entities, snap.Relations)
		s.rebuildEntityLocked()
	}
	s.reindexLocked()
	return s, nil
}
