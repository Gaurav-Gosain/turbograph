package rag

import (
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/quant"
)

// VectorMode selects how chunk embeddings are persisted, trading storage for
// load cost. It is the turbograph-native answer to low-storage indexing: the
// corpus fits in RAM, so vectors are materialized once at load rather than
// recomputed per query.
type VectorMode int

const (
	// VectorsExact stores the raw float32 embeddings. Largest on disk, fastest to
	// load (no recomputation), exact search. The default and backward-compatible.
	VectorsExact VectorMode = iota
	// VectorsCodes stores the compact TurboQuant codes instead of the float32
	// vectors and decodes them to approximate vectors on load. Much smaller, still
	// no recomputation, at quantization-level accuracy.
	VectorsCodes
	// VectorsNone stores no vectors at all; the embeddings are recomputed from the
	// chunk text with the attached embedder on load. Smallest on disk and exact,
	// but the load re-embeds the whole corpus, so it is slow. Requires the same
	// embedding model that built the store.
	VectorsNone
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
	// Codes holds the TurboQuant codes when the store was saved in VectorsCodes
	// mode (Embeds is then empty). Decoded to approximate vectors on load.
	Codes []quant.Code `json:"codes,omitempty"`
	// Hashes maps document id to content hash, persisted so content-level dedup
	// survives a reload. Absent in older snapshots, in which case dedup falls back
	// to ids until the documents are seen again.
	Hashes map[string][32]byte `json:"hashes"`
	// Entities and Relations persist the entity-relationship graph, which is
	// expensive to extract (it uses an LLM), so it is not rebuilt on load.
	Entities  []entity.Entity   `json:"entities"`
	Relations []entity.Relation `json:"relations"`
	// EntVec persists the per-entity embeddings used for dense PPR seeding, so a
	// reload does not have to re-embed the entities. Absent in older snapshots.
	EntVec [][]float32 `json:"ent_vec,omitempty"`
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

// Save serializes the store to w with exact float32 embeddings (VectorsExact).
func (s *Store) Save(w io.Writer) error { return s.SaveLean(w, VectorsExact) }

// SaveLean serializes the store to w, choosing how embeddings are persisted per
// mode. VectorsCodes and VectorsNone shrink the snapshot substantially; see
// VectorMode for the trade-offs. Everything else (chunks, hashes, versions,
// metadata, the entity graph) is stored identically regardless of mode.
func (s *Store) SaveLean(w io.Writer, mode VectorMode) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.hnsw == nil {
		return fmt.Errorf("rag: cannot save an empty store")
	}
	snap := snapshot{Cfg: s.cfg, Dim: s.dim, Chunks: s.chunks, Hashes: s.idHash, Versions: s.versions, DocMeta: s.docMeta, CommSummary: s.commSummary}
	snap.Cfg.Chunker = nil // a custom chunker is not gob-persistable; Strategy is
	switch mode {
	case VectorsCodes:
		// Compact codes in place of the float32 vectors; decoded on load.
		snap.Codes = make([]quant.Code, len(s.embeds))
		for i, v := range s.embeds {
			snap.Codes[i] = s.q.Encode(v)
		}
	case VectorsNone:
		// Store no vectors; they are recomputed from text on load.
	default:
		snap.Embeds = s.embeds
	}
	if s.eg != nil {
		snap.Entities = s.eg.Entities()
		snap.Relations = s.eg.Relations()
		if len(s.entVec) == len(s.entList) {
			snap.EntVec = s.entVec
		}
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// loadVectors materializes the chunk embeddings from whichever representation the
// snapshot carries: the stored float32 vectors (VectorsExact), TurboQuant codes
// decoded through the store's quantizer (VectorsCodes), or a re-embedding of the
// chunk text with the attached embedder (VectorsNone). The store must already be
// initialized so its quantizer exists. It is called under construction, before
// the store is published, so no lock is needed.
func (s *Store) loadVectors(snap *snapshot) ([][]float32, error) {
	switch {
	case len(snap.Embeds) == len(snap.Chunks):
		return snap.Embeds, nil
	case len(snap.Codes) == len(snap.Chunks):
		out := make([][]float32, len(snap.Codes))
		for i := range snap.Codes {
			out[i] = s.q.Decode(snap.Codes[i])
		}
		return out, nil
	default:
		if s.embedder == nil {
			return nil, fmt.Errorf("rag: snapshot has no stored vectors and no embedder to recompute them")
		}
		texts := make([]string, len(snap.Chunks))
		for i, c := range snap.Chunks {
			texts[i] = c.IndexText()
		}
		out, err := s.embedder.Embed(context.Background(), texts)
		if err != nil {
			return nil, fmt.Errorf("rag: recompute embeddings on load: %w", err)
		}
		if len(out) != len(texts) {
			return nil, fmt.Errorf("rag: recompute returned %d vectors for %d chunks", len(out), len(texts))
		}
		return out, nil
	}
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
	embeds, err := s.loadVectors(&snap)
	if err != nil {
		return nil, err
	}
	if err := s.appendChunksLocked(snap.Chunks, embeds); err != nil {
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
		if len(snap.EntVec) == len(s.entList) {
			s.entVec = snap.EntVec
		}
	}
	s.reindexLocked()
	return s, nil
}
