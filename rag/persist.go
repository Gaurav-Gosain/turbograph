package rag

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/index"
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
	Cfg Config `json:"config"`
	Dim int    `json:"dim"`
	// Encoder names the vector space the store was built in (embedding model, its
	// truncation, its instruction prefixes). Absent in older snapshots, in which case a
	// merge can only compare dimensions and says so.
	Encoder string  `json:"encoder,omitempty"`
	Chunks  []Chunk `json:"chunks"`
	// Embeds is the legacy per-chunk vector layout. Reading it costs one allocation per
	// chunk, and gob's decoder was the single largest allocator in the load path: 80% of
	// it, at 100,000 chunks. New snapshots write Flat instead.
	Embeds [][]float32 `json:"embeds,omitempty"`
	// Flat holds the vectors row-major as one contiguous block. Written as raw
	// little-endian float32 bytes rather than as a gob []float32, because gob encodes each
	// float with a variable-length scheme that INFLATES 307 MB of vectors to 450 MB and
	// takes 993 ms to decode, against 307 MB and 431 ms for the raw bytes. It is also the
	// exact layout the vector index wants, so the index adopts the block rather than
	// copying it.
	FlatBytes []byte `json:"-"`
	// FlatF32 is the same block as a gob []float32. Written by exactly one release; read
	// for anyone who has such a file, never written.
	FlatF32 []float32 `json:"flat,omitempty"`
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
	// FactVec persists the query-to-fact index: one embedding per relation, with its two
	// endpoints stored BY NAME (FactSrc/FactTgt) rather than by node index so it survives
	// a reload even if the entity ordering shifts. Without it the first entity-linked query
	// after every restart re-embeds every relation inside the request. Absent in older
	// snapshots and until the entity graph is built.
	FactVec [][]float32 `json:"fact_vec,omitempty"`
	FactSrc []string    `json:"fact_src,omitempty"`
	FactTgt []string    `json:"fact_tgt,omitempty"`
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
	// ExtractCache persists what the model said about each chunk, so reopening a store
	// and adding a document does not re-extract the whole corpus. Absent in older
	// snapshots, in which case the next build pays for what it uses and fills it.
	// A slice, not a map: the key is a hash, and a JSON object cannot have one as a
	// key, which would have broken `export --json` on any store that had a cache.
	ExtractCache []cacheEntry `json:"extract_cache,omitempty"`
	// HNSW persists the vector index's link structure. Rebuilding it means a graph
	// search per vector and was by far the largest cost of opening a store; the links
	// themselves are a few percent of the vectors' size. Absent in older snapshots and
	// when the index was never built, in which case it is reconstructed as before.
	HNSW *index.Graph `json:"hnsw,omitempty"`
}

// cacheEntry is one persisted extraction, carrying the key it is stored under. The
// fields are spelled out rather than embedding cachedExtraction: gob silently drops an
// embedded field whose type is unexported, so the keys persisted and the extractions
// they pointed at came back empty, and every chunk then "hit" a cache of nothing.
type cacheEntry struct {
	Key  [32]byte          `json:"key"`
	Text [32]byte          `json:"text"`
	Ex   entity.Extraction `json:"ex"`
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
	snap := snapshot{Cfg: s.cfg, Dim: s.dim, Encoder: s.encoder, Chunks: s.chunks, Hashes: s.idHash, Versions: s.versions,
		DocMeta: s.docMeta, CommSummary: s.commSummary, ExtractCache: flattenCache(s.extractCache),
		HNSW: s.hnswSnapshot()}
	snap.Cfg.Chunker = nil // a custom chunker is not gob-persistable; Strategy is
	switch mode {
	case VectorsCodes:
		// Compact codes in place of the float32 vectors; decoded on load.
		q := s.quantizer()
		snap.Codes = make([]quant.Code, len(s.embeds))
		for i, v := range s.embeds {
			snap.Codes[i] = q.Encode(v)
		}
	case VectorsNone:
		// Store no vectors; they are recomputed from text on load.
	default:
		// One contiguous block of raw bytes, not one gob-encoded slice per chunk. See the
		// FlatBytes field for why raw.
		snap.FlatBytes = packVectors(s.embeds, s.dim)
	}
	if s.eg != nil {
		snap.Entities = s.eg.Entities()
		snap.Relations = s.eg.Relations()
		if len(s.entVec) == len(s.entList) {
			snap.EntVec = s.entVec
		}
		// Persist the fact index by endpoint NAME, so it can be remapped to whatever node
		// ordering a reload produces.
		if n := len(s.factVec); n > 0 && n == len(s.factSrc) && n == len(s.factTgt) {
			snap.FactVec = s.factVec
			snap.FactSrc = make([]string, n)
			snap.FactTgt = make([]string, n)
			for i := 0; i < n; i++ {
				snap.FactSrc[i] = s.entList[s.factSrc[i]].Name
				snap.FactTgt[i] = s.entList[s.factTgt[i]].Name
			}
		}
	}
	return writeSnapshot(w, &snap)
}

// vecMagic marks a store whose vector block is written outside the gob stream.
//
// The first byte of a gob stream is always a message length, which gob encodes as a
// single byte below 0x80 or as a 0xF8..0xFF marker followed by the bytes. So 0x80 can
// never begin a gob stream, and a file starting with it is unambiguously this format
// and not a store written before it.
var vecMagic = []byte{0x80, 'T', 'G', 'V'}

// writeSnapshot writes the vector block raw, ahead of the gob, with a length prefix.
//
// The block used to go through gob as a []byte, and gob reads a large slice with
// saferio, which grows it incrementally rather than allocating it once: that was 70% of
// all the allocation in opening a store, for data whose size is known exactly. Written
// on its own it is one allocation and one io.ReadFull.
func writeSnapshot(w io.Writer, snap *snapshot) error {
	raw := snap.FlatBytes
	snap.FlatBytes = nil // never inside the gob
	if len(raw) == 0 {
		return gob.NewEncoder(w).Encode(snap)
	}
	if _, err := w.Write(vecMagic); err != nil {
		return err
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(raw)))
	if _, err := w.Write(n[:]); err != nil {
		return err
	}
	if _, err := w.Write(raw); err != nil {
		return err
	}
	return gob.NewEncoder(w).Encode(snap)
}

// readSnapshot reads a store written by writeSnapshot, or one written before it.
func readSnapshot(r io.Reader) (snapshot, error) {
	var snap snapshot
	br := bufio.NewReaderSize(r, 1<<16)
	head, err := br.Peek(len(vecMagic))
	if err == nil && bytes.Equal(head, vecMagic) {
		if _, err := br.Discard(len(vecMagic)); err != nil {
			return snap, err
		}
		var n [8]byte
		if _, err := io.ReadFull(br, n[:]); err != nil {
			return snap, fmt.Errorf("rag: truncated vector block header: %w", err)
		}
		size := binary.LittleEndian.Uint64(n[:])
		if size > 1<<40 {
			return snap, fmt.Errorf("rag: implausible vector block of %d bytes", size)
		}
		raw := make([]byte, size)
		if _, err := io.ReadFull(br, raw); err != nil {
			return snap, fmt.Errorf("rag: truncated vector block: %w", err)
		}
		if err := gob.NewDecoder(br).Decode(&snap); err != nil {
			return snap, err
		}
		snap.FlatBytes = raw
		return snap, nil
	}
	// A store written before the vector block was hoisted out of the gob.
	err = gob.NewDecoder(br).Decode(&snap)
	return snap, err
}

// loadVectors materializes the chunk embeddings from whichever representation the
// snapshot carries: the stored float32 vectors (VectorsExact), TurboQuant codes
// decoded through the store's quantizer (VectorsCodes), or a re-embedding of the
// chunk text with the attached embedder (VectorsNone). The store must already be
// initialized so its quantizer exists. It is called under construction, before
// the store is published, so no lock is needed.
func (s *Store) loadVectors(snap *snapshot) ([][]float32, error) {
	switch {
	case snap.Dim > 0 && len(snap.FlatBytes) == len(snap.Chunks)*snap.Dim*4:
		return s.viewFlat(unpackVectors(snap.FlatBytes), snap.Dim, len(snap.Chunks)), nil
	case snap.Dim > 0 && len(snap.FlatF32) == len(snap.Chunks)*snap.Dim:
		return s.viewFlat(snap.FlatF32, snap.Dim, len(snap.Chunks)), nil
	case len(snap.Embeds) == len(snap.Chunks):
		return snap.Embeds, nil
	case len(snap.Codes) == len(snap.Chunks):
		q := s.quantizer()
		out := make([][]float32, len(snap.Codes))
		for i := range snap.Codes {
			out[i] = q.Decode(snap.Codes[i])
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
	snap, err := readSnapshot(r)
	if err != nil {
		return fmt.Errorf("rag: decode: %w", err)
	}
	// The vectors are stored as one raw block, which is not a JSON-representable thing.
	// The interop format is per-chunk arrays, so materialize them: an export that quietly
	// dropped the vectors because the on-disk layout changed would be a silent data loss
	// in the one command whose entire job is to hand the corpus to another tool.
	if includeVectors && len(snap.Embeds) == 0 && snap.Dim > 0 {
		var flat []float32
		switch {
		case len(snap.FlatBytes) == len(snap.Chunks)*snap.Dim*4:
			flat = unpackVectors(snap.FlatBytes)
		case len(snap.FlatF32) == len(snap.Chunks)*snap.Dim:
			flat = snap.FlatF32
		}
		if flat != nil {
			snap.Embeds = make([][]float32, len(snap.Chunks))
			for i := range snap.Embeds {
				snap.Embeds[i] = flat[i*snap.Dim : (i+1)*snap.Dim]
			}
		}
	}
	snap.FlatBytes, snap.FlatF32 = nil, nil
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
	snap, err := readSnapshot(r)
	if err != nil {
		return nil, fmt.Errorf("rag: decode: %w", err)
	}
	snap.Cfg.withDefaults()
	// redact: true, matching New. Load builds the struct directly, so a default that
	// lives only in New would silently be off for every store opened from disk, which is
	// every store an agent ever touches.
	s := &Store{cfg: snap.Cfg, embedder: embedder, dim: snap.Dim, redact: true}
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
	// Fill the source-of-truth arrays, but do not build the vector index, the lexical
	// index, the similarity graph or the communities. They are derived, and rebuilding
	// them here was two thirds of the cost of opening a store, paid by every command
	// whether or not it ever searched. ensureIndex builds them on first use.
	s.appendToArraysLocked(snap.Chunks, embeds)
	// deferIndex already means "not built yet"; needsRebuild means "built but stale". They
	// are distinct, and conflating them (setting needsRebuild here) both hid the persisted
	// links from a re-save and was simply redundant, since ensureSearchLocked builds the
	// index from the arrays either way.
	s.deferIndex = true
	s.deferGraph = true
	s.savedHNSW = snap.HNSW
	for id, h := range snap.Hashes {
		s.recordHashLocked(id, h)
	}
	s.versions = snap.Versions
	s.docMeta = snap.DocMeta
	s.encoder = snap.Encoder
	s.commSummary = snap.CommSummary
	if len(snap.ExtractCache) > 0 {
		s.extractCache = make(map[[32]byte]cachedExtraction, len(snap.ExtractCache))
		for _, e := range snap.ExtractCache {
			s.extractCache[e.Key] = cachedExtraction{Text: e.Text, Ex: e.Ex}
		}
	}
	if len(snap.Entities) > 0 {
		s.eg = entity.Restore(snap.Entities, snap.Relations)
		s.rebuildEntityLocked()
		if len(snap.EntVec) == len(s.entList) {
			s.entVec = snap.EntVec
		}
		// Restore the fact index, remapping each relation's endpoints from names to the
		// current node indices and dropping any whose endpoints no longer exist.
		if n := len(snap.FactVec); n > 0 && n == len(snap.FactSrc) && n == len(snap.FactTgt) {
			fv := make([][]float32, 0, n)
			fs := make([]int, 0, n)
			ft := make([]int, 0, n)
			for i := 0; i < n; i++ {
				si, ok1 := s.entIndex[snap.FactSrc[i]]
				ti, ok2 := s.entIndex[snap.FactTgt[i]]
				if ok1 && ok2 {
					fv = append(fv, snap.FactVec[i])
					fs = append(fs, si)
					ft = append(ft, ti)
				}
			}
			s.factVec, s.factSrc, s.factTgt = fv, fs, ft
		}
	}
	return s, nil
}

// flattenCache turns the in-memory extraction cache into its persisted form, sorted
// so a save is byte-stable for the same contents.
func flattenCache(m map[[32]byte]cachedExtraction) []cacheEntry {
	if len(m) == 0 {
		return nil
	}
	out := make([]cacheEntry, 0, len(m))
	for k, v := range m {
		out = append(out, cacheEntry{Key: k, Text: v.Text, Ex: v.Ex})
	}
	sort.Slice(out, func(i, j int) bool { return bytes.Compare(out[i].Key[:], out[j].Key[:]) < 0 })
	return out
}

// hnswSnapshot exports the vector index's links, if it is built and current. A stale
// or absent graph is simply not written: it is a cache, and a cache miss costs a
// rebuild, while a wrong graph would silently return the wrong neighbours forever.
func (s *Store) hnswSnapshot() *index.Graph {
	// The index is built and current: snapshot it.
	if !s.deferIndex && !s.needsRebuild && s.hnsw != nil && s.hnsw.Len() == len(s.chunks) {
		g := s.hnsw.Snapshot()
		return &g
	}
	// The index was never built this session -- the store was loaded and saved again
	// without a search, which is what `turbograph add` of an unchanged document does, and
	// what any no-op save after server startup does. The links read from disk still
	// describe the current chunks, so persist them again rather than dropping them and
	// forcing the next open to reconstruct the whole graph.
	if s.deferIndex && !s.needsRebuild && s.savedHNSW != nil && len(s.savedHNSW.Levels) == len(s.chunks) {
		return s.savedHNSW
	}
	return nil
}

// viewFlat hands back per-chunk views into one contiguous block, and keeps the block so
// the vector index can adopt it. No vector is copied, at load or at index build.
func (s *Store) viewFlat(flat []float32, dim, n int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*dim : (i+1)*dim : (i+1)*dim]
	}
	s.flat = flat
	return out
}

// packVectors writes per-chunk vectors as one row-major block of little-endian float32.
func packVectors(vecs [][]float32, dim int) []byte {
	if len(vecs) == 0 || dim <= 0 {
		return nil
	}
	out := make([]byte, len(vecs)*dim*4)
	k := 0
	for _, v := range vecs {
		if len(v) != dim {
			return nil // ragged; fall back to the per-chunk layout
		}
		for _, f := range v {
			binary.LittleEndian.PutUint32(out[k:], math.Float32bits(f))
			k += 4
		}
	}
	return out
}

// unpackVectors reads the block back.
func unpackVectors(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
