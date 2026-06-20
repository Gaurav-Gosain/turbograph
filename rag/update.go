package rag

import (
	"context"
	"fmt"
)

// prepared is a document that has been chunked and embedded off the write lock,
// ready to be applied to the store. It is the unit both the synchronous and the
// streaming ingestion paths produce, so versioning logic lives in one place.
type prepared struct {
	id     string
	hash   [32]byte
	text   string         // original document text, kept for the version history
	meta   map[string]any // user metadata to attach to the document
	chunks []Chunk
	vecs   [][]float32
}

// docEmbeddings returns the embeddings of a document's current chunks, keyed by
// the content hash of each chunk's text. An update reuses these for chunks whose
// text is unchanged, so only the parts of a document that actually changed are
// re-embedded.
func (s *Store) docEmbeddings(id string) map[[32]byte][]float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[[32]byte][]float32)
	for i := range s.chunks {
		if s.chunks[i].DocID == id {
			out[contentHash(s.chunks[i].Text)] = s.embeds[i]
		}
	}
	return out
}

// idHashOf returns the stored content hash for a document id.
func (s *Store) idHashOf(id string) ([32]byte, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	h, ok := s.idHash[id]
	return h, ok
}

// prepareDoc chunks and embeds a document off the write lock. For an update (the
// id already exists) it diffs the new chunks against the current ones by content
// hash and embeds only the chunks whose text changed, reusing existing
// embeddings for the rest.
func (s *Store) prepareDoc(ctx context.Context, d Document) (prepared, error) {
	h := contentHash(d.Text)
	chunks := s.ChunkDocument(d)
	if len(chunks) == 0 {
		return prepared{id: d.ID, hash: h, text: d.Text, meta: d.Meta}, nil
	}
	reuse := s.docEmbeddings(d.ID) // empty for a new document
	vecs := make([][]float32, len(chunks))
	var missText []string
	var missIdx []int
	for i, c := range chunks {
		if emb, ok := reuse[contentHash(c.Text)]; ok && len(emb) > 0 {
			vecs[i] = emb
		} else {
			missIdx = append(missIdx, i)
			missText = append(missText, c.Text)
		}
	}
	if len(missText) > 0 {
		embedded, err := s.embedder.Embed(ctx, missText)
		if err != nil {
			return prepared{}, err
		}
		if len(embedded) != len(missText) {
			return prepared{}, fmt.Errorf("rag: embedder returned %d vectors for %d texts", len(embedded), len(missText))
		}
		for k, idx := range missIdx {
			vecs[idx] = embedded[k]
		}
	}
	return prepared{id: d.ID, hash: h, text: d.Text, meta: d.Meta, chunks: chunks, vecs: vecs}, nil
}

// removeDocLocked deletes all chunks of a document from the source-of-truth
// arrays and its hash mappings. Because the vector and lexical indexes are
// append-only, it flags them for rebuild. The caller must hold the write lock.
func (s *Store) removeDocLocked(id string) int {
	n := 0
	for i := range s.chunks {
		if s.chunks[i].DocID != id {
			s.chunks[n] = s.chunks[i]
			s.embeds[n] = s.embeds[i]
			n++
		}
	}
	removed := len(s.chunks) - n
	s.chunks = s.chunks[:n]
	s.embeds = s.embeds[:n]
	if removed > 0 {
		s.needsRebuild = true
	}
	delete(s.docSet, id)
	if h, ok := s.idHash[id]; ok {
		delete(s.hashes, h)
		delete(s.idHash, id)
	}
	return removed
}

// DeleteDocument removes a document and all of its chunks from the store,
// dropping its metadata and version history, and rebuilds the indexes. It returns
// the number of chunks removed (0 if the document was not present).
func (s *Store) DeleteDocument(id string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	removed := s.removeDocLocked(id)
	delete(s.docMeta, id)
	delete(s.versions, id)
	if removed > 0 {
		s.commSummary = nil // graph changed; community summaries are now stale
		s.reindexLocked()
	}
	return removed
}

// appendToArraysLocked appends chunks to the source-of-truth arrays without
// touching the vector or lexical indexes (which a pending rebuild will recreate).
func (s *Store) appendToArraysLocked(chunks []Chunk, vecs [][]float32) {
	if s.docSet == nil {
		s.docSet = make(map[string]struct{})
	}
	for i, c := range chunks {
		s.chunks = append(s.chunks, c)
		s.embeds = append(s.embeds, vecs[i])
		s.docSet[c.DocID] = struct{}{}
	}
}

// rebuildIndexesLocked recreates the vector and lexical indexes from the current
// chunks and embeddings. It does not re-embed; the embeddings are the source of
// truth. The caller must hold the write lock.
func (s *Store) rebuildIndexesLocked() {
	s.initIndexes()
	for i := range s.chunks {
		s.hnsw.Add(s.chunks[i].ID, s.embeds[i])
		s.bm25.Add(s.chunks[i].ID, s.chunks[i].Text)
	}
	s.bm25.Finalize()
}

// applyPreparedLocked installs a prepared document, treating it as an update when
// the id already exists. It returns whether anything changed. The caller must
// hold the write lock and call reindexLocked afterward.
func (s *Store) applyPreparedLocked(p prepared) bool {
	if len(p.chunks) == 0 {
		return false
	}
	if s.hnsw == nil {
		s.dim = len(p.vecs[0])
		s.initIndexes()
	}
	if len(p.vecs[0]) != s.dim {
		return false
	}
	// Identical content already present under a different id is a duplicate.
	if owner, ok := s.hashes[p.hash]; ok && owner != p.id {
		return false
	}
	if _, idExists := s.docSet[p.id]; idExists {
		if s.idHash[p.id] == p.hash {
			return false // unchanged
		}
		s.removeDocLocked(p.id)
		s.appendToArraysLocked(p.chunks, p.vecs)
		s.recordHashLocked(p.id, p.hash)
		s.recordVersionLocked(p.id, p.hash, p.text, len(p.chunks))
		if len(p.meta) > 0 { // a content update keeps existing metadata unless new is given
			s.recordMetaLocked(p.id, p.meta)
		}
		return true
	}
	// New document. Skip the incremental index add if a rebuild is already pending
	// for this batch, since the rebuild will re-add everything from the arrays.
	if s.needsRebuild {
		s.appendToArraysLocked(p.chunks, p.vecs)
	} else if err := s.appendChunksLocked(p.chunks, p.vecs); err != nil {
		s.appendToArraysLocked(p.chunks, p.vecs)
		s.needsRebuild = true
	}
	s.recordHashLocked(p.id, p.hash)
	s.recordVersionLocked(p.id, p.hash, p.text, len(p.chunks))
	if len(p.meta) > 0 {
		s.recordMetaLocked(p.id, p.meta)
	}
	return true
}

// applyPrepared installs a prepared document under the write lock.
func (s *Store) applyPrepared(p prepared) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.applyPreparedLocked(p)
}
