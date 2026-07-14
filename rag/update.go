package rag

import (
	"context"
	"fmt"

	"github.com/Gaurav-Gosain/turbograph/lexical"
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
	s.contextualize(ctx, []Document{d}, chunks)
	reuse := s.docEmbeddings(d.ID) // empty for a new document
	vecs := make([][]float32, len(chunks))
	var missText []string
	var missIdx []int
	for i, c := range chunks {
		// Reuse a cached embedding only on the plain path: a contextual prefix
		// changes the indexed text, so a body-hash match would be the wrong vector.
		if emb, ok := reuse[contentHash(c.Text)]; ok && len(emb) > 0 && c.Context == "" {
			vecs[i] = emb
		} else {
			missIdx = append(missIdx, i)
			missText = append(missText, c.IndexText())
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
	// Note which chunks are about to disappear, so the entity graph can be kept honest.
	gone := make(map[string]bool)
	for _, c := range s.chunks {
		if c.DocID == id {
			gone[c.ID] = true
		}
	}
	removed := s.removeDocLocked(id)
	delete(s.docMeta, id)
	delete(s.versions, id)
	if removed > 0 {
		s.commSummary = nil // graph changed; community summaries are now stale
		// The entity graph cites chunks. Forgetting a document must forget the entities
		// that only that document evidenced, or entity-seeded retrieval keeps handing back
		// chunk ids that are no longer in the store. This needs no model: an entity whose
		// every mention has been deleted is simply no longer supported by anything.
		if s.eg != nil {
			s.eg.DropChunks(gone)
			s.rebuildEntityLocked()
		}
		// The extraction cache remembers what the model SAID about each chunk: entity
		// names and descriptions derived from its text. Left behind, they are written into
		// the .tg and handed to whoever you share it with, so a forgotten document is not
		// forgotten at all, only hidden from retrieval. Prune it against the live corpus.
		s.pruneExtractCacheLocked()
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
	}
	s.rebuildLexicalLocked()
}

// rebuildLexicalLocked rebuilds only the BM25 index. It is separate because the vector
// index can be restored from disk while the lexical one cannot usefully be: it is
// tokenization, with no distance computations, so recomputing it is cheap and storing
// it would bloat every .tg for no gain.
func (s *Store) rebuildLexicalLocked() {
	s.bm25 = lexical.New(lexical.DefaultConfig())
	ids := make([]string, len(s.chunks))
	texts := make([]string, len(s.chunks))
	for i := range s.chunks {
		ids[i] = s.chunks[i].ID
		texts[i] = s.chunks[i].IndexText()
	}
	// Tokenizing the corpus is the largest remaining cost of opening a store, and it is
	// per-document and independent, so it runs across all cores.
	s.bm25.AddBatch(ids, texts)
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
		s.encoder = encoderOf(s.embedder)
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
		// The document's text is being replaced, so every entity mention citing its old
		// chunks is evidence for a passage that no longer exists. Chunk ids are docID#pos,
		// so the new text REUSES the same ids: leaving the graph alone does not merely
		// strand the mentions, it silently re-points each one at different content. Drop
		// them, exactly as a delete does. Must run before removeDocLocked, while the old
		// chunks are still there to be identified.
		s.dropEntityChunksLocked(p.id)
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

// dropEntityChunksLocked removes a document's chunks from the entity graph and prunes
// the extraction cache. Both the delete path and the update path need it: an update
// replaces a document's text under the same chunk ids, so entity mentions that are not
// dropped end up citing a passage whose content has silently changed underneath them.
func (s *Store) dropEntityChunksLocked(id string) {
	gone := make(map[string]bool)
	for _, c := range s.chunks {
		if c.DocID == id {
			gone[c.ID] = true
		}
	}
	if len(gone) == 0 {
		return
	}
	if s.eg != nil {
		s.eg.DropChunks(gone)
		s.rebuildEntityLocked()
	}
}
