package rag

import (
	"encoding/json"
	"fmt"
	"maps"
)

// MergeStats reports what a Merge did, so the caller can tell the difference
// between a merge that added knowledge and one that was a no-op.
type MergeStats struct {
	Documents int // documents copied from src
	Chunks    int // chunks copied from src
	Skipped   int // documents src had that dst already knew, by id or by content
	Cached    int // extraction-cache entries carried over, so the entity graph is not re-extracted
}

// Merge copies everything src knows into dst, without re-embedding anything: the
// chunks come across with the vectors they were built with. This is what makes a
// .tg file a unit of knowledge you can hand to someone else. Two people index
// different corpora, exchange stores, and merge them into one.
//
// A document dst already has, by id or by content, is skipped rather than
// duplicated, so merging the same store twice is a no-op and merging two stores
// that share sources does not double-count them.
//
// The entity graph is NOT merged: entities and relationships are canonicalized
// across the whole corpus, so a union of two graphs is not the graph of the union.
// The extraction cache IS merged, which means rebuilding the entity graph on the
// merged store re-reads nothing and costs almost nothing. Call BuildEntityGraph
// afterwards to get a correct graph cheaply.
func Merge(dst, src *Store) (MergeStats, error) {
	if dst == nil || src == nil {
		return MergeStats{}, fmt.Errorf("rag: merge needs two stores")
	}
	if dst == src {
		return MergeStats{}, fmt.Errorf("rag: cannot merge a store into itself")
	}

	src.mu.RLock()
	srcDim := src.dim
	srcEnc := src.encoder
	srcChunks := make([]Chunk, len(src.chunks))
	copy(srcChunks, src.chunks)
	srcEmbeds := make([][]float32, len(src.embeds))
	copy(srcEmbeds, src.embeds)
	srcHash := maps.Clone(src.idHash)
	srcMeta := maps.Clone(src.docMeta)
	srcVersions := maps.Clone(src.versions)
	srcCache := maps.Clone(src.extractCache)
	src.mu.RUnlock()

	if len(srcChunks) == 0 {
		return MergeStats{}, nil
	}
	if len(srcEmbeds) != len(srcChunks) {
		return MergeStats{}, fmt.Errorf("rag: source store has %d chunks but %d vectors; it was saved without them (--lean text) and cannot be merged",
			len(srcChunks), len(srcEmbeds))
	}

	dst.mu.Lock()
	if dst.dim != 0 && srcDim != 0 && dst.dim != srcDim {
		dst.mu.Unlock()
		return MergeStats{}, fmt.Errorf("rag: cannot merge a store of dim %d into one of dim %d; they were built with different embedding models",
			srcDim, dst.dim)
	}
	// A matching dimension is not a matching vector space. Two 768-dimensional models do
	// not agree on which 768 dimensions, so merging them produces an index whose
	// distances are meaningless: nothing fails, retrieval just quietly gets worse and
	// stays that way. Refuse it.
	if dstEnc := dst.encoder; dstEnc != "" && srcEnc != "" && dstEnc != srcEnc {
		dst.mu.Unlock()
		return MergeStats{}, fmt.Errorf("rag: refusing to merge stores built with different embedders;\n  into: %s\n  from: %s\nre-index one of them with the other's embedding model",
			dstEnc, srcEnc)
	}
	// Decide which of src's documents are new. A document is already known if dst has
	// its id, or if dst has its exact content under any id.
	take := make(map[string]bool, len(srcHash))
	var skipped int
	for id := range srcHash {
		h := srcHash[id]
		_, byID := dst.idHash[id]
		_, byContent := dst.hashes[h]
		if byID || byContent {
			skipped++
			continue
		}
		take[id] = true
	}
	// A store written before content hashing has no idHash entries; fall back to the
	// document ids present in its chunks so it can still be merged.
	if len(srcHash) == 0 {
		for _, c := range srcChunks {
			if _, known := dst.docSet[c.DocID]; !known {
				take[c.DocID] = true
			} else {
				skipped++
			}
		}
	}
	dst.mu.Unlock()

	chunks := make([]Chunk, 0, len(srcChunks))
	vecs := make([][]float32, 0, len(srcChunks))
	for i, c := range srcChunks {
		if take[c.DocID] {
			chunks = append(chunks, c)
			vecs = append(vecs, srcEmbeds[i])
		}
	}
	if len(chunks) > 0 {
		if err := dst.AddEmbedded(chunks, vecs); err != nil {
			return MergeStats{}, err
		}
	}

	dst.mu.Lock()
	for id := range take {
		if h, ok := srcHash[id]; ok {
			if dst.idHash == nil {
				dst.idHash = map[string][32]byte{}
			}
			if dst.hashes == nil {
				dst.hashes = map[[32]byte]string{}
			}
			dst.idHash[id] = h
			dst.hashes[h] = id
		}
		if m, ok := srcMeta[id]; ok {
			if dst.docMeta == nil {
				dst.docMeta = map[string]json.RawMessage{}
			}
			dst.docMeta[id] = m
		}
		if v, ok := srcVersions[id]; ok {
			if dst.versions == nil {
				dst.versions = map[string][]docVersion{}
			}
			dst.versions[id] = v
		}
	}
	// Carry the extraction cache across for every chunk that came with it, so the
	// merged store's entity graph can be rebuilt without asking the model again. This
	// is the difference between a merge costing seconds and costing many minutes.
	var cached int
	if len(srcCache) > 0 {
		if dst.extractCache == nil {
			dst.extractCache = make(map[[32]byte]cachedExtraction, len(srcCache))
		}
		for k, v := range srcCache {
			if _, dup := dst.extractCache[k]; dup {
				continue
			}
			dst.extractCache[k] = v
			cached++
		}
		// Drop anything whose text is not in the merged corpus, so a merge cannot smuggle
		// in cache entries for documents that were skipped as duplicates.
		dst.pruneExtractCacheLocked()
	}
	dst.mu.Unlock()

	dst.Reindex()

	return MergeStats{
		Documents: len(take),
		Chunks:    len(chunks),
		Skipped:   skipped,
		Cached:    cached,
	}, nil
}
