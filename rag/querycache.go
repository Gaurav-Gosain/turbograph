package rag

import (
	"strings"
	"sync"
)

// queryCache is a process-local LRU cache for query embeddings. It avoids
// repeated embedding network calls for the same query text within a single
// store's lifetime. It is not persisted and is cleared when the encoder
// fingerprint changes (which changes the vector space).
type queryCache struct {
	mu      sync.Mutex
	encoder string
	max     int                  // max entries (0 = unlimited)
	maxB    int                  // max bytes (approximate, 0 = unlimited)
	bytes   int                  // current approximate bytes stored
	items   map[string][]float32 // normalized query -> embedding vector
	order   []string             // LRU order front=MRU, back=evict-first
}

// normalizeQuery collapses whitespace, lowercases, and trims. Used as the
// cache key so that semantically identical queries share the cache entry.
func normalizeQuery(q string) string {
	if q == "" {
		return ""
	}
	out := strings.Join(strings.Fields(strings.ToLower(q)), " ")
	if len(out) == 0 {
		return ""
	}
	// Skip extremely long queries.
	if len(out) > 512 {
		return ""
	}
	return out
}

// NewQueryCache creates a new LRU cache with given limits. Pass 0 for both
// limits to create an effectively unlimited cache (bounded only by process memory).
// The encoder parameter is used to detect fingerprint changes; set to "" if
// not yet known.
func NewQueryCache(maxEntries int, maxBytes int, encoder string) *queryCache {
	return &queryCache{
		max:     maxEntries,
		maxB:    maxBytes,
		encoder: encoder,
		items:   make(map[string][]float32),
		order:   make([]string, 0, maxEntries),
	}
}

// SetEncoder updates the stored encoder fingerprint. If it differs from the
// previous value, the entire cache is cleared because different embedders
// produce incomparable vectors.
func (qc *queryCache) SetEncoder(encoder string) {
	qc.mu.Lock()
	defer qc.mu.Unlock()
	if qc.encoder != encoder {
		qc.encoder = encoder
		qc.items = make(map[string][]float32)
		qc.order = qc.order[:0]
		qc.bytes = 0
	}
}

// HasEncoder reports whether the cache currently has an encoder fingerprint
// set (non-empty).
func (qc *queryCache) HasEncoder() bool {
	qc.mu.Lock()
	defer qc.mu.Unlock()
	return qc.encoder != ""
}

// Get returns the cached embedding vector for the given query, or nil if not
// found. The returned slice is a fresh copy; callers must not modify it. On
// cache miss the returned slice is nil.
func (qc *queryCache) Get(query string) []float32 {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	key := normalizeQuery(query)
	if key == "" {
		return nil
	}

	if _, ok := qc.items[key]; !ok {
		return nil
	}

	v := qc.items[key]
	out := make([]float32, len(v))
	copy(out, v)

	// Move to front (MRU). Rebuild order slice.
	newOrder := make([]string, 0, len(qc.order))
	found := false
	for _, k := range qc.order {
		if k == key {
			found = true
			newOrder = append(newOrder, key)
		} else {
			newOrder = append(newOrder, k)
		}
	}
	if !found {
		newOrder = append(newOrder, key)
	}
	qc.order = newOrder

	return out
}

// Put stores the embedding vector for the given query. If the query already
// exists in the cache, the value is updated and the entry is moved to front
// (MRU). If the cache is at capacity (by entries or bytes), the least-recently-used
// entry (back of order) is evicted.
func (qc *queryCache) Put(query string, vec []float32) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	key := normalizeQuery(query)
	if key == "" {
		return
	}

	var delta int
	if old, exists := qc.items[key]; exists {
		delta = len(vec) - len(old)
		qc.items[key] = vec
	} else {
		delta = len(vec)
		qc.items[key] = vec
	}
	// Move to front: rebuild order, prepending the key (front = MRU, back = evict-first).
	newOrder := make([]string, 0, len(qc.order)+1)
	newOrder = append(newOrder, key)
	for _, k := range qc.order {
		if k != key {
			newOrder = append(newOrder, k)
		}
	}
	qc.order = newOrder

	qc.bytes += delta
	if qc.maxB > 0 && qc.bytes > qc.maxB {
		// Evict from the back until under the byte limit.
		for len(qc.order) > 0 && qc.bytes > qc.maxB {
			evictKey := qc.order[len(qc.order)-1]
			qc.order = qc.order[:len(qc.order)-1]
			if ev, ok := qc.items[evictKey]; ok {
				qc.bytes -= len(ev)
			}
			delete(qc.items, evictKey)
		}
	}

	// Enforce entry count limit (if set).
	for qc.max > 0 && len(qc.order) > qc.max {
		evictKey := qc.order[len(qc.order)-1]
		qc.order = qc.order[:len(qc.order)-1]
		if ev, ok := qc.items[evictKey]; ok {
			qc.bytes -= len(ev)
		}
		delete(qc.items, evictKey)
	}
}

// Reset clears all cached entries.
func (qc *queryCache) Reset() {
	qc.mu.Lock()
	defer qc.mu.Unlock()
	qc.items = make(map[string][]float32)
	qc.order = qc.order[:0]
	qc.bytes = 0
}

// EmbedIfCached queries the cache first; on hit it returns a copy of the cached
// vector and does not call the embedder. On miss it returns nil, signalling the
// caller should embed. The encoder fingerprint must already be set (via SetEncoder
// or Store.encoder) so that stale embeddings are not returned across models.
func (qc *queryCache) EmbedIfCached(query string) ([]float32, bool) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	key := normalizeQuery(query)
	if key == "" {
		return nil, false
	}

	v, ok := qc.items[key]
	if !ok {
		return nil, false
	}

	// Move to front (MRU): rebuild order, dropping the hit and prepending it.
	newOrder := make([]string, 0, len(qc.order))
	newOrder = append(newOrder, key)
	for _, k := range qc.order {
		if k != key {
			newOrder = append(newOrder, k)
		}
	}
	qc.order = newOrder

	out := make([]float32, len(v))
	copy(out, v)
	return out, true
}
