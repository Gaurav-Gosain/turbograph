package rag

import (
	"context"
	"sync/atomic"
	"testing"
)

// TestQueryCacheHitSavesEmbed pins TZ O5: the second retrieve of the same query
// must not call the embedder again, and a whitespace variation of the query hits
// the same normalized entry.
func TestQueryCacheHitSavesEmbed(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: newKeywordEmbedder(64)}
	s := New(emb, Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1})
	doc := "turbograph is a compact graph retrieval system with one binary"
	if err := s.Build(ctx, []Document{{ID: "d1", Text: doc}}); err != nil {
		t.Fatal(err)
	}

	base := atomic.LoadInt64(&emb.embedded)
	res1, err := s.Retrieve(ctx, "graph retrieval", RetrieveParams{TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(res1) == 0 {
		t.Fatal("empty first retrieval")
	}
	afterFirst := atomic.LoadInt64(&emb.embedded)
	if afterFirst != base+1 {
		t.Fatalf("first retrieve embedded %d texts, want 1 (query only)", afterFirst-base)
	}

	if _, err := s.Retrieve(ctx, "graph retrieval", RetrieveParams{TopK: 3}); err != nil {
		t.Fatal(err)
	}
	// Whitespace variant must hit the same normalized key.
	if _, err := s.Retrieve(ctx, "  graph   RETRIEVAL ", RetrieveParams{TopK: 3}); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt64(&emb.embedded); got != afterFirst {
		t.Fatalf("repeat query re-embedded %d texts, want 0 (cache hit)", got-afterFirst)
	}
}

// TestQueryCacheEncoderChangeMiss pins TZ O5: changing the encoder fingerprint
// invalidates the whole cache, because vectors in different spaces are incomparable.
func TestQueryCacheEncoderChangeMiss(t *testing.T) {
	qc := NewQueryCache(0, 0, "model-a")
	qc.Put("hello world", []float32{1, 2, 3})
	if _, ok := qc.EmbedIfCached("hello world"); !ok {
		t.Fatal("expected cache hit before encoder change")
	}
	qc.SetEncoder("model-b")
	if _, ok := qc.EmbedIfCached("hello world"); ok {
		t.Fatal("cache returned a hit after encoder change")
	}
}

// TestQueryCacheEntryCap verifies that the number-of-entries cap evicts the least
// recently used entry (back of the queue).
func TestQueryCacheEntryCap(t *testing.T) {
	qc := NewQueryCache(2, 0, "m")
	qc.Put("one", []float32{1})
	qc.Put("two", []float32{2})
	qc.Put("three", []float32{3})
	if _, ok := qc.EmbedIfCached("one"); ok {
		t.Fatal("oldest entry survived an over-cap insert")
	}
	if _, ok := qc.EmbedIfCached("two"); !ok {
		t.Fatal("second entry unexpectedly evicted")
	}
	if _, ok := qc.EmbedIfCached("three"); !ok {
		t.Fatal("most recent entry unexpectedly evicted")
	}
}

// TestQueryCacheByteCap verifies the byte cap evicts until the stored bytes fit.
func TestQueryCacheByteCap(t *testing.T) {
	qc := NewQueryCache(0, 5, "m") // 3 floats per vector; two vectors overflow
	qc.Put("a", []float32{1, 1, 1})
	qc.Put("b", []float32{2, 2, 2})
	qc.Put("c", []float32{3, 3, 3})
	if _, ok := qc.EmbedIfCached("a"); ok {
		t.Fatal("byte cap did not evict oldest entry")
	}
	if _, ok := qc.EmbedIfCached("c"); !ok {
		t.Fatal("most recent entry unexpectedly evicted")
	}
	if qc.bytes > qc.maxB {
		t.Fatalf("bytes %d exceeds cap %d", qc.bytes, qc.maxB)
	}
}

// TestQueryCacheNormalizeSkipsHuge pins the key normalization: huge queries are
// not cached at all, and the key is case/whitespace-insensitive.
func TestQueryCacheNormalizeSkipsHuge(t *testing.T) {
	if got := normalizeQuery("  Graph\tRETRIEVAL "); got != "graph retrieval" {
		t.Fatalf("normalizeQuery = %q", got)
	}
	big := make([]byte, 600)
	for i := range big {
		big[i] = 'a'
	}
	if got := normalizeQuery(string(big)); got != "" {
		t.Fatalf("huge query should normalize to empty, got len %d", len(got))
	}
	if got := normalizeQuery("   "); got != "" {
		t.Fatalf("blank query should normalize to empty, got %q", got)
	}
}

// TestQueryCachePRFDoesNotCorrupt pins that the returned vector is a copy: PRF
// mutates the query vector in place, and a later retrieve of the same query must
// still see the pristine cached value.
func TestQueryCachePRFDoesNotCorrupt(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: newKeywordEmbedder(96)}
	s := New(emb, Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.1})
	docs := make([]Document, 6)
	for i := range docs {
		docs[i] = Document{ID: string(rune('a' + i)), Text: "alpha beta gamma delta epsilon zeta eta theta"}
	}
	if err := s.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	p := RetrieveParams{TopK: 4, PRF: 1, PRFWeight: 0.3}
	r1, err := s.Retrieve(ctx, "alpha beta", p)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := s.Retrieve(ctx, "alpha beta", p)
	if err != nil {
		t.Fatal(err)
	}
	if len(r1) != len(r2) {
		t.Fatalf("PRF run changed result size: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].Chunk.DocID != r2[i].Chunk.DocID {
			t.Fatalf("rank %d doc differs: %s vs %s", i, r1[i].Chunk.DocID, r2[i].Chunk.DocID)
		}
	}
}
