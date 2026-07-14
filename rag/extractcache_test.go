package rag

import (
	"bytes"
	"context"
	"sync/atomic"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/entity"
)

// countingExtractor records how many chunks actually reached the "model".
type countingExtractor struct{ calls atomic.Int64 }

func (c *countingExtractor) Extract(_ context.Context, text string) (entity.Extraction, error) {
	c.calls.Add(1)
	// One entity named after the text, so each chunk contributes a distinct node.
	return entity.Extraction{
		Entities:  []entity.ExtractedEntity{{Name: "e-" + text, Type: "concept", Description: "d"}},
		Relations: []entity.ExtractedRelation{{Source: "e-" + text, Target: "hub", Description: "in"}},
	}, nil
}

func buildStore(t *testing.T, texts ...string) *Store {
	t.Helper()
	s := New(newKeywordEmbedder(32), Config{Seed: 1})
	docs := make([]Document, len(texts))
	for i, tx := range texts {
		docs[i] = Document{ID: tx, Text: tx}
	}
	if err := s.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return s
}

// TestExtractCacheSkipsSeenChunks is the point of the cache: a rebuild must not
// re-ask the model about a chunk it has already read.
func TestExtractCacheSkipsSeenChunks(t *testing.T) {
	s := buildStore(t, "alpha", "beta", "gamma")
	ex := &countingExtractor{}
	opt := EntityBuildOptions{Model: "m1", Workers: 2}

	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	first := ex.calls.Load()
	if first != 3 {
		t.Fatalf("first build should extract all 3 chunks, got %d", first)
	}
	if got := s.CachedExtractions(); got != 3 {
		t.Fatalf("cache should hold 3 extractions, got %d", got)
	}
	ents, rels := s.EntityCount(), s.RelationCount()

	// A rebuild with nothing changed must reach the model zero times.
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if extra := ex.calls.Load() - first; extra != 0 {
		t.Errorf("rebuild made %d model calls; the cache should have answered all of them", extra)
	}
	// And it must produce the same graph, not an empty one.
	if s.EntityCount() != ents || s.RelationCount() != rels {
		t.Errorf("cached rebuild changed the graph: %d/%d entities/relations, want %d/%d",
			s.EntityCount(), s.RelationCount(), ents, rels)
	}
}

// TestExtractCacheOnlyExtractsNewChunks: the case that motivated the cache. Adding
// one document to an existing corpus must cost one extraction, not len(corpus).
func TestExtractCacheOnlyExtractsNewChunks(t *testing.T) {
	s := buildStore(t, "alpha", "beta", "gamma")
	ex := &countingExtractor{}
	opt := EntityBuildOptions{Model: "m1"}
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	before := ex.calls.Load()

	if err := s.AddDocuments(context.Background(), []Document{{ID: "delta", Text: "delta"}}); err != nil {
		t.Fatal(err)
	}
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if extra := ex.calls.Load() - before; extra != 1 {
		t.Errorf("adding one document cost %d extractions, want 1", extra)
	}
	if s.EntityCount() < 4 {
		t.Errorf("the new document's entity is missing: %d entities", s.EntityCount())
	}
}

// TestExtractCacheKeyedByModel: what a small model said is not what a different
// model would say, so switching models must re-extract.
func TestExtractCacheKeyedByModel(t *testing.T) {
	s := buildStore(t, "alpha", "beta")
	ex := &countingExtractor{}
	if err := s.BuildEntityGraph(context.Background(), ex, EntityBuildOptions{Model: "m1"}); err != nil {
		t.Fatal(err)
	}
	before := ex.calls.Load()
	if err := s.BuildEntityGraph(context.Background(), ex, EntityBuildOptions{Model: "m2"}); err != nil {
		t.Fatal(err)
	}
	if extra := ex.calls.Load() - before; extra != 2 {
		t.Errorf("switching model made %d calls, want 2 (the cache must not be reused across models)", extra)
	}
}

// TestExtractCacheRefresh bypasses the cache on demand.
func TestExtractCacheRefresh(t *testing.T) {
	s := buildStore(t, "alpha", "beta")
	ex := &countingExtractor{}
	opt := EntityBuildOptions{Model: "m1"}
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	before := ex.calls.Load()
	opt.Refresh = true
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if extra := ex.calls.Load() - before; extra != 2 {
		t.Errorf("refresh made %d calls, want 2", extra)
	}
}

// TestExtractCacheSurvivesSaveLoad: the cache is worthless if reopening the store
// throws it away, since that is exactly when a rebuild is most likely.
func TestExtractCacheSurvivesSaveLoad(t *testing.T) {
	s := buildStore(t, "alpha", "beta")
	ex := &countingExtractor{}
	opt := EntityBuildOptions{Model: "m1"}
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(newKeywordEmbedder(32), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if got := loaded.CachedExtractions(); got != 2 {
		t.Fatalf("cache did not survive save/load: %d entries", got)
	}
	ex2 := &countingExtractor{}
	if err := loaded.BuildEntityGraph(context.Background(), ex2, opt); err != nil {
		t.Fatal(err)
	}
	if n := ex2.calls.Load(); n != 0 {
		t.Errorf("rebuild after reload made %d model calls, want 0", n)
	}
	// Counting the calls is not enough, and an earlier version of this test that stopped
	// there passed while the feature was broken: the entries persisted as empty
	// extractions, so every chunk "hit" a cache of nothing and the graph came back empty.
	// What matters is that the reloaded cache rebuilds the same graph.
	if got, want := loaded.EntityCount(), s.EntityCount(); got != want {
		t.Errorf("reloaded cache rebuilt %d entities, want %d", got, want)
	}
	if got, want := loaded.RelationCount(), s.RelationCount(); got != want {
		t.Errorf("reloaded cache rebuilt %d relations, want %d", got, want)
	}
	if loaded.EntityCount() == 0 {
		t.Error("the reloaded graph is empty")
	}
}

// TestExtractCacheDropsDeadChunks: an edited document's old chunks must not keep
// their entries forever, or the cache grows without bound and is persisted that way.
func TestExtractCacheDropsDeadChunks(t *testing.T) {
	s := buildStore(t, "alpha", "beta")
	ex := &countingExtractor{}
	opt := EntityBuildOptions{Model: "m1"}
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if got := s.CachedExtractions(); got != 2 {
		t.Fatalf("want 2 cached, got %d", got)
	}
	// Replace beta's content: its old chunk text leaves the corpus.
	if err := s.AddDocuments(context.Background(), []Document{{ID: "beta", Text: "beta rewritten"}}); err != nil {
		t.Fatal(err)
	}
	if err := s.BuildEntityGraph(context.Background(), ex, opt); err != nil {
		t.Fatal(err)
	}
	if got := s.CachedExtractions(); got != 2 {
		t.Errorf("cache should hold one entry per live chunk (2), got %d", got)
	}
}
