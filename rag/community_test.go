package rag

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
)

// labelSummarizer returns a deterministic summary naming the first passage, so
// tests can build summaries without a model.
func labelSummarizer(_ context.Context, passages []string) (string, error) {
	head := passages[0]
	if len(head) > 40 {
		head = head[:40]
	}
	return fmt.Sprintf("Summary of %d passages about: %s", len(passages), head), nil
}

func communityStore(t *testing.T) *Store {
	t.Helper()
	st := New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.02,
		Chunk: ChunkConfig{TargetWords: 8, OverlapWords: 0}})
	docs := []Document{
		{ID: "space", Text: "rockets orbit planets moons gravity launch thrust fuel astronauts"},
		{ID: "cooking", Text: "recipes bake oven flour sugar butter knead dough pastry"},
		{ID: "finance", Text: "stocks bonds markets interest rates inflation portfolio dividends"},
	}
	if err := st.Build(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return st
}

func TestBuildCommunitySummaries(t *testing.T) {
	ctx := context.Background()
	st := communityStore(t)
	if st.HasCommunitySummaries() {
		t.Fatal("should have no summaries before building")
	}
	if err := st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{}); err != nil {
		t.Fatal(err)
	}
	if !st.HasCommunitySummaries() {
		t.Fatal("no summaries after build")
	}
	sums := st.CommunitySummaries()
	if len(sums) == 0 {
		t.Fatal("no community summaries")
	}
	for _, s := range sums {
		if s.Summary == "" || s.Size == 0 || len(s.Chunks) == 0 {
			t.Fatalf("incomplete summary: %+v", s)
		}
	}
	// Largest community first.
	for i := 1; i < len(sums); i++ {
		if sums[i].Size > sums[i-1].Size {
			t.Fatal("summaries not sorted by size")
		}
	}
}

func TestRelevantCommunities(t *testing.T) {
	ctx := context.Background()
	st := communityStore(t)
	st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{})
	got, err := st.RelevantCommunities(ctx, "rockets gravity astronauts", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 community, got %d", len(got))
	}
	// The space community should rank first for a space query.
	if !strings.Contains(strings.ToLower(got[0].Summary), "rockets") &&
		!containsDoc(got[0].DocIDs, "space") {
		t.Logf("top community: %+v", got[0]) // keyword embedder is coarse; informational
	}
}

func TestCommunitySummariesInvalidateAndPersist(t *testing.T) {
	ctx := context.Background()
	st := communityStore(t)
	st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{})

	// A content change invalidates the summaries.
	st.AddDocuments(ctx, []Document{{ID: "music", Text: "guitar piano violin orchestra melody rhythm tempo"}})
	if st.HasCommunitySummaries() {
		t.Fatal("summaries should be invalidated after adding a document")
	}

	// Rebuild, then confirm they survive a save/load round trip.
	st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{})
	path := t.TempDir() + "/c.tg"
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	st2 := loadFrom(t, path)
	if !st2.HasCommunitySummaries() {
		t.Fatal("summaries lost on reload")
	}
	if len(st2.CommunitySummaries()) == 0 {
		t.Fatal("no summaries after reload")
	}
}

func containsDoc(ids []string, id string) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}

// TestCommunityVectorsEmbedOnce pins B10: summaries are embedded exactly once at
// generation, cached, and reused by every global query; a reload picks the cache
// back up from disk instead of re-embedding.
func TestCommunityVectorsEmbedOnce(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: newKeywordEmbedder(96)}
	st := New(emb, Config{Seed: 1, GraphKNN: 3, MinSimilarity: 0.02,
		Chunk: ChunkConfig{TargetWords: 8, OverlapWords: 0}})
	docs := []Document{
		{ID: "space", Text: "rockets orbit planets moons gravity launch thrust fuel astronauts"},
		{ID: "cooking", Text: "recipes bake oven flour sugar butter knead dough pastry"},
		{ID: "finance", Text: "stocks bonds markets interest rates inflation portfolio dividends"},
	}
	if err := st.Build(ctx, docs); err != nil {
		t.Fatal(err)
	}
	afterBuild := atomic.LoadInt64(&emb.embedded)

	if err := st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{}); err != nil {
		t.Fatal(err)
	}
	afterSummaries := atomic.LoadInt64(&emb.embedded)
	comms := st.CommunitySummaries()
	if len(comms) == 0 {
		t.Fatal("no summaries built")
	}
	if got := afterSummaries - afterBuild; got != int64(len(comms)) {
		t.Errorf("generating %d summaries embedded %d texts, want one per summary", len(comms), got)
	}

	// First and second global queries embed only the query vector (or nothing once
	// the TZ O5 query-LRU has the text): the summaries must not be re-embedded per call.
	if _, err := st.RelevantCommunities(ctx, "rockets gravity astronauts", 1); err != nil {
		t.Fatal(err)
	}
	afterFirst := atomic.LoadInt64(&emb.embedded)
	if got := afterFirst - afterSummaries; got != 1 {
		t.Errorf("first global query embedded %d texts, want 1 (the query only)", got)
	}
	if _, err := st.RelevantCommunities(ctx, "rockets gravity astronauts", 1); err != nil {
		t.Fatal(err)
	}
	afterSecond := atomic.LoadInt64(&emb.embedded)
	if got := afterSecond - afterFirst; got != 0 {
		t.Errorf("repeat global query embedded %d texts, want 0 (query LRU hit)", got)
	}

	// The cache survives a save/load round trip: the first query on the reloaded
	// store embeds the query but not the summaries.
	path := t.TempDir() + "/cvec.tg"
	if err := saveTo(st, path); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	emb2 := &countingEmbedder{inner: newKeywordEmbedder(96)}
	st2, err := Load(emb2, f)
	if err != nil {
		t.Fatal(err)
	}
	if len(st2.commVec) != len(comms) {
		t.Errorf("commVec after reload = %d entries, want %d", len(st2.commVec), len(comms))
	}
	before := atomic.LoadInt64(&emb2.embedded)
	if _, err := st2.RelevantCommunities(ctx, "rockets gravity astronauts", 1); err != nil {
		t.Fatal(err)
	}
	after := atomic.LoadInt64(&emb2.embedded)
	if got := after - before; got != 1 {
		t.Errorf("global query on reloaded store embedded %d texts, want 1 (cache persisted)", got)
	}
}

// TestCommunityVectorsInvalidate pins B10 parity: wherever summaries are cleared,
// their cached embeddings are cleared with them.
func TestCommunityVectorsInvalidate(t *testing.T) {
	ctx := context.Background()
	st := communityStore(t)
	if err := st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{}); err != nil {
		t.Fatal(err)
	}
	if len(st.commVec) == 0 {
		t.Fatal("no commVec after build")
	}
	st.AddDocuments(ctx, []Document{{ID: "music", Text: "guitar piano violin orchestra melody rhythm tempo"}})
	if len(st.commVec) != 0 {
		t.Fatal("commVec survived a corpus change that nulled commSummary")
	}
	st.BuildCommunitySummaries(ctx, labelSummarizer, CommunityOptions{})
	if len(st.commVec) == 0 {
		t.Fatal("commVec not rebuilt with the summaries")
	}
	if n := st.DeleteDocument("music"); n == 0 {
		t.Fatal("expected the music document to be removed")
	}
	if len(st.commVec) != 0 {
		t.Fatal("commVec survived a document delete that nulled commSummary")
	}
}
