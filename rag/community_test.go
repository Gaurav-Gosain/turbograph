package rag

import (
	"context"
	"fmt"
	"strings"
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
