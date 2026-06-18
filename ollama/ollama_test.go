package ollama

import (
	"context"
	"testing"
	"time"
)

func liveClient(t *testing.T) *Client {
	t.Helper()
	c := New()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := c.Ping(ctx); err != nil {
		t.Skipf("ollama not reachable: %v", err)
	}
	return c
}

func TestEmbedLive(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	vecs, err := c.Embed(ctx, []string{"the quick brown fox", "a lazy dog sleeps"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 {
		t.Fatalf("want 2 embeddings, got %d", len(vecs))
	}
	if len(vecs[0]) == 0 || len(vecs[0]) != len(vecs[1]) {
		t.Fatalf("bad embedding dims: %d, %d", len(vecs[0]), len(vecs[1]))
	}
	t.Logf("embedding dim = %d", len(vecs[0]))
}
