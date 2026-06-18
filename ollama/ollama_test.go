package ollama

import (
	"context"
	"strings"
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
	// Skip rather than fail if the default embedding model is not pulled on this
	// machine; pulling models is an operator action, not a test prerequisite.
	models, err := c.ListModels(ctx)
	if err != nil {
		t.Skipf("cannot list models: %v", err)
	}
	found := false
	for _, m := range models {
		if m == c.EmbedModel || strings.HasPrefix(m, c.EmbedModel+":") {
			found = true
		}
	}
	if !found {
		t.Skipf("embedding model %q not installed; run: ollama pull %s", c.EmbedModel, c.EmbedModel)
	}
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
