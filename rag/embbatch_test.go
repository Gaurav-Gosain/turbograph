package rag

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// recordingEmbedder records the size of every Embed call and returns one
// vector per text whose first coordinate encodes the text identity
// ("text-<n>" -> n), so tests can assert a strict text<->vector
// correspondence that survives batch boundaries.
type recordingEmbedder struct {
	sizes []int // texts in each Embed call, in call order
	errAt int   // if >= 0, the errAt-th Embed call (0-based) fails
}

func (e *recordingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if e.errAt >= 0 && len(e.sizes) == e.errAt {
		return nil, errors.New("embed failed")
	}
	e.sizes = append(e.sizes, len(texts))
	out := make([][]float32, len(texts))
	for i, t := range texts {
		var n int
		if _, err := fmt.Sscanf(t, "text-%d", &n); err != nil {
			n = -1
		}
		out[i] = []float32{float32(n)}
	}
	return out, nil
}

func textsN(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("text-%d", i)
	}
	return out
}

func TestEmbedBatchedSplitsLargeRequests(t *testing.T) {
	rec := &recordingEmbedder{errAt: -1}
	s := &Store{embedder: rec, cfg: Config{EmbedBatchSize: 64}}
	vecs, err := s.embedBatched(context.Background(), textsN(250))
	if err != nil {
		t.Fatalf("embedBatched: %v", err)
	}
	want := []int{64, 64, 64, 58}
	if len(rec.sizes) != len(want) {
		t.Fatalf("got %d embed calls %v, want %d", len(rec.sizes), rec.sizes, len(want))
	}
	for i, n := range want {
		if rec.sizes[i] != n {
			t.Errorf("call %d had %d texts, want %d", i, rec.sizes[i], n)
		}
	}
	if len(vecs) != 250 {
		t.Fatalf("got %d vectors, want 250", len(vecs))
	}
	for i, v := range vecs {
		if v[0] != float32(i) {
			t.Fatalf("vector %d encodes text %v, order not preserved", i, v[0])
		}
	}
}

func TestEmbedBatchedPassesThroughUnderBatch(t *testing.T) {
	rec := &recordingEmbedder{errAt: -1}
	s := &Store{embedder: rec, cfg: Config{EmbedBatchSize: 64}}
	if _, err := s.embedBatched(context.Background(), textsN(50)); err != nil {
		t.Fatalf("embedBatched: %v", err)
	}
	if len(rec.sizes) != 1 || rec.sizes[0] != 50 {
		t.Fatalf("expected one call with 50 texts, got %v", rec.sizes)
	}
}

func TestEmbedBatchedBatchOne(t *testing.T) {
	rec := &recordingEmbedder{errAt: -1}
	s := &Store{embedder: rec, cfg: Config{EmbedBatchSize: 1}}
	vecs, err := s.embedBatched(context.Background(), textsN(3))
	if err != nil {
		t.Fatalf("embedBatched: %v", err)
	}
	if len(rec.sizes) != 3 {
		t.Fatalf("got %d calls, want 3", len(rec.sizes))
	}
	for _, n := range rec.sizes {
		if n != 1 {
			t.Errorf("call size %d, want 1", n)
		}
	}
	if len(vecs) != 3 || vecs[2][0] != 2 {
		t.Fatalf("order not preserved: %d vectors", len(vecs))
	}
}

func TestConfigWithDefaultsSetsEmbedBatch(t *testing.T) {
	c := Config{}
	c.withDefaults()
	if c.EmbedBatchSize != 64 {
		t.Fatalf("default EmbedBatchSize = %d, want 64", c.EmbedBatchSize)
	}
}

func TestEmbedBatchedStopsOnError(t *testing.T) {
	rec := &recordingEmbedder{errAt: 1}
	s := &Store{embedder: rec, cfg: Config{EmbedBatchSize: 2}}
	if _, err := s.embedBatched(context.Background(), textsN(5)); err == nil {
		t.Fatal("expected an error from the second batch")
	}
	// First batch (2 texts) succeeded, second failed, so exactly one call is
	// recorded; the third batch was never reached.
	if len(rec.sizes) != 1 || rec.sizes[0] != 2 {
		t.Fatalf("expected only the first batch (2 texts) to be recorded, got %v", rec.sizes)
	}
}

func TestEmbedBatchedEmptyInput(t *testing.T) {
	for _, in := range [][]string{nil, {}} {
		rec := &recordingEmbedder{errAt: -1}
		s := &Store{embedder: rec, cfg: Config{EmbedBatchSize: 64}}
		vecs, err := s.embedBatched(context.Background(), in)
		if err != nil || vecs != nil {
			t.Fatalf("empty input: got (%v, %v), want (nil, nil)", vecs, err)
		}
		if len(rec.sizes) != 0 {
			t.Fatalf("embedder called on empty input: %v", rec.sizes)
		}
	}
}

func TestEmbedBatchedKeepsQueryAsymmetry(t *testing.T) {
	rec := &recordingEmbedder{errAt: -1}
	vecs, err := embedQuery(context.Background(), rec, []string{"a", "b"})
	if err != nil {
		t.Fatalf("embedQuery: %v", err)
	}
	if len(rec.sizes) != 1 || rec.sizes[0] != 2 || len(vecs) != 2 {
		t.Fatalf("plain embedder: query went through Embed, got sizes %v", rec.sizes)
	}
}
