package rag

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func saveTo(st *Store, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return st.Save(f)
}

func loadFrom(t *testing.T, path string) *Store {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	st, err := Load(newKeywordEmbedder(96), f)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func feed(docs []Document) <-chan Document {
	ch := make(chan Document)
	go func() {
		defer close(ch)
		for _, d := range docs {
			ch <- d
		}
	}()
	return ch
}

func genDocs(n int) []Document {
	docs := make([]Document, n)
	for i := range docs {
		docs[i] = Document{
			ID:   fmt.Sprintf("doc%d", i),
			Text: fmt.Sprintf("topic%d alpha beta shared content number %d about subject %d", i%7, i, i%7),
		}
	}
	return docs
}

func newIngestStore() *Store {
	return New(newKeywordEmbedder(96), Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.1,
		Chunk: ChunkConfig{TargetWords: 200}})
}

func TestIngestBasic(t *testing.T) {
	st := newIngestStore()
	docs := genDocs(50)
	prog, err := st.Ingest(context.Background(), feed(docs), len(docs), IngestOptions{Workers: 8})
	if err != nil {
		t.Fatal(err)
	}
	if prog.Done != 50 || prog.Failed != 0 {
		t.Fatalf("unexpected progress: %+v", prog)
	}
	if st.DocCount() != 50 {
		t.Errorf("doc count = %d, want 50", st.DocCount())
	}
	if st.Len() < 50 {
		t.Errorf("chunk count too low: %d", st.Len())
	}
	// Graph was reindexed exactly once at the end and is queryable.
	res, err := st.Retrieve(context.Background(), "topic3 subject", RetrieveParams{TopK: 3})
	if err != nil || len(res) == 0 {
		t.Fatalf("retrieve after ingest failed: %v", err)
	}
}

func TestIngestDedup(t *testing.T) {
	st := newIngestStore()
	docs := genDocs(30)
	if _, err := st.Ingest(context.Background(), feed(docs), len(docs), IngestOptions{Workers: 4}); err != nil {
		t.Fatal(err)
	}
	// Re-ingesting the same documents must skip all of them, adding no chunks.
	chunksBefore := st.Len()
	prog, err := st.Ingest(context.Background(), feed(docs), len(docs), IngestOptions{Workers: 4})
	if err != nil {
		t.Fatal(err)
	}
	if prog.Skipped != 30 || prog.Done != 0 {
		t.Errorf("expected all skipped, got %+v", prog)
	}
	if st.Len() != chunksBefore {
		t.Errorf("dedup failed: chunks grew from %d to %d", chunksBefore, st.Len())
	}
}

func TestIngestResumeAfterCrash(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "ingest.journal")
	spath := filepath.Join(dir, "store.tg")
	docs := genDocs(40)

	// First run: ingest only the first 20 docs, checkpointing to disk. This stands
	// in for a crash partway through.
	st1 := newIngestStore()
	j1, _ := OpenJournal(jpath)
	save1 := func() error { return saveTo(st1, spath) }
	if _, err := st1.Ingest(context.Background(), feed(docs[:20]), 40,
		IngestOptions{Workers: 4, Journal: j1, Save: save1, CheckpointEvery: 5}); err != nil {
		t.Fatal(err)
	}
	j1.Close()
	if j1.DoneCount() != 20 {
		t.Fatalf("journal done = %d, want 20", j1.DoneCount())
	}

	// Resume: load the checkpointed store and the journal, feed ALL 40 docs. The
	// first 20 must be skipped, the rest ingested, with no duplicates.
	st2 := loadFrom(t, spath)
	if st2.DocCount() != 20 {
		t.Fatalf("loaded store has %d docs, want 20", st2.DocCount())
	}
	j2, _ := OpenJournal(jpath)
	defer j2.Close()
	prog, err := st2.Ingest(context.Background(), feed(docs), 40,
		IngestOptions{Workers: 4, Journal: j2, Save: func() error { return saveTo(st2, spath) }, CheckpointEvery: 5})
	if err != nil {
		t.Fatal(err)
	}
	if prog.Skipped != 20 || prog.Done != 20 {
		t.Errorf("resume progress wrong: %+v", prog)
	}
	if st2.DocCount() != 40 {
		t.Errorf("final doc count = %d, want 40", st2.DocCount())
	}
}

// failEmbedder wraps the keyword embedder and errors whenever a chunk text
// contains a poison marker, to exercise per-document error tolerance.
type failEmbedder struct {
	inner *keywordEmbedder
	calls int64
}

func (e *failEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	atomic.AddInt64(&e.calls, 1)
	for _, t := range texts {
		if strings.Contains(t, "POISON") {
			return nil, fmt.Errorf("simulated embed failure")
		}
	}
	return e.inner.Embed(ctx, texts)
}

func TestIngestErrorTolerance(t *testing.T) {
	emb := &failEmbedder{inner: newKeywordEmbedder(96)}
	st := New(emb, Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.1, Chunk: ChunkConfig{TargetWords: 200}})
	docs := genDocs(20)
	docs[3].Text = "POISON broken document"
	docs[11].Text = "POISON another broken one"

	prog, err := st.Ingest(context.Background(), feed(docs), len(docs), IngestOptions{Workers: 6})
	if err != nil {
		t.Fatal(err)
	}
	if prog.Failed != 2 {
		t.Errorf("expected 2 failures, got %d (%+v)", prog.Failed, prog)
	}
	if prog.Done != 18 {
		t.Errorf("expected 18 done, got %d", prog.Done)
	}
	if st.DocCount() != 18 {
		t.Errorf("store should hold only the 18 good docs, has %d", st.DocCount())
	}
}

func TestIngestParallelDeterministic(t *testing.T) {
	docs := genDocs(80)
	run := func(workers int) int {
		st := newIngestStore()
		p, err := st.Ingest(context.Background(), feed(docs), len(docs), IngestOptions{Workers: workers})
		if err != nil {
			t.Fatal(err)
		}
		return p.Done + st.Len()
	}
	a, b := run(1), run(16)
	if a != b {
		t.Errorf("result differs by worker count: serial=%d parallel=%d", a, b)
	}
}

func TestIngestCancel(t *testing.T) {
	st := newIngestStore()
	ctx, cancel := context.WithCancel(context.Background())
	// Feed slowly and cancel after a few documents.
	ch := make(chan Document)
	go func() {
		defer close(ch)
		for i := 0; i < 100; i++ {
			select {
			case <-ctx.Done():
				return
			case ch <- Document{ID: fmt.Sprintf("d%d", i), Text: fmt.Sprintf("alpha beta gamma doc %d", i)}:
				if i == 10 {
					cancel()
				}
			}
		}
	}()
	prog, err := st.Ingest(ctx, ch, 100, IngestOptions{Workers: 2})
	if err == nil {
		t.Error("expected ctx error on cancel")
	}
	// Whatever was ingested before cancellation must be consistent and queryable.
	if st.DocCount() != prog.Done {
		t.Errorf("doc count %d != done %d after cancel", st.DocCount(), prog.Done)
	}
}
