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

// TestIngestDedupWithinDoc: with the flag on, repeated passages inside one document
// are embedded and indexed once; off, they are indexed every time.
func TestIngestDedupWithinDoc(t *testing.T) {
	sentence := "The quick brown fox jumps over the lazy dog near the river bank."
	text := strings.TrimSpace(strings.Repeat(sentence+" ", 24))
	cfg := Config{Seed: 1, Chunk: ChunkConfig{Strategy: StrategySentence, TargetWords: 2}}
	mk := func() *Store { return New(newKeywordEmbedder(96), cfg) }

	off := mk()
	_, err := off.Ingest(context.Background(), feed([]Document{{ID: "d", Text: text}}), 1,
		IngestOptions{Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	on := mk()
	progOn, err := on.Ingest(context.Background(), feed([]Document{{ID: "d", Text: text}}), 1,
		IngestOptions{Workers: 1, DedupWithinDoc: true})
	if err != nil {
		t.Fatal(err)
	}
	if progOn.Chunks != 1 {
		t.Errorf("dedup within doc: %d unique chunks, want 1 (the whole doc is one repeated sentence)", progOn.Chunks)
	}
	if on.Len() >= off.Len() || on.Len() != progOn.Chunks {
		t.Errorf("dedup left %d chunks vs %d without, want fewer (and equal to the embed count)", on.Len(), off.Len())
	}
	if progOn.ChunksEmbedded != progOn.Chunks {
		t.Errorf("dedup embedded %d chunks for %d stored, want them equal (duplicates neither stored nor embedded)",
			progOn.ChunksEmbedded, progOn.Chunks)
	}
}

// TestIngestChunksEmbeddedReuseVectors: ChunksEmbedded equals Chunks on a first
// ingest, but an update that keeps some chunk text reuses its embeddings, so only
// the changed chunk is embedded.
func TestIngestChunksEmbeddedReuseVectors(t *testing.T) {
	st := New(newKeywordEmbedder(96), Config{Seed: 1,
		Chunk: ChunkConfig{Strategy: StrategySentence, TargetWords: 8}})
	mkText := func(tail string) string {
		var b strings.Builder
		for i := 1; i <= 30; i++ {
			fmt.Fprintf(&b, "Sentence number %d about the topic with some detail words. ", i)
			if i == 30 {
				b.WriteString(tail + " ")
			}
		}
		return b.String()
	}
	first := mkText("")
	prog1, err := st.Ingest(context.Background(), feed([]Document{{ID: "d", Text: first}}), 1, IngestOptions{Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	if prog1.ChunksEmbedded != prog1.Chunks || prog1.Chunks == 0 {
		t.Fatalf("first ingest should embed every chunk: embedded=%d chunks=%d", prog1.ChunksEmbedded, prog1.Chunks)
	}
	// Change only the last sentence: every earlier chunk reuses its vector.
	updated := mkText("This replacement sentence only now appears here.")
	prog2, err := st.Ingest(context.Background(), feed([]Document{{ID: "d", Text: updated}}), 1, IngestOptions{Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	if prog2.Chunks < 1 || prog2.ChunksEmbedded < 1 {
		t.Fatalf("update should add and embed the changed tail chunk, got %+v", prog2)
	}
	if prog2.ChunksEmbedded >= prog2.Chunks {
		t.Errorf("update embedded %d of %d chunks, want < (earlier chunks reused)", prog2.ChunksEmbedded, prog2.Chunks)
	}
}

// TestIngestConfigWorkersDefault: Workers:0 falls back to Config.IngestWorkers, and
// the result is deterministic (same docs, same count as an explicit single-worker run).
func TestIngestConfigWorkersDefault(t *testing.T) {
	docs := genDocs(40)
	run := func(opt IngestOptions) (int, int) {
		st := New(newKeywordEmbedder(96), Config{Seed: 1, IngestWorkers: 3,
			Chunk: ChunkConfig{TargetWords: 200}})
		p, err := st.Ingest(context.Background(), feed(docs), len(docs), opt)
		if err != nil {
			t.Fatal(err)
		}
		return p.Done, st.Len()
	}
	doneCfg, chunksCfg := run(IngestOptions{}) // Workers 0 -> Config.IngestWorkers
	if doneCfg != 40 {
		t.Errorf("config-workers run ingested %d docs, want 40", doneCfg)
	}
	doneSer, chunksSer := run(IngestOptions{Workers: 1}) // explicit serial baseline
	if doneCfg != doneSer || chunksCfg != chunksSer {
		t.Errorf("config-workers (%d/%d) diverged from serial (%d/%d)",
			doneCfg, chunksCfg, doneSer, chunksSer)
	}
}
