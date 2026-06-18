package rag

import (
	"context"
	"runtime"
	"sync"
)

// Progress reports the running state of a bulk ingestion.
type Progress struct {
	Total   int // documents offered, if known (0 means unknown/streaming)
	Done    int // documents durably ingested this run
	Failed  int // documents that errored
	Skipped int // documents already present (resumed)
	Chunks  int // chunks added this run
}

// IngestOptions configures a bulk ingestion.
type IngestOptions struct {
	// Workers is how many documents are embedded concurrently. Embedding is the
	// bottleneck, so this is the main parallelism knob. Defaults to GOMAXPROCS.
	Workers int
	// Journal, if set, records durably-ingested documents so an interrupted run
	// resumes without re-embedding completed work.
	Journal *Journal
	// Save, if set, checkpoints the store to durable storage. It is called every
	// CheckpointEvery documents and once at the end, always before the matching
	// journal entries are written so a "done" record always implies a saved store.
	Save func() error
	// CheckpointEvery bounds how much embedding work a crash can lose. 0 disables
	// intermediate checkpoints (only a final save). Ignored if Save is nil.
	CheckpointEvery int
	// OnProgress, if set, is called after each document with a snapshot.
	OnProgress func(Progress)
}

type ingestResult struct {
	id      string
	prep    prepared
	err     error
	skipped bool
}

// Ingest indexes a stream of documents with bounded parallelism, error
// tolerance, resume support, and periodic checkpointing. Embedding runs across
// Workers goroutines off the write lock; indexing is serialized; the graph is
// rebuilt once at the end. Cancelling ctx stops the run cleanly after
// checkpointing what has completed, and returns ctx.Err().
//
// total is the document count if known (for progress display); pass 0 when
// streaming an unknown number.
func (s *Store) Ingest(ctx context.Context, docs <-chan Document, total int, opt IngestOptions) (Progress, error) {
	if opt.Workers <= 0 {
		opt.Workers = runtime.GOMAXPROCS(0)
	}

	results := make(chan ingestResult, opt.Workers*2)
	var wg sync.WaitGroup
	wg.Add(opt.Workers)
	for w := 0; w < opt.Workers; w++ {
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case d, ok := <-docs:
					if !ok {
						return
					}
					results <- s.prepareForIngest(ctx, d, opt.Journal)
				}
			}
		}()
	}
	go func() { wg.Wait(); close(results) }()

	prog := Progress{Total: total}
	var pending []string // ingested but not yet checkpointed
	checkpoint := func() error {
		if opt.Save != nil {
			if err := opt.Save(); err != nil {
				return err
			}
		}
		if opt.Journal != nil && len(pending) > 0 {
			if err := opt.Journal.MarkDone(pending...); err != nil {
				return err
			}
		}
		pending = pending[:0]
		return nil
	}

	var loopErr error
	for r := range results {
		switch {
		case r.skipped:
			prog.Skipped++
		case r.err != nil:
			prog.Failed++
			if opt.Journal != nil {
				opt.Journal.MarkFailed(r.id, r.err.Error())
			}
		default:
			// applyPrepared installs the document (as an add or an update),
			// returning false if a concurrent duplicate won the race.
			if !s.applyPrepared(r.prep) {
				prog.Skipped++
				break
			}
			prog.Done++
			prog.Chunks += len(r.prep.chunks)
			pending = append(pending, r.id)
			if opt.Save != nil && opt.CheckpointEvery > 0 && len(pending) >= opt.CheckpointEvery {
				if err := checkpoint(); err != nil {
					loopErr = err
				}
			}
		}
		if opt.OnProgress != nil {
			opt.OnProgress(prog)
		}
	}

	// Final checkpoint and graph rebuild. Reindex is in-memory; the durable store
	// rebuilds its graph on load, so reindexing after the final save is fine.
	if err := checkpoint(); err != nil && loopErr == nil {
		loopErr = err
	}
	s.Reindex()

	if err := ctx.Err(); err != nil {
		return prog, err
	}
	return prog, loopErr
}

// prepareForIngest decides whether a document needs work and, if so, chunks and
// embeds it (reusing embeddings for unchanged chunks on an update). Documents
// already completed (per the journal), already present with identical content, or
// duplicating content under another id are skipped without embedding.
func (s *Store) prepareForIngest(ctx context.Context, d Document, j *Journal) ingestResult {
	h := contentHash(d.Text)
	if j != nil && j.Done(d.ID) {
		return ingestResult{id: d.ID, skipped: true}
	}
	if owner, ok := s.ContentOwner(h); ok && owner != d.ID {
		return ingestResult{id: d.ID, skipped: true} // identical content under another id
	}
	if cur, ok := s.idHashOf(d.ID); ok && cur == h {
		return ingestResult{id: d.ID, skipped: true} // unchanged
	}
	p, err := s.prepareDoc(ctx, d)
	if err != nil {
		return ingestResult{id: d.ID, err: err}
	}
	if len(p.chunks) == 0 {
		return ingestResult{id: d.ID, skipped: true}
	}
	return ingestResult{id: d.ID, prep: p}
}
