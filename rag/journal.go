package rag

import (
	"bufio"
	"encoding/json"
	"os"
	"sync"
)

// Journal is an append-only record of which documents have been durably ingested.
// It lets an interrupted ingestion resume without re-embedding work that is
// already saved: a document is marked done only after the store containing it has
// been checkpointed to disk, so a "done" entry always implies the document is
// recoverable. Failed documents are recorded for visibility but are retried on
// the next run.
type Journal struct {
	mu   sync.Mutex
	f    *os.File
	w    *bufio.Writer
	done map[string]struct{}
}

type journalEntry struct {
	ID  string `json:"id"`
	OK  bool   `json:"ok"`
	Err string `json:"err,omitempty"`
}

// OpenJournal opens (or creates) a journal at path, replaying existing entries to
// rebuild the set of completed documents.
func OpenJournal(path string) (*Journal, error) {
	done := make(map[string]struct{})
	if f, err := os.Open(path); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			var e journalEntry
			if json.Unmarshal(sc.Bytes(), &e) == nil && e.OK {
				done[e.ID] = struct{}{}
			}
		}
		f.Close()
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	return &Journal{f: f, w: bufio.NewWriter(f), done: done}, nil
}

// Done reports whether a document has been durably ingested.
func (j *Journal) Done(id string) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	_, ok := j.done[id]
	return ok
}

// DoneCount returns the number of completed documents.
func (j *Journal) DoneCount() int {
	j.mu.Lock()
	defer j.mu.Unlock()
	return len(j.done)
}

// MarkDone records documents as durably ingested and flushes to disk, so the
// record survives a crash.
func (j *Journal) MarkDone(ids ...string) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	for _, id := range ids {
		if _, ok := j.done[id]; ok {
			continue
		}
		b, _ := json.Marshal(journalEntry{ID: id, OK: true})
		j.w.Write(b)
		j.w.WriteByte('\n')
		j.done[id] = struct{}{}
	}
	if err := j.w.Flush(); err != nil {
		return err
	}
	return j.f.Sync()
}

// MarkFailed records that a document failed to ingest. It is informational; the
// document will be retried on the next run.
func (j *Journal) MarkFailed(id, errMsg string) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	b, _ := json.Marshal(journalEntry{ID: id, OK: false, Err: errMsg})
	j.w.Write(b)
	j.w.WriteByte('\n')
	return j.w.Flush()
}

// Close flushes and closes the journal.
func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.w.Flush()
	return j.f.Close()
}
