package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// atomicTestEmbedder is a tiny deterministic embedder for save/load round trips:
// a normalized hash of the text spread over the vector. Determinism is what matters
// here (the same text must produce the same vector so loads are stable), not quality.
type atomicTestEmbedder struct{}

func (atomicTestEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	const dim = 8
	out := make([][]float32, len(texts))
	for i, t := range texts {
		h := sha256.Sum256([]byte(t))
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = float32(binary.LittleEndian.Uint32(h[j*4:])) / float32(1<<32)
		}
		out[i] = v
	}
	return out, nil
}

func (atomicTestEmbedder) Fingerprint() string { return "atomic-test" }

// buildAtomicStore builds a tiny store with the test embedder and returns it.
func buildAtomicStore(t *testing.T, docs ...rag.Document) *rag.Store {
	t.Helper()
	st := rag.New(atomicTestEmbedder{}, rag.Config{})
	if err := st.Build(context.Background(), docs); err != nil {
		t.Fatalf("build: %v", err)
	}
	return st
}

// TestAtomicSave_LoadReadsNewDataAfterSave checks the round trip: a document added
// and re-saved is present after a fresh Load.
func TestAtomicSave_LoadReadsNewDataAfterSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "kb.tg")
	st := buildAtomicStore(t, rag.Document{ID: "a", Text: "alpha"})
	if err := saveStore(st, path, rag.VectorsExact); err != nil {
		t.Fatalf("save: %v", err)
	}

	st2 := buildAtomicStore(t, rag.Document{ID: "a", Text: "alpha"}, rag.Document{ID: "b", Text: "beta"})
	ctx := context.Background()
	if err := st2.AddDocuments(ctx, []rag.Document{{ID: "b", Text: "beta"}}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := saveStore(st2, path, rag.VectorsExact); err != nil {
		t.Fatalf("re-save: %v", err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	loaded, err := rag.Load(atomicTestEmbedder{}, f)
	if err != nil {
		t.Fatalf("load after save: %v", err)
	}
	if !loaded.HasDoc("b") {
		t.Fatal("Load after successful save does not see the new document")
	}
	if got, want := loaded.DocCount(), 2; got != want {
		t.Fatalf("DocCount = %d, want %d", got, want)
	}
}

// TestAtomicSave_OldStoreSurvivesFailedSave simulates an interrupted save: the save
// must fail without touching the previous good .tg. An unbuilt store errors inside
// SaveLean, which is the failure point a crash could hit mid-write.
func TestAtomicSave_OldStoreSurvivesFailedSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "kb.tg")
	st := buildAtomicStore(t, rag.Document{ID: "a", Text: "alpha"})
	if err := saveStore(st, path, rag.VectorsExact); err != nil {
		t.Fatalf("save: %v", err)
	}
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// SaveLean refuses to serialize an empty store (hnsw == nil), so this save
	// fails before the rename; a crash during a real save is the same shape.
	empty := rag.New(atomicTestEmbedder{}, rag.Config{})
	if err := saveStore(empty, path, rag.VectorsExact); err == nil {
		t.Fatal("expected save of an empty store to fail")
	}

	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(good, after) {
		t.Fatal("a failed save modified the previously good .tg; primary must be left untouched")
	}
	// And the primary still loads.
	if _, err := rag.Load(atomicTestEmbedder{}, bytes.NewReader(after)); err != nil {
		t.Fatalf("primary unreadable after failed save: %v", err)
	}
}

// TestAtomicSave_StrayTempLeavesPrimaryIntact simulates a crash between the temp
// write and the rename: a stray temp file appears next to the primary, and the
// primary must remain readable and byte-identical.
func TestAtomicSave_StrayTempLeavesPrimaryIntact(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kb.tg")
	st := buildAtomicStore(t, rag.Document{ID: "a", Text: "alpha"})
	if err := saveStore(st, path, rag.VectorsExact); err != nil {
		t.Fatalf("save: %v", err)
	}
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Simulate the interrupted write: same directory, same temp-name pattern, but
	// never renamed into place.
	tmp, err := os.CreateTemp(dir, ".tg-*")
	if err != nil {
		t.Fatal(err)
	}
	if err := st.SaveLean(tmp, rag.VectorsExact); err != nil {
		tmp.Close()
		t.Fatalf("temp save: %v", err)
	}
	if err := tmp.Close(); err != nil {
		t.Fatal(err)
	}

	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(good, after) {
		t.Fatal("a stray temp write changed the primary")
	}
	if _, err := rag.Load(atomicTestEmbedder{}, bytes.NewReader(after)); err != nil {
		t.Fatalf("primary unreadable: %v", err)
	}
}

// TestAtomicSave_BackupKeepsPreviousStore checks the keep-one .bak behaviour: after
// a second save, path+".bak" holds the previously published store.
func TestAtomicSave_BackupKeepsPreviousStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "kb.tg")
	st := buildAtomicStore(t, rag.Document{ID: "a", Text: "alpha"})
	if err := saveStoreOpt(st, path, rag.VectorsExact, true); err != nil {
		t.Fatalf("first save: %v", err)
	}

	st2 := buildAtomicStore(t, rag.Document{ID: "a", Text: "alpha"}, rag.Document{ID: "b", Text: "beta"})
	if err := saveStoreOpt(st2, path, rag.VectorsExact, true); err != nil {
		t.Fatalf("second save: %v", err)
	}

	bak, err := os.ReadFile(path + ".bak")
	if err != nil {
		t.Fatalf("no .bak after a backup save: %v", err)
	}
	loaded, err := rag.Load(atomicTestEmbedder{}, bytes.NewReader(bak))
	if err != nil {
		t.Fatalf("bak unreadable: %v", err)
	}
	if loaded.HasDoc("b") {
		t.Fatal(".bak holds the new store; it must hold the PREVIOUS one")
	}
	if !loaded.HasDoc("a") {
		t.Fatal(".bak missing the previous document")
	}
}
