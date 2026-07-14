package index

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math/rand"
	"testing"
)

// TestRestoreIsIdenticalToRebuild is the property that makes persisting the graph
// safe. A restored index must return exactly what a rebuilt one returns; a link
// structure that silently degrades recall would be much worse than a slow load,
// because nothing would ever tell you.
func TestRestoreIsIdenticalToRebuild(t *testing.T) {
	const dim, n = 64, 2000
	rng := rand.New(rand.NewSource(7))
	vecs := make([][]float32, n)
	ids := make([]string, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i], ids[i] = v, fmt.Sprintf("c%04d", i)
	}
	cfg := HNSWConfig{M: 16, EfConstruction: 200, Seed: 1}

	built := NewHNSW(dim, cfg)
	for i := range vecs {
		built.Add(ids[i], vecs[i])
	}

	// Round-trip the graph through gob, as the store does.
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(built.Snapshot()); err != nil {
		t.Fatal(err)
	}
	onDisk := buf.Len() // read it BEFORE decoding, which drains the buffer
	var g Graph
	if err := gob.NewDecoder(&buf).Decode(&g); err != nil {
		t.Fatal(err)
	}
	t.Logf("graph on disk: %d KB for %d vectors (%d KB of raw float32)",
		onDisk/1024, n, n*dim*4/1024)

	restored, ok := RestoreHNSW(dim, cfg, ids, vecs, g)
	if !ok {
		t.Fatal("restore rejected a graph it produced itself")
	}

	// Every query must return the same ranked list.
	for qi := 0; qi < 50; qi++ {
		qv := make([]float32, dim)
		for j := range qv {
			qv[j] = float32(rng.NormFloat64())
		}
		a := built.Search(qv, 10, 64)
		b := restored.Search(qv, 10, 64)
		if len(a) != len(b) {
			t.Fatalf("query %d: rebuilt returned %d hits, restored returned %d", qi, len(a), len(b))
		}
		for i := range a {
			if a[i].ID != b[i].ID {
				t.Fatalf("query %d rank %d: rebuilt=%s restored=%s (the restored graph is not the same graph)",
					qi, i, a[i].ID, b[i].ID)
			}
		}
	}
}

// TestRestoreRejectsAStaleGraph: a graph that does not describe these vectors must be
// refused, not silently used, or a store that grew since it was saved would search a
// graph missing its newest chunks.
func TestRestoreRejectsAStaleGraph(t *testing.T) {
	const dim = 8
	cfg := HNSWConfig{M: 8, EfConstruction: 32, Seed: 1}
	h := NewHNSW(dim, cfg)
	for i := 0; i < 5; i++ {
		h.Add(fmt.Sprintf("c%d", i), make([]float32, dim))
	}
	g := h.Snapshot()

	// Six vectors, a graph describing five.
	ids := []string{"c0", "c1", "c2", "c3", "c4", "c5"}
	vecs := make([][]float32, 6)
	for i := range vecs {
		vecs[i] = make([]float32, dim)
	}
	if _, ok := RestoreHNSW(dim, cfg, ids, vecs, g); ok {
		t.Error("a graph describing 5 nodes was accepted for 6 vectors")
	}
}
