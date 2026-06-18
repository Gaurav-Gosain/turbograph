package lexical

import (
	"reflect"
	"testing"
)

func TestRRFFusesTwoLists(t *testing.T) {
	dense := []Result{{ID: "a", Score: 9}, {ID: "b", Score: 8}, {ID: "c", Score: 7}}
	sparse := []Result{{ID: "b", Score: 5}, {ID: "d", Score: 4}, {ID: "a", Score: 3}}

	fused := RRF(DefaultRRFK, dense, sparse)

	// Every ID across both lists should appear exactly once.
	if len(fused) != 4 {
		t.Fatalf("expected 4 fused results, got %d: %+v", len(fused), fused)
	}
	seen := map[string]int{}
	for _, r := range fused {
		seen[r.ID]++
	}
	for _, id := range []string{"a", "b", "c", "d"} {
		if seen[id] != 1 {
			t.Fatalf("id %q appears %d times, want 1", id, seen[id])
		}
	}

	// "a" (rank 0 dense, rank 2 sparse) and "b" (rank 1 dense, rank 0 sparse)
	// both appear in both lists and must outrank "c"/"d", which appear once.
	ra, rb := rank(fused, "a"), rank(fused, "b")
	rc, rd := rank(fused, "c"), rank(fused, "d")
	if ra > rc || ra > rd || rb > rc || rb > rd {
		t.Fatalf("docs present in both lists should outrank single-list docs: %+v", fused)
	}
}

func TestRRFAgreementBeatsSingleListTop(t *testing.T) {
	// "shared" is rank 1 in both lists. "soloA"/"soloB" are rank 0 but each in
	// only one list. Summed reciprocal ranks should lift the agreed-upon doc
	// above either solo leader.
	listA := []Result{{ID: "soloA"}, {ID: "shared"}, {ID: "x"}}
	listB := []Result{{ID: "soloB"}, {ID: "shared"}, {ID: "y"}}

	fused := RRF(DefaultRRFK, listA, listB)
	if fused[0].ID != "shared" {
		t.Fatalf("doc ranked high in both lists should win, got %+v", fused)
	}
}

func TestRRFScoreMath(t *testing.T) {
	// With k=60, a doc at rank 0 in two lists scores 2/60; a doc at rank 0 in
	// one list scores 1/60. Verify the exact values.
	listA := []Result{{ID: "both"}, {ID: "onlyA"}}
	listB := []Result{{ID: "both"}, {ID: "onlyB"}}
	fused := RRF(60, listA, listB)

	sBoth, _ := score(fused, "both")
	sOnly, _ := score(fused, "onlyA")
	if !approx(float64(sBoth), 2.0/60.0) {
		t.Fatalf("both-list score = %v, want %v", sBoth, 2.0/60.0)
	}
	if !approx(float64(sOnly), 1.0/61.0) {
		t.Fatalf("single-list rank-1 score = %v, want %v", sOnly, 1.0/61.0)
	}
}

func approx(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-6
}

func TestRRFDefaultKFallback(t *testing.T) {
	list := []Result{{ID: "a"}, {ID: "b"}}
	if !reflect.DeepEqual(RRF(0, list), RRF(DefaultRRFK, list)) {
		t.Fatal("k=0 should fall back to DefaultRRFK")
	}
	if !reflect.DeepEqual(RRF(-1, list), RRF(DefaultRRFK, list)) {
		t.Fatal("negative k should fall back to DefaultRRFK")
	}
}

func TestRRFEdgeCases(t *testing.T) {
	if got := RRF(60); got != nil {
		t.Fatalf("no lists should return nil, got %+v", got)
	}
	if got := RRF(60, nil, nil); got != nil {
		t.Fatalf("empty lists should return nil, got %+v", got)
	}
	if got := RRF(60, []Result{}); got != nil {
		t.Fatalf("single empty list should return nil, got %+v", got)
	}

	// A single non-empty list passes through, preserving order.
	single := []Result{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	got := RRF(60, single)
	want := []string{"a", "b", "c"}
	for i, w := range want {
		if got[i].ID != w {
			t.Fatalf("single list order wrong at %d: got %q want %q", i, got[i].ID, w)
		}
	}
}

func TestRRFDuplicateWithinListCountsOnce(t *testing.T) {
	// A duplicate ID in one list must count only at its first (best) position,
	// so the result equals fusing the deduped list.
	dup := []Result{{ID: "a"}, {ID: "b"}, {ID: "a"}}
	deduped := []Result{{ID: "a"}, {ID: "b"}}
	if !reflect.DeepEqual(RRF(60, dup), RRF(60, deduped)) {
		t.Fatalf("within-list duplicate not collapsed: %+v vs %+v",
			RRF(60, dup), RRF(60, deduped))
	}
}

func TestRRFDeterministic(t *testing.T) {
	a := []Result{{ID: "p"}, {ID: "q"}, {ID: "r"}}
	b := []Result{{ID: "r"}, {ID: "s"}, {ID: "p"}}
	first := RRF(60, a, b)
	for i := 0; i < 50; i++ {
		if !reflect.DeepEqual(first, RRF(60, a, b)) {
			t.Fatal("RRF is not deterministic")
		}
	}
}

func TestRRFTieBreakByID(t *testing.T) {
	// Two docs each at rank 0 in one distinct list tie on score; ID breaks it.
	a := []Result{{ID: "zeta"}}
	b := []Result{{ID: "alpha"}}
	fused := RRF(60, a, b)
	if fused[0].ID != "alpha" {
		t.Fatalf("tie should break by ID ascending, got %+v", fused)
	}
}

// TestHybridPipeline exercises the intended end-to-end use: a BM25 ranking
// fused with a simulated dense ranking via RRF.
func TestHybridPipeline(t *testing.T) {
	ix := Build(DefaultConfig(),
		[]string{"d1", "d2", "d3"},
		[]string{
			"neural networks for image classification",
			"reciprocal rank fusion combines retrievers",
			"sparse lexical search with bm25 scoring",
		},
	)
	sparse := ix.Search("bm25 scoring", 10)
	if len(sparse) == 0 || sparse[0].ID != "d3" {
		t.Fatalf("sparse retrieval wrong: %+v", sparse)
	}
	// Dense retriever (simulated) liked d2 and d3.
	dense := []Result{{ID: "d2", Score: 0.9}, {ID: "d3", Score: 0.8}}
	fused := RRF(DefaultRRFK, sparse, dense)
	// d3 is endorsed by both retrievers and should lead.
	if fused[0].ID != "d3" {
		t.Fatalf("hybrid fusion should surface d3 first, got %+v", fused)
	}
}
