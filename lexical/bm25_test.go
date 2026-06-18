package lexical

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// rank returns the 0-based position of id in rs, or -1 if absent.
func rank(rs []Result, id string) int {
	for i, r := range rs {
		if r.ID == id {
			return i
		}
	}
	return -1
}

func score(rs []Result, id string) (float32, bool) {
	for _, r := range rs {
		if r.ID == id {
			return r.Score, true
		}
	}
	return 0, false
}

func TestSearchRanksMatchAboveNonMatch(t *testing.T) {
	ix := Build(DefaultConfig(),
		[]string{"hit", "miss"},
		[]string{
			"the quick brown fox jumps over the lazy dog",
			"completely unrelated text about sailing ships",
		},
	)
	got := ix.Search("brown fox", 10)
	if len(got) != 1 {
		t.Fatalf("expected only the matching doc, got %d results: %+v", len(got), got)
	}
	if got[0].ID != "hit" {
		t.Fatalf("expected hit first, got %q", got[0].ID)
	}
	if got[0].Score <= 0 {
		t.Fatalf("matching doc should have positive score, got %v", got[0].Score)
	}
}

func TestIDFRareTermsWeighMore(t *testing.T) {
	// "common" appears in every document; "rare" appears in exactly one. A doc
	// matched on the rare term must outrank a doc matched on the common term.
	ids := []string{"d0", "d1", "d2", "d3", "raredoc"}
	texts := []string{
		"common alpha beta",
		"common gamma delta",
		"common epsilon zeta",
		"common eta theta",
		"common rare", // shares "common" with all, plus the unique "rare"
	}
	ix := Build(DefaultConfig(), ids, texts)

	if ix.idf["rare"] <= ix.idf["common"] {
		t.Fatalf("rare IDF (%v) should exceed common IDF (%v)",
			ix.idf["rare"], ix.idf["common"])
	}

	// A query for both terms: the doc owning the rare term should win clearly.
	got := ix.Search("common rare", 10)
	if got[0].ID != "raredoc" {
		t.Fatalf("doc with rare term should rank first, got %+v", got)
	}

	// Direct single-term comparison: searching "rare" yields more score than
	// any single-doc match on "common".
	rareHits := ix.Search("rare", 10)
	commonHits := ix.Search("common", 10)
	if rareHits[0].Score <= commonHits[0].Score {
		t.Fatalf("rare-term score (%v) should exceed common-term score (%v)",
			rareHits[0].Score, commonHits[0].Score)
	}
}

func TestStopwordsAndShortTokensIgnored(t *testing.T) {
	ix := Build(DefaultConfig(),
		[]string{"d0"},
		[]string{"the a of to in is it"}, // all stopwords or length < 2
	)
	if ix.docLen[0] != 0 {
		t.Fatalf("expected all tokens dropped, docLen = %d", ix.docLen[0])
	}
	for _, q := range []string{"the", "a", "of", "i", "x"} {
		if got := ix.Search(q, 10); got != nil {
			t.Fatalf("query %q should match nothing, got %+v", q, got)
		}
	}

	// Mixed content: only the content word survives and is indexed.
	ix2 := Build(DefaultConfig(),
		[]string{"d0"},
		[]string{"it is the quasar"},
	)
	if _, ok := ix2.postings["quasar"]; !ok {
		t.Fatal("content word 'quasar' should be indexed")
	}
	if _, ok := ix2.postings["the"]; ok {
		t.Fatal("stopword 'the' should not be indexed")
	}
}

func TestLengthNormalizationPenalizesLongDocs(t *testing.T) {
	// Two docs each contain "signal" exactly once. The short doc should score
	// higher because BM25 length normalization penalizes the padded long doc.
	short := "signal here"
	long := "signal " + strings.Repeat("padding ", 200)

	ix := Build(DefaultConfig(), []string{"short", "long"}, []string{short, long})
	got := ix.Search("signal", 10)
	if got[0].ID != "short" {
		t.Fatalf("short doc should outrank long doc, got %+v", got)
	}

	// With b=0 length is ignored, so the single-occurrence scores must be equal.
	ixNoNorm := Build(Config{K1: 1.2, B: 0}, []string{"short", "long"}, []string{short, long})
	flat := ixNoNorm.Search("signal", 10)
	s0, _ := score(flat, "short")
	s1, _ := score(flat, "long")
	if math.Abs(float64(s0-s1)) > 1e-6 {
		t.Fatalf("with b=0 scores should be equal, got short=%v long=%v", s0, s1)
	}

	// And the penalty should grow with b: the long doc's score at b=0.75 is
	// strictly below its score at b=0.
	longB0, _ := score(flat, "long")
	longB75, _ := score(ix.Search("signal", 10), "long")
	if !(longB75 < longB0) {
		t.Fatalf("higher b should penalize the long doc more: b0=%v b75=%v", longB0, longB75)
	}
}

func TestTermFrequencyHelps(t *testing.T) {
	// Same length, but one doc mentions the term more often. More occurrences
	// should score higher (term-frequency saturation still increases).
	ix := Build(DefaultConfig(),
		[]string{"once", "thrice"},
		[]string{
			"signal alpha beta gamma",
			"signal signal signal gamma",
		},
	)
	got := ix.Search("signal", 10)
	if got[0].ID != "thrice" {
		t.Fatalf("doc with more occurrences should rank first, got %+v", got)
	}
}

func TestTopKTruncation(t *testing.T) {
	ix := Build(DefaultConfig(),
		[]string{"a", "b", "c", "d"},
		[]string{"term x", "term y", "term z", "term w"},
	)
	got := ix.Search("term", 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
}

func TestEdgeCases(t *testing.T) {
	t.Run("empty corpus", func(t *testing.T) {
		ix := New(DefaultConfig())
		if got := ix.Search("anything", 5); got != nil {
			t.Fatalf("empty corpus should return nil, got %+v", got)
		}
		if ix.Len() != 0 {
			t.Fatalf("empty index Len should be 0, got %d", ix.Len())
		}
	})

	t.Run("empty query", func(t *testing.T) {
		ix := Build(DefaultConfig(), []string{"d0"}, []string{"hello world"})
		if got := ix.Search("", 5); got != nil {
			t.Fatalf("empty query should return nil, got %+v", got)
		}
		if got := ix.Search("   ,. !! ", 5); got != nil {
			t.Fatalf("punctuation-only query should return nil, got %+v", got)
		}
	})

	t.Run("non-positive k", func(t *testing.T) {
		ix := Build(DefaultConfig(), []string{"d0"}, []string{"hello world"})
		if got := ix.Search("hello", 0); got != nil {
			t.Fatalf("k=0 should return nil, got %+v", got)
		}
		if got := ix.Search("hello", -3); got != nil {
			t.Fatalf("negative k should return nil, got %+v", got)
		}
	})

	t.Run("k larger than corpus", func(t *testing.T) {
		ix := Build(DefaultConfig(), []string{"d0", "d1"}, []string{"alpha term", "beta term"})
		got := ix.Search("term", 100)
		if len(got) != 2 {
			t.Fatalf("expected all 2 matches, got %d", len(got))
		}
	})

	t.Run("query term absent from corpus", func(t *testing.T) {
		ix := Build(DefaultConfig(), []string{"d0"}, []string{"hello world"})
		if got := ix.Search("nonexistent", 5); got != nil {
			t.Fatalf("absent term should return nil, got %+v", got)
		}
	})
}

func TestDeterminism(t *testing.T) {
	ids := []string{"a", "b", "c", "d", "e"}
	texts := []string{
		"alpha beta term", "term gamma", "term delta epsilon",
		"term term zeta", "eta term theta",
	}
	ix := Build(DefaultConfig(), ids, texts)

	first := ix.Search("term", 10)
	for i := 0; i < 50; i++ {
		got := ix.Search("term", 10)
		if !reflect.DeepEqual(first, got) {
			t.Fatalf("search is not deterministic:\n %+v\nvs\n %+v", first, got)
		}
	}
}

func TestTieBreakByID(t *testing.T) {
	// Two identical docs produce identical scores; the tie must resolve by ID.
	ix := Build(DefaultConfig(),
		[]string{"zeta", "alpha"},
		[]string{"identical text body", "identical text body"},
	)
	got := ix.Search("identical text", 10)
	if got[0].ID != "alpha" || got[1].ID != "zeta" {
		t.Fatalf("tie should break by ID ascending, got %+v", got)
	}
}

func TestRepeatedQueryTermNotDoubleCounted(t *testing.T) {
	ix := Build(DefaultConfig(), []string{"d0", "d1"}, []string{"alpha beta", "alpha gamma"})
	single := ix.Search("alpha", 10)
	doubled := ix.Search("alpha alpha", 10)
	if !reflect.DeepEqual(single, doubled) {
		t.Fatalf("repeated query term changed results: %+v vs %+v", single, doubled)
	}
}

func TestAddAfterFinalizeRecomputes(t *testing.T) {
	// Searching finalizes the index; a subsequent Add must invalidate stats so
	// the new document participates in ranking and IDF reflects the new N.
	ix := New(DefaultConfig())
	ix.Add("d0", "shared unique")
	_ = ix.Search("shared", 10) // forces finalize

	ix.Add("d1", "shared other")
	got := ix.Search("unique", 10)
	if len(got) != 1 || got[0].ID != "d0" {
		t.Fatalf("post-Add search wrong: %+v", got)
	}
	// IDF of "shared" must now reflect df=2 over N=2.
	want := math.Log(1 + (2-2+0.5)/(2+0.5))
	if math.Abs(ix.idf["shared"]-want) > 1e-9 {
		t.Fatalf("IDF not recomputed after Add: got %v want %v", ix.idf["shared"], want)
	}
}

func TestConfigClamping(t *testing.T) {
	ix := New(Config{K1: -5, B: 9})
	if ix.cfg.K1 != 0 {
		t.Fatalf("negative K1 should clamp to 0, got %v", ix.cfg.K1)
	}
	if ix.cfg.B != 1 {
		t.Fatalf("B>1 should clamp to 1, got %v", ix.cfg.B)
	}
}

func TestBuildMismatchedLengths(t *testing.T) {
	// Build should index only the paired prefix without panicking.
	ix := Build(DefaultConfig(), []string{"a", "b", "c"}, []string{"only one"})
	if ix.Len() != 1 {
		t.Fatalf("expected 1 indexed doc, got %d", ix.Len())
	}
}
