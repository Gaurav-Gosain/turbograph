package lexical

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestAddBatchIsIdenticalToAdd: the parallel build must produce exactly the index the
// sequential one produces. An index that is subtly different does not fail, it just
// ranks everything slightly differently, forever, and nothing says so.
func TestAddBatchIsIdenticalToAdd(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	vocab := make([]string, 300)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("w%03d", i)
	}
	n := 2000
	ids := make([]string, n)
	texts := make([]string, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("d%04d", i)
		s := ""
		for j := 0; j < 40; j++ {
			s += vocab[rng.Intn(len(vocab))] + " "
		}
		texts[i] = s
	}

	seq := New(DefaultConfig())
	for i := range ids {
		seq.Add(ids[i], texts[i])
	}
	seq.Finalize()

	par := New(DefaultConfig())
	par.AddBatch(ids, texts)
	par.Finalize()

	if seq.Len() != par.Len() {
		t.Fatalf("length differs: %d vs %d", seq.Len(), par.Len())
	}
	if len(seq.postings) != len(par.postings) {
		t.Fatalf("vocabulary differs: %d terms vs %d", len(seq.postings), len(par.postings))
	}
	for term, want := range seq.postings {
		got := par.postings[term]
		if len(got) != len(want) {
			t.Fatalf("term %q: %d postings vs %d", term, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("term %q posting %d: %+v vs %+v", term, i, got[i], want[i])
			}
		}
	}
	// And the rankings must be identical, which is what actually matters.
	for _, q := range []string{"w001 w002", "w100", "w050 w150 w250"} {
		a := seq.Search(q, 10)
		b := par.Search(q, 10)
		if len(a) != len(b) {
			t.Fatalf("query %q: %d results vs %d", q, len(a), len(b))
		}
		for i := range a {
			if a[i].ID != b[i].ID || a[i].Score != b[i].Score {
				t.Fatalf("query %q rank %d: %v vs %v", q, i, a[i], b[i])
			}
		}
	}
}
