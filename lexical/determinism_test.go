package lexical

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestSearchIsDeterministic: the same index and the same query must give the same
// answer, every time.
//
// topK scans a Go map, and Go randomizes map iteration order on every run, so a
// comparison on score alone let the map decide which document survived a tie at the
// k-th place. The same query returned different results from one call to the next --
// silently, since every result was legitimately scored. It also meant no benchmark
// number was exactly reproducible.
func TestSearchIsDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	vocab := make([]string, 300)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("w%03d", i)
	}
	ix := New(DefaultConfig())
	for i := 0; i < 2000; i++ {
		s := ""
		for j := 0; j < 40; j++ {
			s += vocab[rng.Intn(len(vocab))] + " "
		}
		ix.Add(fmt.Sprintf("d%04d", i), s)
	}
	ix.Finalize()

	// The SAME index, the SAME query, 20 times.
	first := ix.Search("w001 w002", 10)
	for run := 1; run < 20; run++ {
		got := ix.Search("w001 w002", 10)
		for i := range first {
			if got[i].ID != first[i].ID {
				t.Fatalf("run %d: rank %d is %s, but the first run said %s (same index, same query)",
					run, i, got[i].ID, first[i].ID)
			}
		}
	}
	t.Log("20 identical searches agreed")
}
