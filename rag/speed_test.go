package rag

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

// The speed benchmarks measure turbograph, not Ollama.
//
// The BEIR runs report an end-to-end query latency of about 140 ms, and essentially all
// of it is the HTTP round trip to embed the query. Quoting that as "query latency" would
// credit the index with the embedder's cost and tell you nothing about the engine. These
// benchmarks use an in-process embedder, so what they time is search: the HNSW walk, the
// BM25 lookup, the fusion, and the ranking.
//
// Run: go test ./rag/ -run xxx -bench Speed -benchtime 200x

// speedCorpus gives each document its own vocabulary, as a real corpus does.
//
// This matters more than it looks. A templated corpus, where every document is the same
// sentence with the numbers changed, puts the query's terms in EVERY document, so BM25
// has to score the whole corpus and the search appears to scale linearly. Measuring on
// one of those told me turbograph's search was O(n) and sent me looking for a bug that
// does not exist. The pathological case is worth measuring, but it is not the typical
// one, and it must not be mistaken for it. See BenchmarkSpeedSearchWorstCase.
func speedCorpus(n int) []Document {
	rng := rand.New(rand.NewSource(3))
	vocab := make([]string, 20000)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("term%05d", i)
	}
	docs := make([]Document, n)
	for i := range docs {
		w := make([]byte, 0, 800)
		for j := 0; j < 90; j++ {
			w = append(w, vocab[rng.Intn(len(vocab))]...)
			w = append(w, ' ')
		}
		docs[i] = Document{ID: fmt.Sprintf("d%06d.md", i), Text: string(w)}
	}
	return docs
}

// worstCaseCorpus is the pathological shape: every document shares the query's terms, so
// the lexical index cannot narrow anything and BM25 scores the entire corpus.
func worstCaseCorpus(n int) []Document {
	docs := make([]Document, n)
	for i := range docs {
		docs[i] = Document{
			ID: fmt.Sprintf("w%06d.md", i),
			Text: fmt.Sprintf("Document %d. Project Helios-%d is a research effort at Lab%d led by engineer %d. "+
				"It depends on the Caldera-%d subsystem, built in Northgate and funded by the Orenda Foundation. "+
				"The retry queue is capped at %d attempts for reasons of idempotency, and the cache is "+
				"invalidated on every write to the ledger.", i, i, i%17, i%53, i%29, i%7),
		}
	}
	return docs
}

// speedStore builds once and caches on disk, so a benchmark process is not dominated by
// its own setup. 768 dimensions, because that is what a real embedding model emits and
// a 64-dimensional benchmark understates every per-vector cost by an order of magnitude.
func speedStore(tb testing.TB, n int) *Store {
	tb.Helper()
	const dim = 768
	path := fmt.Sprintf("/tmp/tg-speed-%d.tg", n)
	if b, err := os.ReadFile(path); err == nil {
		s, err := Load(newKeywordEmbedder(dim), bytes.NewReader(b))
		if err == nil {
			return s
		}
	}
	s := New(newKeywordEmbedder(dim), Config{Seed: 1})
	if err := s.Build(context.Background(), speedCorpus(n)); err != nil {
		tb.Fatal(err)
	}
	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		tb.Fatal(err)
	}
	os.WriteFile(path, buf.Bytes(), 0o644)
	return s
}

// BenchmarkSpeedSearch is the number comparable with another engine's: how long one
// query takes once the vector is in hand.
func BenchmarkSpeedSearch(b *testing.B) {
	for _, n := range []int{1000, 10000, 100000} {
		s := speedStore(b, n)
		// Warm the derived indexes so the first query is not charged for them.
		s.Retrieve(context.Background(), "warm", RetrieveParams{TopK: 10})
		b.Run(fmt.Sprintf("chunks=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := s.Retrieve(context.Background(), speedQuery, RetrieveParams{TopK: 10}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// speedQuery hits terms that exist but are not in every document.
const speedQuery = "term00042 term01337 term09999"

// BenchmarkSpeedSearchWorstCase: every document contains every query term, so BM25
// cannot narrow the candidate set and has to score the whole corpus. It is the ceiling,
// not the typical case, and it is here so the typical case is not mistaken for it.
func BenchmarkSpeedSearchWorstCase(b *testing.B) {
	for _, n := range []int{10000, 100000} {
		s := New(newKeywordEmbedder(768), Config{Seed: 1})
		if err := s.Build(context.Background(), worstCaseCorpus(n)); err != nil {
			b.Fatal(err)
		}
		s.Retrieve(context.Background(), "warm", RetrieveParams{TopK: 10})
		b.Run(fmt.Sprintf("chunks=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s.Retrieve(context.Background(), "Caldera subsystem Northgate ledger", RetrieveParams{TopK: 10})
			}
		})
	}
}

// BenchmarkSpeedSearchGraph is the same, with the PageRank graph signal on. It is the
// arm that has to justify its cost.
func BenchmarkSpeedSearchGraph(b *testing.B) {
	s := speedStore(b, 10000)
	s.Retrieve(context.Background(), "warm", RetrieveParams{TopK: 10, GraphMix: 0.2})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Retrieve(context.Background(), speedQuery,
			RetrieveParams{TopK: 10, GraphMix: 0.2}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpeedOpen is what every CLI invocation pays before it does anything.
func BenchmarkSpeedOpen(b *testing.B) {
	for _, n := range []int{10000, 100000} {
		blob, err := os.ReadFile(fmt.Sprintf("/tmp/tg-speed-%d.tg", n))
		if err != nil {
			speedStore(b, n)
			blob, err = os.ReadFile(fmt.Sprintf("/tmp/tg-speed-%d.tg", n))
			if err != nil {
				b.Fatal(err)
			}
		}
		b.Run(fmt.Sprintf("chunks=%d/size=%dMB", n, len(blob)/1024/1024), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := Load(newKeywordEmbedder(768), bytes.NewReader(blob)); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestSpeedFootprint reports the resident memory of a loaded, searchable store. An
// engine that is fast because it holds the whole corpus in RAM should say how much.
func TestSpeedFootprint(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 100k-chunk store")
	}
	const n = 100000
	s := speedStore(t, n)
	s.Retrieve(context.Background(), "warm", RetrieveParams{TopK: 10})

	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	// Without this the store is unreachable after its last use, the GC collects it before
	// the measurement, and the test cheerfully reports 90 MB for a gigabyte of vectors.
	runtime.KeepAlive(s)

	vecMB := n * 768 * 4 / 1_000_000
	t.Logf("%d chunks at 768-d: heap %d MB (the vectors alone are %d MB)",
		n, m.HeapAlloc/1_000_000, vecMB)
	t.Logf("  the store's embeddings are views into the index's buffer, so the vectors")
	t.Logf("  are resident once rather than twice; see TestVectorsAreNotHeldTwice")
}

// TestVectorsAreNotHeldTwice pins the sharing. It is easy for this to regress silently:
// any path that appends to the index without re-pointing the store's views leaves the
// old buffer alive, and the only symptom is that memory quietly doubles.
func TestVectorsAreNotHeldTwice(t *testing.T) {
	s := New(newKeywordEmbedder(64), Config{Seed: 1})
	if err := s.Build(context.Background(), speedCorpus(200)); err != nil {
		t.Fatal(err)
	}
	// Force the index up, then assert the store's vectors ARE the index's vectors.
	if _, err := s.Retrieve(context.Background(), speedQuery, RetrieveParams{TopK: 5}); err != nil {
		t.Fatal(err)
	}
	shared := func(when string) {
		t.Helper()
		s.mu.RLock()
		defer s.mu.RUnlock()
		if s.hnsw == nil || s.hnsw.Len() != len(s.embeds) {
			t.Fatalf("%s: index and store disagree on size", when)
		}
		for i := range s.embeds {
			v := s.hnsw.Vector(i)
			if len(v) != len(s.embeds[i]) {
				t.Fatalf("%s: chunk %d length mismatch", when, i)
			}
			if len(v) > 0 && &v[0] != &s.embeds[i][0] {
				t.Errorf("%s: chunk %d is a copy, not a view: the vectors are resident twice", when, i)
				return
			}
		}
	}
	shared("after build")

	// An append reallocates the index's buffer. If the views are not re-pointed, they
	// keep the old array alive and the duplication is back.
	if err := s.AddDocuments(context.Background(),
		[]Document{{ID: "extra", Text: "term00042 term01337 an additional document"}}); err != nil {
		t.Fatal(err)
	}
	shared("after an append")
}

// BenchmarkSpeedOpenAndSearch is what an agent's `turbograph search` actually costs:
// open the store, build the index from what was saved, run one query.
func BenchmarkSpeedOpenAndSearch(b *testing.B) {
	speedStore(b, 100000)
	blob, err := os.ReadFile("/tmp/tg-speed-100000.tg")
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		st, err := Load(newKeywordEmbedder(768), bytes.NewReader(blob))
		if err != nil {
			b.Fatal(err)
		}
		if _, err := st.Retrieve(context.Background(), speedQuery, RetrieveParams{TopK: 10}); err != nil {
			b.Fatal(err)
		}
	}
}
