package rag

import (
	"context"
	"fmt"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// benchStore builds a deterministic store of n documents with varied vocabulary,
// the workload the per-query retrieval hot path runs against.
func benchStore(b *testing.B, n int) *Store {
	b.Helper()
	vocab := make([]string, 400)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("term%d", i)
	}
	rng := quant.NewPCG(99)
	docs := make([]Document, n)
	for i := range docs {
		words := make([]byte, 0, 256)
		for w := 0; w < 40; w++ {
			words = append(words, vocab[rng.Uint64()%uint64(len(vocab))]...)
			words = append(words, ' ')
		}
		docs[i] = Document{ID: fmt.Sprintf("d%d", i), Text: string(words)}
	}
	s := New(newKeywordEmbedder(256), Config{Seed: 1, GraphKNN: 8, MinSimilarity: 0.1})
	if err := s.Build(context.Background(), docs); err != nil {
		b.Fatal(err)
	}
	return s
}

func BenchmarkRetrieve(b *testing.B) {
	s := benchStore(b, 3000)
	ctx := context.Background()
	q := "term12 term200 term37 term150 term5"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Retrieve(ctx, q, RetrieveParams{TopK: 10}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRetrieveGraph(b *testing.B) {
	s := benchStore(b, 3000)
	ctx := context.Background()
	q := "term12 term200 term37 term150 term5"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Retrieve(ctx, q, RetrieveParams{TopK: 10, GraphMix: 0.3}); err != nil {
			b.Fatal(err)
		}
	}
}
