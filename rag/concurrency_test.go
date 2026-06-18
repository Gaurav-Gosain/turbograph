package rag

import (
	"context"
	"sync"
	"testing"
)

// TestConcurrentRetrieve exercises many simultaneous queries against one store.
// Run with -race to assert the read path is free of data races.
func TestConcurrentRetrieve(t *testing.T) {
	st := newTestStore(t)
	ctx := context.Background()
	queries := []string{
		"alpha core principle", "omega downstream application",
		"bridge linking domains", "primary subject matter",
	}
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				q := queries[(g+i)%len(queries)]
				if _, err := st.Retrieve(ctx, q, RetrieveParams{TopK: 3}); err != nil {
					t.Errorf("retrieve failed: %v", err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}
