package rag

import (
	"bytes"
	"context"
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	st := newTestStore(t)
	var buf bytes.Buffer
	if err := st.Save(&buf); err != nil {
		t.Fatal(err)
	}
	emb := newKeywordEmbedder(96)
	loaded, err := Load(emb, &buf)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != st.Len() {
		t.Fatalf("len mismatch: %d vs %d", loaded.Len(), st.Len())
	}
	q := "alpha core principle"
	a, _ := st.Retrieve(context.Background(), q, RetrieveParams{TopK: 5})
	b, _ := loaded.Retrieve(context.Background(), q, RetrieveParams{TopK: 5})
	if len(a) != len(b) {
		t.Fatalf("result count mismatch")
	}
	for i := range a {
		if a[i].Chunk.ID != b[i].Chunk.ID {
			t.Errorf("rank %d differs after reload: %s vs %s", i, a[i].Chunk.ID, b[i].Chunk.ID)
		}
	}
}
