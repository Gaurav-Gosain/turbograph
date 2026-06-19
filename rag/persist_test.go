package rag

import (
	"bytes"
	"context"
	"encoding/json"
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

func TestExportJSON(t *testing.T) {
	st := New(newKeywordEmbedder(64), Config{Seed: 1, Chunk: ChunkConfig{TargetWords: 8}})
	if err := st.Build(context.Background(), []Document{{ID: "d", Text: "alpha beta gamma delta epsilon zeta", Meta: map[string]any{"k": "v"}}}); err != nil {
		t.Fatal(err)
	}
	var buf, jsonBuf bytes.Buffer
	if err := st.Save(&buf); err != nil {
		t.Fatal(err)
	}
	if err := ExportJSON(&buf, &jsonBuf, false); err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(jsonBuf.Bytes(), &got); err != nil {
		t.Fatalf("export not valid json: %v", err)
	}
	if got["chunks"] == nil || got["doc_meta"] == nil {
		t.Fatalf("export missing fields: %v", keysOf(got))
	}
	chunks := got["chunks"].([]any)
	c0 := chunks[0].(map[string]any)
	if _, ok := c0["start"]; !ok {
		t.Fatalf("chunk missing start offset: %v", c0)
	}
	if got["embeds"] != nil {
		t.Fatal("no-vectors should omit embeds")
	}
}

func keysOf(m map[string]any) []string {
	var k []string
	for key := range m {
		k = append(k, key)
	}
	return k
}
