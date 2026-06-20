package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// captionBackend is a stubBackend that also captions images, so the multimodal
// path can be tested without a real vision model.
type captionBackend struct {
	stubBackend
	caption string
	gotByte int
}

func (c *captionBackend) CaptionImage(_ context.Context, _, _ string, image []byte) (string, error) {
	c.gotByte = len(image)
	return c.caption, nil
}

func TestAssetStorePutOpen(t *testing.T) {
	a, err := newAssetStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	id, err := a.Put([]byte("PNGDATA"), "png")
	if err != nil {
		t.Fatal(err)
	}
	if !assetID.MatchString(id) || !strings.HasSuffix(id, ".png") {
		t.Fatalf("bad id %q", id)
	}
	// Identical bytes are content-addressed to the same id.
	id2, _ := a.Put([]byte("PNGDATA"), "png")
	if id2 != id {
		t.Fatalf("not content-addressed: %q vs %q", id, id2)
	}
	data, ct, err := a.Open(id)
	if err != nil || string(data) != "PNGDATA" {
		t.Fatalf("open: %v %q", err, data)
	}
	if ct != "image/png" {
		t.Fatalf("content type %q", ct)
	}
	// Path traversal and junk ids are rejected.
	if _, _, err := a.Open("../secret"); err == nil {
		t.Fatal("expected rejection of traversal id")
	}
}

func TestIngestImageEndToEnd(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05})
	// Seed a text doc so the index exists.
	if err := store.Build(context.Background(), []rag.Document{{ID: "seed", Text: "unrelated text about weather"}}); err != nil {
		t.Fatal(err)
	}
	srv := New(store)
	if err := srv.EnableAssets(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	srv.SetGenerator(&captionBackend{caption: "A bar chart showing revenue rising each quarter of 2026."}, "vision", "embed")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	img := []byte("\x89PNG\r\n\x1a\nfake image bytes")
	body, _ := json.Marshal(ingestImageRequest{ID: "chart.png", B64: base64.StdEncoding.EncodeToString(img), Ext: "png", Model: "vision"})
	resp, err := http.Post(ts.URL+"/api/ingest/image", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("ingest status %d: %s", resp.StatusCode, b)
	}
	var out struct {
		ImageRef string `json:"image_ref"`
		Caption  string `json:"caption"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if out.ImageRef == "" || !strings.Contains(out.Caption, "revenue") {
		t.Fatalf("bad ingest result: %+v", out)
	}

	// The image is retrievable by its caption, and the chunk is marked as an image.
	res, err := store.Retrieve(context.Background(), "revenue chart quarter", rag.RetrieveParams{TopK: 3})
	if err != nil || len(res) == 0 {
		t.Fatalf("retrieve: %v", err)
	}
	var found *rag.Retrieved
	for i := range res {
		if res[i].Chunk.DocID == "chart.png" {
			found = &res[i]
		}
	}
	if found == nil {
		t.Fatalf("image not retrieved: %+v", res)
	}
	if found.Chunk.Kind != "image" || found.Chunk.ImageRef != out.ImageRef {
		t.Fatalf("chunk not marked as image: %+v", found.Chunk)
	}

	// The asset is served back.
	ar, err := http.Get(ts.URL + "/api/asset/" + out.ImageRef)
	if err != nil {
		t.Fatal(err)
	}
	defer ar.Body.Close()
	got, _ := io.ReadAll(ar.Body)
	if !bytes.Equal(got, img) {
		t.Fatalf("asset bytes mismatch")
	}
	if ct := ar.Header.Get("Content-Type"); ct != "image/png" {
		t.Fatalf("asset content type %q", ct)
	}
}

func TestIngestImageUnconfigured(t *testing.T) {
	store := rag.New(hashEmbedder{dim: 64}, rag.Config{Seed: 1})
	store.Build(context.Background(), []rag.Document{{ID: "s", Text: "hello world text"}})
	srv := New(store) // no EnableAssets
	srv.SetGenerator(stubBackend{}, "m", "e")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	body, _ := json.Marshal(ingestImageRequest{ID: "x.png", B64: "AAAA"})
	resp, _ := http.Post(ts.URL+"/api/ingest/image", "application/json", bytes.NewReader(body))
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 when assets unconfigured, got %d", resp.StatusCode)
	}
}

func BenchmarkAssetPut(b *testing.B) {
	a, _ := newAssetStore(b.TempDir())
	data := bytes.Repeat([]byte{0x42}, 64*1024) // 64 KiB image
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Vary one byte so each iteration hashes/writes distinct content.
		data[0] = byte(i)
		if _, err := a.Put(data, "png"); err != nil {
			b.Fatal(err)
		}
	}
}
