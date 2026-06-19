package storage

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
)

// fakeS3 is an in-memory object store speaking just enough of the S3 REST API
// (PUT/GET/DELETE object, ListObjectsV2) to round-trip the client, including its
// SigV4 signing (the requests are signed; the fake does not verify them).
type fakeS3 struct {
	mu      sync.Mutex
	objects map[string][]byte
}

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	// Path is /<bucket>/<key...>; strip the leading /bucket/ segment.
	parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 2)
	key := ""
	if len(parts) == 2 {
		key = parts[1]
	}
	switch r.Method {
	case http.MethodPut:
		buf, _ := io.ReadAll(r.Body)
		f.objects[key] = buf
		w.WriteHeader(http.StatusOK)
	case http.MethodGet:
		if r.URL.Query().Get("list-type") == "2" {
			f.list(w, r.URL.Query().Get("prefix"))
			return
		}
		data, ok := f.objects[key]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write(data)
	case http.MethodDelete:
		if _, ok := f.objects[key]; !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		delete(f.objects, key)
		w.WriteHeader(http.StatusNoContent)
	}
}

func (f *fakeS3) list(w http.ResponseWriter, prefix string) {
	var keys []string
	for k := range f.objects {
		if strings.HasPrefix(k, prefix) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	fmt.Fprint(w, `<?xml version="1.0"?><ListBucketResult>`)
	for _, k := range keys {
		fmt.Fprintf(w, "<Contents><Key>%s</Key></Contents>", k)
	}
	fmt.Fprint(w, `</ListBucketResult>`)
}

func TestS3RoundTrip(t *testing.T) {
	srv := httptest.NewServer(&fakeS3{objects: map[string][]byte{}})
	defer srv.Close()

	for _, prefix := range []string{"", "tenant/a"} {
		s, err := NewS3(S3Config{
			Endpoint: srv.URL, Bucket: "b", Region: "us-east-1",
			AccessKey: "AK", SecretKey: "SK", Prefix: prefix,
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx := context.Background()

		// Missing object -> ErrNotExist.
		if _, err := s.Get(ctx, "nope.tg"); err != ErrNotExist {
			t.Fatalf("prefix %q: expected ErrNotExist, got %v", prefix, err)
		}
		// Put / Get round-trips the bytes.
		want := []byte("hello turbograph")
		if err := s.Put(ctx, "default.tg", want); err != nil {
			t.Fatalf("prefix %q: put: %v", prefix, err)
		}
		got, err := s.Get(ctx, "default.tg")
		if err != nil || string(got) != string(want) {
			t.Fatalf("prefix %q: get mismatch: %q %v", prefix, got, err)
		}
		// Put another, list strips the configured prefix.
		s.Put(ctx, "other.tg", []byte("x"))
		keys, err := s.List(ctx, "")
		if err != nil {
			t.Fatalf("prefix %q: list: %v", prefix, err)
		}
		if len(keys) != 2 || keys[0] != "default.tg" || keys[1] != "other.tg" {
			t.Fatalf("prefix %q: unexpected list: %v", prefix, keys)
		}
		// Delete then confirm gone.
		if err := s.Delete(ctx, "other.tg"); err != nil {
			t.Fatalf("prefix %q: delete: %v", prefix, err)
		}
		if _, err := s.Get(ctx, "other.tg"); err != ErrNotExist {
			t.Fatalf("prefix %q: deleted object still present: %v", prefix, err)
		}
	}
}
