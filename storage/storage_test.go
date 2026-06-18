package storage

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestLocalRoundTrip(t *testing.T) {
	ctx := context.Background()
	l, err := NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Put(ctx, "a.tg", []byte("hello")); err != nil {
		t.Fatal(err)
	}
	if err := l.Put(ctx, "b.tg", []byte("world")); err != nil {
		t.Fatal(err)
	}
	got, err := l.Get(ctx, "a.tg")
	if err != nil || string(got) != "hello" {
		t.Fatalf("get a.tg = %q, %v", got, err)
	}
	keys, _ := l.List(ctx, "")
	if len(keys) != 2 || keys[0] != "a.tg" || keys[1] != "b.tg" {
		t.Errorf("list = %v", keys)
	}
	if err := l.Delete(ctx, "a.tg"); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Get(ctx, "a.tg"); !errors.Is(err, ErrNotExist) {
		t.Errorf("expected ErrNotExist after delete, got %v", err)
	}
	if err := l.Delete(ctx, "missing"); !errors.Is(err, ErrNotExist) {
		t.Errorf("expected ErrNotExist deleting missing, got %v", err)
	}
}

// TestSigV4MatchesAWSExample verifies the signer against the worked GET example
// from the AWS SigV4 documentation, whose signature is published. If this passes,
// the signing implementation is correct.
func TestSigV4MatchesAWSExample(t *testing.T) {
	req, _ := http.NewRequest("GET", "https://examplebucket.s3.amazonaws.com/test.txt", nil)
	req.Header.Set("Range", "bytes=0-9")
	when := time.Date(2013, 5, 24, 0, 0, 0, 0, time.UTC)
	signV4(req, nil, when, "us-east-1", "s3",
		"AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

	want := "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
	auth := req.Header.Get("Authorization")
	if !strings.Contains(auth, "Signature="+want) {
		t.Errorf("SigV4 signature mismatch.\n got: %s\nwant signature: %s", auth, want)
	}
	if !strings.Contains(auth, "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date") {
		t.Errorf("unexpected signed headers: %s", auth)
	}
}

func TestNewS3Validation(t *testing.T) {
	if _, err := NewS3(S3Config{}); err == nil {
		t.Error("expected error for empty config")
	}
	s, err := NewS3(S3Config{Endpoint: "http://localhost:9000", Bucket: "b", AccessKey: "k", SecretKey: "s"})
	if err != nil {
		t.Fatal(err)
	}
	if got := s.objectURL("x.tg"); got != "http://localhost:9000/b/x.tg" {
		t.Errorf("objectURL = %s", got)
	}
}
