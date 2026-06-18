// Package storage abstracts where persisted stores live. A Blob is a minimal
// object store (put, get, delete, list of byte blobs by key) with two
// implementations: the local filesystem and any S3-compatible service (AWS S3,
// MinIO, Cloudflare R2, and similar). The S3 client is implemented on the
// standard library with hand-rolled SigV4 signing, so there is no heavy SDK
// dependency.
package storage

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ErrNotExist is returned by Get and Delete when a key is absent.
var ErrNotExist = errors.New("storage: key does not exist")

// Blob is a small object store keyed by string. Implementations must be safe for
// concurrent use.
type Blob interface {
	Put(ctx context.Context, key string, data []byte) error
	Get(ctx context.Context, key string) ([]byte, error)
	Delete(ctx context.Context, key string) error
	List(ctx context.Context, prefix string) ([]string, error)
}

// Local is a Blob backed by a directory on the local filesystem. Keys map to file
// names; subdirectories are created as needed.
type Local struct {
	dir string
}

// NewLocal creates a Local blob rooted at dir, creating it if necessary.
func NewLocal(dir string) (*Local, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Local{dir: dir}, nil
}

func (l *Local) path(key string) string { return filepath.Join(l.dir, filepath.FromSlash(key)) }

// Put writes the blob, replacing it atomically (write to a temp file then rename)
// so a crash never leaves a half-written store.
func (l *Local) Put(_ context.Context, key string, data []byte) error {
	p := l.path(key)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, p)
}

// Get reads the blob, mapping a missing file to ErrNotExist.
func (l *Local) Get(_ context.Context, key string) ([]byte, error) {
	data, err := os.ReadFile(l.path(key))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotExist
	}
	return data, err
}

// Delete removes the blob, mapping a missing file to ErrNotExist.
func (l *Local) Delete(_ context.Context, key string) error {
	err := os.Remove(l.path(key))
	if errors.Is(err, os.ErrNotExist) {
		return ErrNotExist
	}
	return err
}

// List returns the keys under the directory that start with prefix, sorted.
func (l *Local) List(_ context.Context, prefix string) ([]string, error) {
	entries, err := os.ReadDir(l.dir)
	if err != nil {
		return nil, err
	}
	var keys []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasPrefix(e.Name(), ".tmp-") {
			continue
		}
		if strings.HasPrefix(e.Name(), prefix) {
			keys = append(keys, e.Name())
		}
	}
	sort.Strings(keys)
	return keys, nil
}
