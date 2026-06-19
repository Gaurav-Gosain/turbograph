package rag

import (
	"bytes"
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/Gaurav-Gosain/turbograph/storage"
)

// Manager owns a set of named, independent stores ("buckets"). Each bucket is a
// separate corpus with its own quantizer, indexes, similarity graph, and
// communities, so they can be kept apart for multitenancy or simply to organize
// different document sets. Buckets are persisted as separate blobs through a
// storage.Blob, which can be the local filesystem or any S3-compatible service. A
// nil blob keeps buckets in memory only.
//
// The Manager is safe for concurrent use. Individual stores are themselves
// concurrency-safe for reads.
type Manager struct {
	mu       sync.RWMutex
	blob     storage.Blob
	embedder Embedder
	cfg      Config
	stores   map[string]*Store
}

// SetConfig updates the configuration used when new buckets are created (for
// example to change the chunking strategy). Existing buckets keep the config they
// were built with. Safe for concurrent use.
func (m *Manager) SetConfig(cfg Config) {
	m.mu.Lock()
	m.cfg = cfg
	m.mu.Unlock()
}

// Config returns the manager's current new-bucket configuration.
func (m *Manager) Config() Config {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.cfg
}

// SetEmbedder swaps the embedder used for new buckets. Existing buckets keep
// theirs, since their stored vectors come from the original embedder; changing
// the embedding model for a populated bucket would mix incompatible vectors.
func (m *Manager) SetEmbedder(e Embedder) {
	m.mu.Lock()
	m.embedder = e
	m.mu.Unlock()
}

const storeExt = ".tg"

var bucketName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`)

// ValidBucketName reports whether name is an acceptable bucket identifier. The
// restriction also prevents path traversal, since names become blob keys.
func ValidBucketName(name string) bool { return bucketName.MatchString(name) }

// NewManager creates a manager that persists to a local directory, loading any
// existing buckets. A directory of "" keeps buckets in memory.
func NewManager(dir string, embedder Embedder, cfg Config) (*Manager, error) {
	var blob storage.Blob
	if dir != "" {
		l, err := storage.NewLocal(dir)
		if err != nil {
			return nil, err
		}
		blob = l
	}
	return NewManagerBlob(blob, embedder, cfg)
}

// NewManagerBlob creates a manager persisting through an arbitrary blob store (for
// example S3), loading any existing buckets.
func NewManagerBlob(blob storage.Blob, embedder Embedder, cfg Config) (*Manager, error) {
	m := &Manager{blob: blob, embedder: embedder, cfg: cfg, stores: map[string]*Store{}}
	if blob == nil {
		return m, nil
	}
	keys, err := blob.List(context.Background(), "")
	if err != nil {
		return nil, err
	}
	for _, key := range keys {
		if !strings.HasSuffix(key, storeExt) {
			continue
		}
		name := strings.TrimSuffix(key, storeExt)
		if !ValidBucketName(name) {
			continue
		}
		data, err := blob.Get(context.Background(), key)
		if err != nil {
			fmt.Printf("manager: skipping bucket %q: %v\n", name, err)
			continue
		}
		st, err := Load(embedder, bytes.NewReader(data))
		if err != nil {
			fmt.Printf("manager: skipping bucket %q: %v\n", name, err)
			continue
		}
		m.stores[name] = st
	}
	return m, nil
}

// Put inserts an externally constructed store under name. It is used to wrap a
// single store as a one-bucket manager.
func (m *Manager) Put(name string, st *Store) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stores[name] = st
}

// List returns the bucket names in sorted order.
func (m *Manager) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	names := make([]string, 0, len(m.stores))
	for n := range m.stores {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Get returns the store for name, if it exists.
func (m *Manager) Get(name string) (*Store, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	st, ok := m.stores[name]
	return st, ok
}

// Create makes a new empty bucket, erroring if the name is invalid or taken.
func (m *Manager) Create(name string) (*Store, error) {
	if !ValidBucketName(name) {
		return nil, fmt.Errorf("rag: invalid bucket name %q", name)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.stores[name]; ok {
		return nil, fmt.Errorf("rag: bucket %q already exists", name)
	}
	st := New(m.embedder, m.cfg)
	m.stores[name] = st
	return st, nil
}

// GetOrCreate returns the bucket, creating it if absent.
func (m *Manager) GetOrCreate(name string) (*Store, error) {
	if st, ok := m.Get(name); ok {
		return st, nil
	}
	st, err := m.Create(name)
	if err != nil {
		if st2, ok := m.Get(name); ok {
			return st2, nil
		}
		return nil, err
	}
	return st, nil
}

// Delete removes a bucket and its persisted blob.
func (m *Manager) Delete(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.stores[name]; !ok {
		return fmt.Errorf("rag: no such bucket %q", name)
	}
	delete(m.stores, name)
	if m.blob != nil {
		m.blob.Delete(context.Background(), name+storeExt)
	}
	return nil
}

// Save persists one bucket. It is a no-op for an in-memory manager.
func (m *Manager) Save(name string) error {
	if m.blob == nil {
		return nil
	}
	st, ok := m.Get(name)
	if !ok {
		return fmt.Errorf("rag: no such bucket %q", name)
	}
	var buf bytes.Buffer
	if err := st.Save(&buf); err != nil {
		return err
	}
	return m.blob.Put(context.Background(), name+storeExt, buf.Bytes())
}

// SaveAll persists every bucket.
func (m *Manager) SaveAll() error {
	for _, name := range m.List() {
		if err := m.Save(name); err != nil {
			return err
		}
	}
	return nil
}

// Path returns the blob key for a bucket (empty for an in-memory manager).
func (m *Manager) Path(name string) string {
	if m.blob == nil {
		return ""
	}
	return name + storeExt
}
