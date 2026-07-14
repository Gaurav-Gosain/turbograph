package rag

import (
	"fmt"
	"os"
	"path/filepath"
)

// Lock is an advisory lock over a store file, held for the duration of a read or a
// read-modify-write cycle.
//
// It exists because the store's mutex is in-process only, and the unit of work an
// agent performs is a whole process: `turbograph add` loads the .tg, changes it, and
// writes it back. Two of those running at once — which is the normal case, since
// agents fan out — both load the same store, and the second one's save annihilates
// the first one's document. Nothing errors. The work is simply gone.
//
// The lock is taken on a sidecar <store>.lock rather than on the store itself,
// because saving replaces the store by rename: a lock held on the old inode would
// protect nothing once the file underneath it had been swapped.
type Lock struct {
	f    *os.File
	path string
}

// LockStore takes an advisory lock for the store at path. A read lock is shared, so
// concurrent readers do not block each other; a write lock is exclusive. It blocks
// until the lock is available.
//
// This is a local-filesystem guarantee. flock and LockFileEx are not dependable over
// NFS or SMB, so two machines writing the same store on a network share are not
// protected. Do not do that.
func LockStore(path string, read bool) (*Lock, error) {
	lp := path + ".lock"
	if dir := filepath.Dir(lp); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	f, err := os.OpenFile(lp, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("rag: cannot open the store lock %s: %w", lp, err)
	}
	if err := lockFD(f, read); err != nil {
		f.Close()
		return nil, fmt.Errorf("rag: cannot lock %s: %w", lp, err)
	}
	return &Lock{f: f, path: lp}, nil
}

// Unlock releases the lock. The sidecar file is left in place: removing it would race
// with another process that has already opened it and is waiting on the lock.
func (l *Lock) Unlock() error {
	if l == nil || l.f == nil {
		return nil
	}
	err := unlockFD(l.f)
	if cerr := l.f.Close(); err == nil {
		err = cerr
	}
	l.f = nil
	return err
}
