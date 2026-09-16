//go:build unix

package rag

import (
	"os"

	"golang.org/x/sys/unix"
)

// lockFD takes an advisory lock on fd, blocking until it is available. Shared when
// read is set, exclusive otherwise. With try set, it fails immediately instead of
// blocking when the lock is held elsewhere.
func lockFD(f *os.File, read, try bool) error {
	how := unix.LOCK_EX
	if read {
		how = unix.LOCK_SH
	}
	if try {
		how |= unix.LOCK_NB
	}
	return unix.Flock(int(f.Fd()), how)
}

func unlockFD(f *os.File) error {
	return unix.Flock(int(f.Fd()), unix.LOCK_UN)
}
