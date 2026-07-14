//go:build unix

package rag

import (
	"os"

	"golang.org/x/sys/unix"
)

// lockFD takes an advisory lock on fd, blocking until it is available. Shared when
// read is set, exclusive otherwise.
func lockFD(f *os.File, read bool) error {
	how := unix.LOCK_EX
	if read {
		how = unix.LOCK_SH
	}
	return unix.Flock(int(f.Fd()), how)
}

func unlockFD(f *os.File) error {
	return unix.Flock(int(f.Fd()), unix.LOCK_UN)
}
