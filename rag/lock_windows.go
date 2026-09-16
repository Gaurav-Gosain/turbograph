//go:build windows

package rag

import (
	"os"

	"golang.org/x/sys/windows"
)

// lockFD takes an advisory lock on fd, blocking until it is available. Windows has no
// flock, but LockFileEx over the whole file is the equivalent, and it is what the Go
// toolchain itself uses for its module cache locks. With try set, it fails
// immediately instead of blocking when the lock is held elsewhere.
func lockFD(f *os.File, read, try bool) error {
	var flags uint32 // shared by default
	if !read {
		flags = windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	if try {
		flags |= windows.LOCKFILE_FAIL_IMMEDIATELY
	}
	ol := new(windows.Overlapped)
	return windows.LockFileEx(windows.Handle(f.Fd()), flags, 0, ^uint32(0), ^uint32(0), ol)
}

func unlockFD(f *os.File) error {
	ol := new(windows.Overlapped)
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, ^uint32(0), ^uint32(0), ol)
}
