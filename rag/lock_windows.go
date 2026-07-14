//go:build windows

package rag

import (
	"os"

	"golang.org/x/sys/windows"
)

// lockFD takes an advisory lock on fd, blocking until it is available. Windows has no
// flock, but LockFileEx over the whole file is the equivalent, and it is what the Go
// toolchain itself uses for its module cache locks.
func lockFD(f *os.File, read bool) error {
	var flags uint32 // shared by default
	if !read {
		flags = windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	ol := new(windows.Overlapped)
	return windows.LockFileEx(windows.Handle(f.Fd()), flags, 0, ^uint32(0), ^uint32(0), ol)
}

func unlockFD(f *os.File) error {
	ol := new(windows.Overlapped)
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, ^uint32(0), ^uint32(0), ol)
}
