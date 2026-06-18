//go:build !amd64 || noasm

package index

// dotf falls back to the portable implementation on non-amd64 platforms or when
// built with -tags noasm.
func dotf(a, b []float32) float32 { return dotfGo(a, b) }
