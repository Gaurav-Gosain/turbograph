//go:build amd64 && !noasm

package index

import "golang.org/x/sys/cpu"

// dotProductAVX computes the inner product of a and b over the largest multiple
// of 8 elements, using AVX (256-bit) loads and multiplies. It is implemented in
// dotf_amd64.s. The caller guarantees len(a) == len(b) and that the length is a
// multiple of 8.
func dotProductAVX(a, b []float32) float32

// useAVXDot gates the kernel on AVX, which has been standard since 2011. With
// -tags noasm or on a CPU without AVX, the portable path is used instead.
var useAVXDot = cpu.X86.HasAVX

// dotf returns the inner product, dispatching to the AVX kernel for the bulk of
// the vector and handling any tail in Go. The instruction set is checked once.
func dotf(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if !useAVXDot || n < 8 {
		return dotfGo(a[:n], b[:n])
	}
	main := n &^ 7 // largest multiple of 8
	s := dotProductAVX(a[:main], b[:main])
	for i := main; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}
