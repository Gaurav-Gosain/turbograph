package index

// dotfGo is the portable inner product. Eight independent accumulators break the
// floating-point dependency chain (a naive running sum serializes every add), and
// reslicing to a common length lets the compiler drop bounds checks and partially
// vectorize the body. It is the fallback when the AVX kernel is unavailable, and
// the reference the SIMD path is tested against.
func dotfGo(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	a = a[:n]
	b = b[:n]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}
