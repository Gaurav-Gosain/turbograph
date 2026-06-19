package quant

import (
	"math"
	"testing"
)

// FuzzEncodeDecode feeds arbitrary byte patterns in as vector coordinates and
// checks the codec never panics and always produces a finite, correctly shaped
// reconstruction. Quantization is data-oblivious, so it must tolerate any input,
// including NaN/Inf-adjacent and all-zero vectors.
func FuzzEncodeDecode(f *testing.F) {
	const dim = 16
	q := New(Config{Dim: dim, Bits: 4, ResidualDims: 8, Seed: 1})
	f.Add([]byte{0})
	f.Add([]byte{1, 2, 3, 4, 5})
	f.Add([]byte{255, 0, 128, 64})
	f.Fuzz(func(t *testing.T, data []byte) {
		v := make([]float32, dim)
		for i := range v {
			if len(data) == 0 {
				break
			}
			// Map a byte to [-1,1]; deterministic and bounded so a single fuzz
			// input never produces NaN/Inf in the source vector.
			v[i] = (float32(data[i%len(data)]) - 128) / 128
		}
		code := q.Encode(v)
		if len(code.Codes) != q.Dim() {
			t.Fatalf("code length %d != padded dim %d", len(code.Codes), q.Dim())
		}
		dec := q.Decode(code)
		if len(dec) != q.Dim() {
			t.Fatalf("decoded length %d != padded dim %d", len(dec), q.Dim())
		}
		for i, x := range dec {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("decoded[%d] is non-finite: %v", i, x)
			}
		}
		// A prepared query must score every code without panicking.
		qr := q.PrepareQuery(v)
		if s := qr.Score(code); math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			t.Fatalf("score is non-finite: %v", s)
		}
	})
}
