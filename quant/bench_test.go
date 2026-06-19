package quant

import "testing"

func TestBenchmarkRecallAndSize(t *testing.T) {
	opt := BenchOptions{N: 800, Queries: 40, TopK: 10, Seed: 7}
	// 4-bit with residual debiasing should recall the exact top-10 well.
	r := Benchmark(Config{Dim: 256, Bits: 4, ResidualDims: 32, Seed: 1}, opt)
	if r.RecallAtK < 0.85 {
		t.Errorf("4-bit recall@10 too low: %.3f", r.RecallAtK)
	}
	if r.CompressionRatio < 3 {
		t.Errorf("4-bit should compress >=3x vs float32, got %.2f", r.CompressionRatio)
	}
	if r.CodeBytes <= 0 || r.EncodeVecsPerSec <= 0 || r.QueryScoresPerSec <= 0 {
		t.Errorf("throughput/size fields not populated: %+v", r)
	}
}

func TestBenchmarkMoreBitsMoreRecall(t *testing.T) {
	opt := BenchOptions{N: 800, Queries: 40, TopK: 10, Seed: 3}
	lo := Benchmark(Config{Dim: 256, Bits: 2, ResidualDims: 32, Seed: 1}, opt)
	hi := Benchmark(Config{Dim: 256, Bits: 6, ResidualDims: 32, Seed: 1}, opt)
	if hi.RecallAtK < lo.RecallAtK {
		t.Errorf("more bits should not reduce recall: 2-bit %.3f vs 6-bit %.3f", lo.RecallAtK, hi.RecallAtK)
	}
	// Higher bit rate uses more bytes.
	if hi.CodeBytes <= lo.CodeBytes {
		t.Errorf("6-bit codes should be larger than 2-bit: %d vs %d", hi.CodeBytes, lo.CodeBytes)
	}
}

func TestBenchmarkDeterministic(t *testing.T) {
	cfg := Config{Dim: 128, Bits: 4, ResidualDims: 16, Seed: 1}
	opt := BenchOptions{N: 500, Queries: 20, TopK: 10, Seed: 9}
	a := Benchmark(cfg, opt)
	b := Benchmark(cfg, opt)
	if a.RecallAtK != b.RecallAtK || a.CodeBytes != b.CodeBytes {
		t.Errorf("benchmark accuracy/size must be deterministic: %+v vs %+v", a, b)
	}
}
