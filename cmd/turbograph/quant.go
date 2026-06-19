package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/quant"
)

// cmdQuant is the entry point for turboquant codec tooling.
func cmdQuant(args []string) error {
	if len(args) == 0 {
		quantUsage()
		return nil
	}
	switch args[0] {
	case "bench":
		return cmdQuantBench(args[1:])
	case "-h", "--help", "help":
		quantUsage()
		return nil
	default:
		quantUsage()
		return fmt.Errorf("unknown quant subcommand %q", args[0])
	}
}

func quantUsage() {
	fmt.Fprint(os.Stderr, `turbograph quant - TurboQuant codec tooling

usage:
  turbograph quant bench [flags]   benchmark accuracy/size/speed across bit rates

run "turbograph quant bench -h" for flags.
`)
}

// cmdQuantBench measures the TurboQuant codec across bit rates on synthetic
// clustered unit vectors and prints a table: how much each configuration
// compresses, how well its low-variance Score ranking recovers the exact
// nearest neighbors, and how fast it encodes and scores.
func cmdQuantBench(args []string) error {
	fs := flag.NewFlagSet("quant bench", flag.ExitOnError)
	dim := fs.Int("dim", 768, "vector dimension")
	n := fs.Int("n", 5000, "database vectors")
	queries := fs.Int("queries", 100, "query vectors")
	topk := fs.Int("topk", 10, "recall cutoff k")
	bitsStr := fs.String("bits", "1,2,4,8", "comma-separated bit rates to compare")
	residual := fs.Int("residual", 32, "QJL residual projections (0-64); 0 disables debiasing")
	clusters := fs.Int("clusters", 16, "synthetic cluster centers")
	seed := fs.Int64("seed", 1, "determinism seed")
	fs.Parse(args)

	var bits []int
	for _, s := range strings.Split(*bitsStr, ",") {
		b, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || b < 1 || b > 8 {
			return fmt.Errorf("invalid bit rate %q (want 1-8)", s)
		}
		bits = append(bits, b)
	}

	fmt.Printf("TurboQuant codec benchmark: dim=%d  db=%d  queries=%d  topk=%d  residual=%d\n\n",
		*dim, *n, *queries, *topk, *residual)
	fmt.Printf("%-5s %10s %8s %9s %10s %12s %12s\n",
		"bits", "code(B)", "ratio", "recall", "cb-mse", "encode/s", "score/s")
	fmt.Println(strings.Repeat("-", 74))
	for _, b := range bits {
		r := quant.Benchmark(
			quant.Config{Dim: *dim, Bits: b, ResidualDims: *residual, Seed: uint64(*seed)},
			quant.BenchOptions{N: *n, Queries: *queries, TopK: *topk, Clusters: *clusters, Seed: uint64(*seed)},
		)
		fmt.Printf("%-5d %10d %7.1fx %8.3f %10.4f %12s %12s\n",
			r.Bits, r.CodeBytes, r.CompressionRatio, r.RecallAtK, r.CodebookMSE,
			humanRate(r.EncodeVecsPerSec), humanRate(r.QueryScoresPerSec))
	}
	return nil
}

// humanRate formats a per-second rate compactly (e.g. 1.2M, 850k).
func humanRate(v float64) string {
	switch {
	case v >= 1e6:
		return fmt.Sprintf("%.1fM", v/1e6)
	case v >= 1e3:
		return fmt.Sprintf("%.0fk", v/1e3)
	default:
		return fmt.Sprintf("%.0f", v)
	}
}
