package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// hardTile aggregates metrics for one family of the hard set.
type hardTile struct {
	Cases   int          `json:"cases"`
	Mean    eval.Metrics `json:"mean"`
	Success []string     `json:"recalled_5,omitempty"`      // queries recalled at k
	Missed  []string     `json:"failed_recall_5,omitempty"` // queries not recalled at k
}

// HardReport is the machine-readable output of the hard-set run, written when the
// HARD_REPORT environment variable names a path. It is the CI artifact that makes
// a ranking change visible as a number before it can regress a real corpus.
type HardReport struct {
	Dataset  string              `json:"dataset"`
	Embedder string              `json:"embedder"`
	Cases    int                 `json:"cases"`
	Run      hardRunSpec         `json:"run"`
	Mean     eval.Metrics        `json:"mean"`
	ByFamily map[string]hardTile `json:"by_family"`
}

// hardRunSpec is the scalar, serializable subset of the retrieval settings that
// produced the report.
type hardRunSpec struct {
	Seed          uint64  `json:"seed"`
	GraphKNN      int     `json:"graph_knn"`
	MinSimilarity float32 `json:"min_similarity"`
	ChunkWords    int     `json:"chunk_words"`
	ChunkOverlap  int     `json:"chunk_overlap"`
	GraphMix      float32 `json:"graph_mix"`
	SeedK         int     `json:"seed_k"`
	K             int     `json:"k"`
}

// hardFloors are the per-family regression gates on the hard set.
type hardFloors struct {
	RecallAtK float64
	MRR       float64
	NDCGAtK   float64
}

// hardCfg chunks the hard corpus small, so the distraught family really splits its
// answer across chunks and the retrieval must fuse a multi-chunk document back
// together.
func hardCfg() rag.Config {
	return rag.Config{
		Seed: 1, GraphKNN: 4, MinSimilarity: 0.05,
		Chunk: rag.ChunkConfig{TargetWords: 18, OverlapWords: 3},
	}
}

func (t hardTile) gate(fl hardFloors) []string {
	var bad []string
	if t.Mean.RecallAtK+1e-9 < fl.RecallAtK {
		bad = append(bad, "recall@k")
	}
	if t.Mean.MRR+1e-9 < fl.MRR {
		bad = append(bad, "mrr")
	}
	if t.Mean.NDCGAtK+1e-9 < fl.NDCGAtK {
		bad = append(bad, "ndcg@k")
	}
	return bad
}

// TestRetrievalHardRegression scores the full pipeline on the synthetic hard set
// and gates each family's mean metrics above a floor. It always runs with the
// in-process HashEmbedder, so it is as fast in CI as anywhere else, and it writes
// a JSON report when HARD_REPORT is set.
func TestRetrievalHardRegression(t *testing.T) {
	ds, families := HardDataset()
	if len(ds.Cases) < 30 {
		t.Fatalf("hard set has %d cases, want >= 30", len(ds.Cases))
	}
	ctx := context.Background()
	cfg := hardCfg()
	opt := Options{
		K: 5, DocLevel: true,
		// GraphMix 0.2 is the ON setting for retrieval: without it the multi_hop
		// family (answer in one document, corroborating document reachable only
		// through the chunk-similarity graph) is provably only half-recallable.
		// SeedK is capped at 2 so the personalized-PageRank seeds concentrate on
		// the answer and the graph boost surfaces the bridge instead of being
		// diluted across the zero-similarity dense tail.
		Params: rag.RetrieveParams{GraphMix: 0.2, SeedK: 2},
	}
	rep, err := Evaluate(ctx, hardEmbedder(), cfg, ds, opt)
	if err != nil {
		t.Fatal(err)
	}

	byFamily := make(map[string]hardTile)
	for i, c := range rep.Cases {
		if i >= len(families) {
			break
		}
		f := families[i]
		tl := byFamily[f]
		tl.Cases++
		tl.Mean.RecallAtK += c.Metrics.RecallAtK
		tl.Mean.PrecisionAtK += c.Metrics.PrecisionAtK
		tl.Mean.MRR += c.Metrics.MRR
		tl.Mean.NDCGAtK += c.Metrics.NDCGAtK
		tl.Mean.ContextPrecisionAtK += c.Metrics.ContextPrecisionAtK
		if c.Metrics.RecallAtK > 0 {
			tl.Success = append(tl.Success, c.Query)
		} else {
			tl.Missed = append(tl.Missed, c.Query)
		}
		byFamily[f] = tl
	}
	for f, tl := range byFamily {
		n := float64(tl.Cases)
		if n == 0 {
			continue
		}
		tl.Mean.RecallAtK /= n
		tl.Mean.PrecisionAtK /= n
		tl.Mean.MRR /= n
		tl.Mean.NDCGAtK /= n
		tl.Mean.ContextPrecisionAtK /= n
		byFamily[f] = tl
	}

	report := HardReport{
		Dataset:  ds.Name,
		Embedder: fmt.Sprintf("HashEmbedder(dim=%d)", hardDim),
		Cases:    len(rep.Cases),
		Run: hardRunSpec{
			Seed: cfg.Seed, GraphKNN: cfg.GraphKNN, MinSimilarity: cfg.MinSimilarity,
			ChunkWords: cfg.Chunk.TargetWords, ChunkOverlap: cfg.Chunk.OverlapWords,
			GraphMix: opt.Params.GraphMix, SeedK: opt.Params.SeedK, K: rep.K,
		},
		Mean:     rep.Mean,
		ByFamily: byFamily,
	}
	j, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("\n%s", j)
	if p := os.Getenv("HARD_REPORT"); p != "" {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err == nil {
			if werr := os.WriteFile(p, j, 0o644); werr != nil {
				t.Logf("HARD_REPORT write: %v", werr)
			}
		} else {
			t.Logf("HARD_REPORT mkdir: %v", err)
		}
	}

	// Floors per family. A regression that only shows up on one family is caught
	// here, where the average of the easy gold suite would hide it.
	gates := map[string]hardFloors{
		"multi_hop":  {RecallAtK: 0.85, MRR: 0.55, NDCGAtK: 0.70},
		"near_miss":  {RecallAtK: 0.90, MRR: 0.75, NDCGAtK: 0.85},
		"distraught": {RecallAtK: 0.90, MRR: 0.75, NDCGAtK: 0.85},
	}
	for f, tl := range byFamily {
		for _, bad := range tl.gate(gates[f]) {
			t.Errorf("%s: %s regressed (got %.3f, floor %.3f)",
				f, bad, metricOf(tl.Mean, bad), floorOf(gates[f], bad))
		}
		for _, q := range tl.Missed {
			t.Logf("%s miss: %q", f, q)
		}
	}
}

func metricOf(m eval.Metrics, name string) float64 {
	switch name {
	case "recall@k":
		return m.RecallAtK
	case "mrr":
		return m.MRR
	default:
		return m.NDCGAtK
	}
}

func floorOf(f hardFloors, name string) float64 {
	switch name {
	case "recall@k":
		return f.RecallAtK
	case "mrr":
		return f.MRR
	default:
		return f.NDCGAtK
	}
}
