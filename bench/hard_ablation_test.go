package bench

import (
	"context"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

// TestHardAblationFamilies pins that the multi_hop family is actually hard: the
// query shares vocabulary with only the answer document, and the corroborating
// bridge document is reachable solely through the chunk-similarity graph edge to
// the answer. The graph-boosted arm must therefore out-recall pure dense by a wide
// margin. If that stops holding, the family is no longer discriminating and the
// suite should be re-tuned, not silently weakened.
func TestHardAblationFamilies(t *testing.T) {
	ds, families := HardDataset()
	ctx := context.Background()
	cfg := hardCfg()

	var multi []int
	for i, f := range families {
		if f == "multi_hop" {
			multi = append(multi, i)
		}
	}
	subset := func(idx []int) *Dataset {
		out := &Dataset{Name: ds.Name + "-subset", Docs: ds.Docs}
		for _, i := range idx {
			out.Cases = append(out.Cases, ds.Cases[i])
		}
		return out
	}

	multiRS, _, err := Ablate(ctx, hardEmbedder(), cfg, subset(multi), []Arm{
		{Name: "dense only", Params: rag.RetrieveParams{LexicalWeight: -1, SeedK: 2}},
		{Name: "hybrid+graph 0.2", Params: rag.RetrieveParams{GraphMix: 0.2, SeedK: 2}},
	}, Options{K: 5, DocLevel: true})
	if err != nil {
		t.Fatal(err)
	}
	rm, rd := multiRS[1].Report.Mean.RecallAtK, multiRS[0].Report.Mean.RecallAtK
	t.Logf("multi_hop recall@5: dense-only %.3f  hybrid+graph %.3f", rd, rm)
	if rm < rd+0.15 {
		t.Errorf("multi_hop is not multi-hop: graph arm recall@5=%.3f should beat dense-only %.3f by a wide margin", rm, rd)
	}
}
