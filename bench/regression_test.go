package bench

import (
	"context"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// goldDataset is a small, deterministic, topically-separated corpus with labeled
// queries. With the deterministic HashEmbedder, the right document is the clear
// lexical match for each query, so the end-to-end pipeline (chunk, embed, index,
// fuse dense and lexical, rank, collapse to documents, score) should rank it at
// or near the top. This is a mechanics regression gate, run in CI offline with no
// model and no network; the absolute scores are not comparable to the literature
// numbers in docs/benchmarks.md, which use a real embedder.
func goldDataset() *Dataset {
	docs := []rag.Document{
		{ID: "space", Text: "Rockets reach orbit by burning propellant for thrust. Astronauts experience microgravity aboard the space station as it circles the planet."},
		{ID: "cooking", Text: "A good sourdough needs a ripe starter, a long fermentation, and a hot oven. Knead the dough, proof it, then bake until the crust is dark."},
		{ID: "finance", Text: "Central banks raise interest rates to curb inflation. Bond yields rise and equity valuations compress when the cost of capital increases."},
		{ID: "biology", Text: "Photosynthesis in chloroplasts converts sunlight and carbon dioxide into glucose, releasing oxygen as a byproduct of the light reactions."},
		{ID: "programming", Text: "A hash map stores key value pairs with average constant time lookup by hashing the key into a bucket array and resolving collisions."},
		{ID: "sports", Text: "In tennis a player wins a game by four points, a set by six games, and a match by two or three sets depending on the tournament."},
		{ID: "music", Text: "A major scale has seven notes and a pattern of whole and half steps. Chords stack thirds, and a cadence resolves tension back to the tonic."},
		{ID: "history", Text: "The printing press let movable type reproduce books quickly, spreading literacy across Europe and accelerating the scientific revolution."},
		{ID: "medicine", Text: "Vaccines train the immune system by presenting an antigen, so memory cells recognize the pathogen and respond faster on later exposure."},
		{ID: "weather", Text: "A cold front forms when a cold air mass displaces warm air, lifting it rapidly, which can trigger thunderstorms and a sharp temperature drop."},
		{ID: "geology", Text: "Plate tectonics moves continents over the mantle. Where plates converge, subduction builds mountains and feeds volcanoes along the boundary."},
		{ID: "networking", Text: "TCP provides reliable ordered delivery over IP using sequence numbers, acknowledgements, and a sliding window for flow and congestion control."},
	}
	cases := []eval.Case{
		{Query: "how do rockets reach orbit and what is microgravity", Relevant: []string{"space"}},
		{Query: "sourdough starter fermentation and baking the dough", Relevant: []string{"cooking"}},
		{Query: "why do central banks raise interest rates to fight inflation", Relevant: []string{"finance"}},
		{Query: "photosynthesis converts sunlight and carbon dioxide into glucose", Relevant: []string{"biology"}},
		{Query: "hash map constant time lookup by hashing keys into buckets", Relevant: []string{"programming"}},
		{Query: "tennis scoring points games sets and match", Relevant: []string{"sports"}},
		{Query: "major scale notes chords and cadence resolving to the tonic", Relevant: []string{"music"}},
		{Query: "printing press movable type spreading literacy in europe", Relevant: []string{"history"}},
		{Query: "how vaccines train the immune system with an antigen", Relevant: []string{"medicine"}},
		{Query: "cold front lifting warm air triggering thunderstorms", Relevant: []string{"weather"}},
		{Query: "plate tectonics subduction building mountains and volcanoes", Relevant: []string{"geology"}},
		{Query: "tcp reliable ordered delivery sequence numbers and window", Relevant: []string{"networking"}},
	}
	return &Dataset{Name: "gold", Docs: docs, Cases: cases}
}

func TestRetrievalRegression(t *testing.T) {
	ds := goldDataset()
	cfg := rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05}
	rep, err := Evaluate(context.Background(), HashEmbedder{Dim: 256}, cfg, ds, Options{K: 5, DocLevel: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("gold suite (n=%d): recall@5=%.3f ndcg@5=%.3f mrr=%.3f precision@5=%.3f",
		len(ds.Cases), rep.Mean.RecallAtK, rep.Mean.NDCGAtK, rep.Mean.MRR, rep.Mean.PrecisionAtK)

	// Floors gate the pipeline mechanics. The right document is the clear lexical
	// match, so the pipeline must rank it at or very near the top.
	if rep.Mean.RecallAtK < 0.95 {
		t.Errorf("recall@5 regressed: %.3f < 0.95", rep.Mean.RecallAtK)
	}
	if rep.Mean.MRR < 0.85 {
		t.Errorf("MRR regressed: %.3f < 0.85", rep.Mean.MRR)
	}
	if rep.Mean.NDCGAtK < 0.85 {
		t.Errorf("ndcg@5 regressed: %.3f < 0.85", rep.Mean.NDCGAtK)
	}
}

func TestLexicalFusionHelpsOrHolds(t *testing.T) {
	ds := goldDataset()
	base := rag.Config{Seed: 1, GraphKNN: 4, MinSimilarity: 0.05}
	ctx := context.Background()

	withLex, err := Evaluate(ctx, HashEmbedder{Dim: 256}, base, ds, Options{K: 5, DocLevel: true})
	if err != nil {
		t.Fatal(err)
	}
	noLexCfg := base
	noLexCfg.DisableLexical = true
	noLex, err := Evaluate(ctx, HashEmbedder{Dim: 256}, noLexCfg, ds, Options{K: 5, DocLevel: true})
	if err != nil {
		t.Fatal(err)
	}
	// Hybrid fusion should not do worse than pure dense on this suite.
	if withLex.Mean.NDCGAtK+1e-9 < noLex.Mean.NDCGAtK {
		t.Errorf("lexical fusion hurt ndcg: with=%.3f without=%.3f", withLex.Mean.NDCGAtK, noLex.Mean.NDCGAtK)
	}
}
