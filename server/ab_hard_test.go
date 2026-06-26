package server

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// The easy benchmark (TestABRetrievalImprovements) shows turbograph's baseline is
// already at ceiling when every document is a single self-contained chunk: there
// is no fragmentation to repair, so contextual retrieval cannot help. This test
// builds the condition contextual retrieval actually targets: longer documents
// that name an entity once and then refer to it anaphorically ("the programme",
// "it", "the effort"). When such a document is chunked, the later chunks carry a
// fact but lose the entity name, so a query naming the entity cannot match them
// lexically and only weakly matches them densely. Contextual retrieval prepends a
// situating sentence (generated from the whole document, which does contain the
// name) to each chunk's indexed text, which should rescue those fragment chunks.
//
// Scoring is at the CHUNK level against the specific answer-bearing chunk, because
// doc-level recall would be satisfied by the entity-naming first chunk and hide
// the effect entirely.
//
//	TG_AB=1 go test ./server/ -run TestABContextualHardMode -v -timeout 40m
type hardCase struct {
	q       string // names the entity; the answer lives in an anaphoric chunk
	locator string // a substring unique to the answer-bearing chunk
	answer  string
}

// Each document names its subject in the first sentence, then states facts with
// anaphora. Sentences are short so the recursive chunker keeps them in separate
// chunks (the later ones without the subject's name).
var hardDocs = []rag.Document{
	{ID: "helios", Text: "Project Helios is a fusion programme based at Verdant Labs. " +
		"The programme relies on the Caldera reactor for plasma confinement. " +
		"It is directed by a physicist who trained at Aldon University. " +
		"The effort is funded through a renewable Orenda Foundation grant."},
	{ID: "borealis", Text: "Project Borealis is a grid-scale battery programme at Verdant Labs. " +
		"The programme is built on the Ferrite cell chemistry. " +
		"It is led by an engineer who avoids rare-earth metals. " +
		"The initiative reports a cycle life beyond twelve thousand charges."},
	{ID: "selene", Text: "Project Selene was a lunar regolith study at the old Meridian campus. " +
		"The programme was discontinued after a funding review. " +
		"Its lead engineer later moved to a reactor team in Northgate. " +
		"The effort published a final survey of basalt processing yields."},
	{ID: "vela", Text: "Project Vela is a tidal energy programme run by the Marine Board. " +
		"The programme tests its turbines off the Saltmarsh coast. " +
		"It is supervised by a fluid-dynamics researcher from Brindle College. " +
		"The trial measures peak output during the spring tides."},
	{ID: "solaris", Text: "Project Solaris is a solar-thermal pilot at the Cobalt Institute. " +
		"The programme stores heat in a molten-salt loop. " +
		"It is sited in the arid Drift Valley for strong irradiance. " +
		"The pilot sustains output for six hours after sunset."},
	{ID: "amber", Text: "Amber Dynamics is a flywheel-storage company in Granite Bay. " +
		"The firm builds high-speed rotors for frequency regulation. " +
		"Its chief engineer previously worked on an offshore wind array. " +
		"The company sells its units to coastal grid operators."},
	{ID: "quill", Text: "The Quill Trust is a philanthropic fund for early-stage geoscience. " +
		"The trust was endowed by a geologist and investor. " +
		"It is headquartered in Ashfall City beside a dormant range. " +
		"The fund backed a geothermal survey of the Ashfall gradients."},
	{ID: "marrow", Text: "The Marrow survey mapped heat beneath the Ashfall mountain range. " +
		"The study was commissioned to find geothermal prospects. " +
		"It concluded in the year after a long drilling campaign. " +
		"The work identified three high-gradient zones for future wells."},
}

var hardCases = []hardCase{
	{"Which reactor does Project Helios rely on?", "Caldera reactor for plasma confinement", "Caldera"},
	{"Who funds Project Helios?", "renewable Orenda Foundation grant", "Orenda Foundation"},
	{"What cell chemistry is Project Borealis built on?", "built on the Ferrite cell chemistry", "Ferrite"},
	{"What cycle life does Project Borealis report?", "cycle life beyond twelve thousand", "twelve thousand"},
	{"Where did the Project Selene lead engineer move?", "reactor team in Northgate", "Northgate"},
	{"Off which coast does Project Vela test its turbines?", "off the Saltmarsh coast", "Saltmarsh"},
	{"Where is Project Solaris sited?", "arid Drift Valley", "Drift Valley"},
	{"How long does the Project Solaris pilot sustain output after sunset?", "six hours after sunset", "six hours"},
	{"What does Amber Dynamics build for frequency regulation?", "high-speed rotors for frequency regulation", "rotors"},
	{"Where is the Quill Trust headquartered?", "headquartered in Ashfall City", "Ashfall City"},
}

func TestABContextualHardMode(t *testing.T) {
	if os.Getenv("TG_AB") == "" {
		t.Skip("set TG_AB=1 (and have Ollama running) to run the model-backed hard-mode benchmark")
	}
	chatModel := envOr("TG_AB_CHAT", "qwen3.5:4b")
	embedModel := envOr("TG_AB_EMBED", "nomic-embed-text")

	client := ollama.New()
	if url := os.Getenv("TG_AB_URL"); url != "" {
		client.BaseURL = url
	}
	client.SetEmbedModel(embedModel)
	ctx, cancel := context.WithTimeout(context.Background(), 38*time.Minute)
	defer cancel()

	// Small chunks with no overlap so a fact sentence lands in its own chunk,
	// stripped of the entity name introduced earlier in the document.
	cfg := rag.Config{Seed: 1, MinSimilarity: 0.02,
		Chunk: rag.ChunkConfig{Strategy: rag.StrategyRecursive, TargetWords: 12, OverlapWords: 0}}

	corpus := append(append([]rag.Document{}, hardDocs...), abDistractors...)
	t.Logf("hard corpus: %d docs (%d fragmented + %d distractors), chunk target=12 words",
		len(corpus), len(hardDocs), len(abDistractors))

	storePlain := rag.New(client, cfg)
	if err := storePlain.Build(ctx, corpus); err != nil {
		t.Fatalf("build plain: %v", err)
	}
	cfgC := cfg
	storeCtx := rag.New(client, cfgC)
	storeCtx.SetContextualizer(genAdapter{c: client, model: chatModel})
	if err := storeCtx.Build(ctx, corpus); err != nil {
		t.Fatalf("build contextual: %v", err)
	}

	// Locate the gold chunk id for each question by its unique substring.
	goldPlain := goldChunkIDs(t, storePlain)
	goldCtx := goldChunkIDs(t, storeCtx)

	score := func(name string, st *rag.Store, gold []string) {
		var r1, r3, mrr float64
		for i, hc := range hardCases {
			res, err := st.Retrieve(ctx, hc.q, rag.RetrieveParams{TopK: 10})
			if err != nil {
				t.Fatalf("%s retrieve %q: %v", name, hc.q, err)
			}
			ranked := make([]string, len(res))
			for j, r := range res {
				ranked[j] = r.Chunk.ID
			}
			rel := map[string]struct{}{gold[i]: {}}
			r1 += eval.RecallAtK(ranked, rel, 1)
			r3 += eval.RecallAtK(ranked, rel, 3)
			mrr += eval.MRR(ranked, rel)
		}
		n := float64(len(hardCases))
		t.Logf("%-22s chunkRecall@1=%.3f chunkRecall@3=%.3f mrr=%.3f", name, r1/n, r3/n, mrr/n)
	}

	t.Log("=== HARD MODE: chunk-level retrieval of anaphoric fact chunks (n=" + itoa(len(hardCases)) + ") ===")
	score("plain dense+bm25", storePlain, goldPlain)
	score("contextual retrieval", storeCtx, goldCtx)
}

// goldChunkIDs maps each hard case to the id of the chunk whose body contains its
// unique locator substring. The locator is retrieved verbatim (an exact-phrase
// query that BM25 ranks first) and the top result whose text actually contains
// the substring is the gold chunk. It fails loudly if no retrieved chunk contains
// the locator, which would mean the chunker split the sentence unexpectedly.
func goldChunkIDs(t *testing.T, st *rag.Store) []string {
	t.Helper()
	out := make([]string, len(hardCases))
	for i, hc := range hardCases {
		res, err := st.Retrieve(context.Background(), hc.locator, rag.RetrieveParams{TopK: 10})
		if err != nil {
			t.Fatalf("locate %q: %v", hc.locator, err)
		}
		var found string
		for _, r := range res {
			if strings.Contains(r.Chunk.Text, hc.locator) {
				found = r.Chunk.ID
				break
			}
		}
		if found == "" {
			t.Fatalf("locator %q matched no retrieved chunk; adjust corpus or chunk size", hc.locator)
		}
		out[i] = found
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
