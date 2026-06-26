package server

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/ollama"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// abDoc and abCase define a tiny self-contained, fictional knowledge base. The
// universe is invented so a language model cannot answer from parametric memory:
// every correct answer must come from retrieval, which is exactly what we want to
// measure. Many questions are multi-hop (the answer lives by combining two facts
// in two different documents), which is where the entity graph, decomposition,
// and graph-fact injection are supposed to help. Distractor documents on adjacent
// topics are mixed in so retrieval is not saturated: returning a handful of docs
// from a tiny corpus would otherwise score a perfect recall for free.
type abCase struct {
	q      string
	gold   []string // document ids that must be retrieved to answer
	answer string   // short gold answer for cover / F1 / exact-match
}

var abDocs = []rag.Document{
	{ID: "helios", Text: "Project Helios is a fusion research effort at Verdant Labs. It is led by Dr. Mira Tan. Helios relies on the Caldera reactor for its plasma experiments."},
	{ID: "caldera", Text: "The Caldera reactor was built in the town of Northgate. It produces a stable magnetic confinement field. Caldera was funded by the Orenda Foundation."},
	{ID: "mira", Text: "Dr. Mira Tan earned her doctorate in plasma physics from Aldon University. She previously worked on the Selene project before joining Verdant Labs."},
	{ID: "selene", Text: "The Selene project studied lunar regolith processing. It was discontinued in 2019. Selene's lead engineer was Tomas Reyes."},
	{ID: "orenda", Text: "The Orenda Foundation funds clean-energy research across the continent. Its headquarters are in Northgate. The foundation was established by Priya Anand."},
	{ID: "verdant", Text: "Verdant Labs is a private research institute focused on fusion and energy storage. It operates two flagship projects, Helios and Borealis. Verdant Labs is headquartered in Aldon City."},
	{ID: "borealis", Text: "Project Borealis develops grid-scale battery storage. It is led by Dr. Sasha Vex. Borealis uses the Ferrite cell chemistry."},
	{ID: "ferrite", Text: "The Ferrite cell chemistry offers high cycle life at low cost. It was invented by Dr. Sasha Vex. Ferrite cells avoid rare-earth metals."},
	{ID: "tomas", Text: "Tomas Reyes is a robotics engineer. After Selene ended, he moved to the Caldera reactor team at Northgate. He specializes in remote handling systems."},
	{ID: "priya", Text: "Priya Anand is a philanthropist and former energy executive. She founded the Orenda Foundation in 2010. She sits on the board of Aldon University."},
	{ID: "aldon", Text: "Aldon University is a research university in Aldon City. Its plasma physics department is internationally ranked. The university partners with Verdant Labs."},
	{ID: "northgate", Text: "Northgate is an industrial town known for energy infrastructure. It hosts the Caldera reactor and the Orenda Foundation headquarters. The Northgate grid is fully renewable."},
}

// abDistractors are plausible, adjacent-topic documents that never answer any
// question. They give the retriever something to get wrong.
var abDistractors = []rag.Document{
	{ID: "d_solaris", Text: "Project Solaris is a solar-thermal pilot at Cobalt Institute, led by Dr. Owen Pike. It uses molten-salt storage in the Drift Valley."},
	{ID: "d_cobalt", Text: "Cobalt Institute studies concentrated solar power and hydrogen electrolysis. It is based in Drift Valley and directed by Dr. Owen Pike."},
	{ID: "d_pike", Text: "Dr. Owen Pike is a thermal engineer who trained at Brindle College. He led the Marrow geothermal survey before joining Cobalt Institute."},
	{ID: "d_vela", Text: "Project Vela develops tidal turbines for coastal grids. Its lead is Dr. Hana Roe. Vela is tested off the Saltmarsh coast."},
	{ID: "d_roe", Text: "Dr. Hana Roe is a fluid-dynamics researcher. She published the Saltmarsh tidal atlas and advises the Brindle College energy board."},
	{ID: "d_brindle", Text: "Brindle College is a small technical school known for materials science. It runs the annual Drift Valley energy symposium."},
	{ID: "d_marrow", Text: "The Marrow geothermal survey mapped heat gradients beneath the Ashfall range. It was funded by the Quill Trust and concluded in 2016."},
	{ID: "d_quill", Text: "The Quill Trust supports early-stage geoscience. It was founded by Eli Frost and is headquartered in Ashfall City."},
	{ID: "d_frost", Text: "Eli Frost is a geologist and investor. He chairs the Quill Trust and lectures occasionally at Brindle College."},
	{ID: "d_amber", Text: "Amber Dynamics builds flywheel storage for frequency regulation. Its chief engineer is Nadia Volk. The company operates from Granite Bay."},
	{ID: "d_volk", Text: "Nadia Volk designs high-speed rotors. Before Amber Dynamics she worked on the Granite Bay wind array."},
	{ID: "d_granite", Text: "Granite Bay is a coastal city with a large offshore wind array. It hosts Amber Dynamics and a maritime research dock."},
	{ID: "d_ion", Text: "The Ion cell chemistry is a competing battery design using rare-earth cathodes. It offers high density but a shorter cycle life."},
	{ID: "d_drift", Text: "Drift Valley is an arid region used for solar testbeds. It receives strong year-round irradiance and hosts the Cobalt Institute campus."},
	{ID: "d_saltmarsh", Text: "Saltmarsh is a tidal estuary studied for marine energy. Its strong currents make it ideal for turbine trials."},
	{ID: "d_ashfall", Text: "Ashfall City sits beside a dormant volcanic range. It is a center for geothermal research and home to the Quill Trust."},
}

var abCases = []abCase{
	{"In which town was the reactor used by Project Helios built?", []string{"helios", "caldera"}, "Northgate"},
	{"Who founded the foundation that funded the Caldera reactor?", []string{"caldera", "orenda"}, "Priya Anand"},
	{"What cell chemistry is used by the battery project at Verdant Labs?", []string{"verdant", "borealis"}, "Ferrite"},
	{"Who invented the chemistry used by Project Borealis?", []string{"borealis", "ferrite"}, "Sasha Vex"},
	{"Where did the lead of Project Helios earn her doctorate?", []string{"helios", "mira"}, "Aldon University"},
	{"Which project did the lead of Helios work on before Verdant Labs?", []string{"helios", "mira"}, "Selene"},
	{"Who was the lead engineer of the project Mira Tan worked on before Verdant Labs?", []string{"mira", "selene"}, "Tomas Reyes"},
	{"What reactor team did the former Selene lead engineer join?", []string{"selene", "tomas"}, "Caldera"},
	{"In which city is the institute that runs Project Helios headquartered?", []string{"helios", "verdant"}, "Aldon City"},
	{"Which foundation is headquartered in the same town as the Caldera reactor?", []string{"caldera", "orenda"}, "Orenda Foundation"},
	{"Who leads Project Borealis?", []string{"borealis"}, "Sasha Vex"},
	{"What does Project Selene study?", []string{"selene"}, "lunar regolith processing"},
}

// terseAnswerSystem constrains the generator to a short span so answer scoring
// reflects whether the retrieved context contained the answer, not the model's
// verbosity. CoverMatch / F1 then measure correctness robustly.
const terseAnswerSystem = "Answer the question using only the provided context. " +
	"Reply with the shortest possible answer: a name, place, number, or short phrase. " +
	"Do not write a sentence. If the context lacks the answer, reply exactly: unknown."

// TestABRetrievalImprovements is a manual, model-backed A/B benchmark. It is
// skipped unless TG_AB=1 and a real Ollama is reachable, so it never runs in CI.
// It quantifies the lift from the cognee/RAG-research features (dense entity
// seeding, multi-hop decomposition, knowledge-graph fact injection, contextual
// retrieval) on retrieval recall and end-to-end answer accuracy, using a real
// embedder and a real chat model over a corpus padded with distractors.
//
//	TG_AB=1 go test ./server/ -run TestABRetrievalImprovements -v -timeout 40m
//
// Models are overridable: TG_AB_CHAT (default qwen3.5:4b), TG_AB_EMBED (default
// nomic-embed-text), TG_AB_URL (default http://localhost:11434).
func TestABRetrievalImprovements(t *testing.T) {
	if os.Getenv("TG_AB") == "" {
		t.Skip("set TG_AB=1 (and have Ollama running) to run the model-backed A/B benchmark")
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

	corpus := append(append([]rag.Document{}, abDocs...), abDistractors...)
	t.Logf("corpus: %d docs (%d answer-bearing + %d distractors), embed=%s chat=%s",
		len(corpus), len(abDocs), len(abDistractors), embedModel, chatModel)

	// storeA: the plain pipeline, plus an entity graph for the graph-based arms.
	storeA := rag.New(client, rag.Config{Seed: 1, GraphKNN: 6, MinSimilarity: 0.05})
	if err := storeA.Build(ctx, corpus); err != nil {
		t.Fatalf("build A: %v", err)
	}
	ex := entity.NewLLMExtractor(genAdapter{c: client, model: chatModel})
	if err := storeA.BuildEntityGraph(ctx, ex, rag.EntityBuildOptions{Workers: 4, BatchSize: 1}); err != nil {
		t.Fatalf("entity graph: %v", err)
	}
	t.Logf("entity graph: %d entities", storeA.EntityCount())

	// storeC: contextual retrieval on, isolating that index-time lift. No entity
	// graph needed here (contextual retrieval is orthogonal and the contextual arm
	// uses the plain dense+bm25 lane).
	storeC := rag.New(client, rag.Config{Seed: 1, GraphKNN: 6, MinSimilarity: 0.05})
	storeC.SetContextualizer(genAdapter{c: client, model: chatModel})
	if err := storeC.Build(ctx, corpus); err != nil {
		t.Fatalf("build C (contextual): %v", err)
	}

	sA := New(storeA)
	sA.SetGenerator(client, chatModel, embedModel)
	sC := New(storeC)
	sC.SetGenerator(client, chatModel, embedModel)

	base := chatRequest{TopK: 8, MinSim: 0}
	withEntity := base
	withEntity.EntityMix = 0.3
	withDecomp := withEntity
	withDecomp.Decompose = true

	scoreRetrieval := func(name string, srv *Server, st *rag.Store, req chatRequest) {
		var r3, r5, mrr, bothHit float64
		var mh int
		for _, qc := range abCases {
			rq := req
			rq.Query = qc.q
			res, _, err := srv.retrieveForChat(ctx, st, rq, chatModel)
			if err != nil {
				t.Fatalf("%s: retrieve %q: %v", name, qc.q, err)
			}
			ranked := docRank(res)
			rel := setOf(qc.gold)
			r3 += eval.RecallAtK(ranked, rel, 3)
			r5 += eval.RecallAtK(ranked, rel, 5)
			mrr += eval.MRR(ranked, rel)
			if len(qc.gold) == 2 { // multi-hop: did BOTH gold docs make the top 5?
				mh++
				if allIn(ranked, qc.gold, 5) {
					bothHit++
				}
			}
		}
		n := float64(len(abCases))
		t.Logf("%-26s recall@3=%.3f recall@5=%.3f mrr=%.3f bothGold@5=%.3f",
			name, r3/n, r5/n, mrr/n, bothHit/float64(mh))
	}

	t.Log("=== RETRIEVAL (doc-level, n=" + fmt.Sprint(len(abCases)) + ", multi-hop bothGold over " + fmt.Sprint(countMH()) + ") ===")
	scoreRetrieval("baseline dense+bm25", sA, storeA, base)
	scoreRetrieval("+ contextual retrieval", sC, storeC, base)
	scoreRetrieval("+ entity seeding", sA, storeA, withEntity)
	scoreRetrieval("+ entity + decomposition", sA, storeA, withDecomp)

	scoreAnswers := func(name string, srv *Server, st *rag.Store, req chatRequest, useFacts bool) {
		var cover, f1 float64
		for _, qc := range abCases {
			rq := req
			rq.Query = qc.q
			res, _, err := srv.retrieveForChat(ctx, st, rq, chatModel)
			if err != nil {
				t.Fatalf("%s answer retrieve %q: %v", name, qc.q, err)
			}
			var facts []string
			if useFacts {
				facts = graphFacts(st, res)
			}
			prompt := buildChatPrompt(qc.q, res, nil, facts)
			ans, err := genAdapter{c: client, model: chatModel}.Generate(ctx, terseAnswerSystem, prompt)
			if err != nil {
				t.Fatalf("%s generate %q: %v", name, qc.q, err)
			}
			if eval.CoverMatch(ans, qc.answer) {
				cover++
			}
			f1 += eval.AnswerF1(ans, qc.answer)
		}
		n := float64(len(abCases))
		t.Logf("%-34s cover=%.3f f1=%.3f", name, cover/n, f1/n)
	}

	// Graph facts are isolated by holding retrieval fixed (the same base candidate
	// set) and toggling only the injected facts, so the delta is the facts alone
	// and not a side effect of a different retrieval configuration.
	t.Log("=== ANSWER QUALITY (terse, n=" + fmt.Sprint(len(abCases)) + ") ===")
	scoreAnswers("base retrieval, facts OFF", sA, storeA, base, false)
	scoreAnswers("base retrieval, facts ON", sA, storeA, base, true)
	scoreAnswers("contextual retrieval", sC, storeC, base, false)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func countMH() int {
	n := 0
	for _, c := range abCases {
		if len(c.gold) == 2 {
			n++
		}
	}
	return n
}

// docRank collapses retrieved chunks to their document ids, first occurrence
// wins, preserving rank order (the BEIR document-level convention).
func docRank(res []rag.Retrieved) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(res))
	for _, r := range res {
		id := r.Chunk.DocID
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

// allIn reports whether every id in want appears within the first k of ranked.
func allIn(ranked, want []string, k int) bool {
	if k > len(ranked) {
		k = len(ranked)
	}
	top := setOf(ranked[:k])
	for _, w := range want {
		if _, ok := top[w]; !ok {
			return false
		}
	}
	return true
}

func setOf(ids []string) map[string]struct{} {
	m := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		m[id] = struct{}{}
	}
	return m
}
