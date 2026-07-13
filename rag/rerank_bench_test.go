package rag

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/ollama"
)

// TestRerankBlendBenchmark measures the reranker's blend policy against a real
// model on a labelled corpus, so the change from a fixed model weight to a
// position-aware one is grounded rather than argued. Skipped unless TG_RERANK=1.
//
//	TG_RERANK=1 go test ./rag/ -run TestRerankBlendBenchmark -v -timeout 30m
//
// The store over-retrieves a candidate pool (as the server does) and each policy
// reranks it down to k. A reranker that hijacks the head with a noisy judgement
// shows up as a drop in recall and MRR against the gold documents.
func TestRerankBlendBenchmark(t *testing.T) {
	if os.Getenv("TG_RERANK") == "" {
		t.Skip("set TG_RERANK=1 (and have Ollama running) to run the rerank blend benchmark")
	}
	embedModel := envOrLean("TG_RERANK_EMBED", "nomic-embed-text")
	chatModel := envOrLean("TG_RERANK_CHAT", "qwen3.5:4b")

	client := ollama.New()
	client.SetEmbedModel(embedModel)
	gen := boundGen{c: client, model: chatModel}
	ctx := context.Background()

	docs, cases := rerankCorpus()
	s := New(client, Config{Seed: 1, GraphKNN: 6, MinSimilarity: 0.05})
	if err := s.Build(ctx, docs); err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Logf("corpus: %d docs, %d queries, embed=%s reranker=%s", len(docs), len(cases), embedModel, chatModel)

	const (
		k    = 5  // final results scored
		pool = 20 // candidates handed to the reranker, as the server does
	)
	policies := []struct {
		name   string
		weight func(rank, n int) float32
	}{
		{"no rerank        ", nil},
		{"fixed w=0.7 (old)", func(_, _ int) float32 { return 0.7 }},
		{"position-aware   ", blendWeight},
	}
	for _, p := range policies {
		var recall, mrr float64
		// The defect this change targets is the reranker demoting a correct top hit.
		// Count that directly: how often a policy breaks a top-1 that retrieval got
		// right, and how often it fixes a top-1 that retrieval got wrong.
		var broke, fixed int
		for _, c := range cases {
			res, err := s.Retrieve(ctx, c.Query, RetrieveParams{TopK: pool})
			if err != nil {
				t.Fatalf("retrieve: %v", err)
			}
			rel := relevantOf(c.Relevant)
			baseTop1Correct := len(res) > 0 && isRelevant(res[0], rel)

			if p.weight != nil {
				res = rerankWith(ctx, gen, c.Query, res, k, p.weight)
			} else if len(res) > k {
				res = res[:k]
			}
			newTop1Correct := len(res) > 0 && isRelevant(res[0], rel)
			switch {
			case baseTop1Correct && !newTop1Correct:
				broke++
			case !baseTop1Correct && newTop1Correct:
				fixed++
			}
			ranked := docRankOf(res)
			recall += eval.RecallAtK(ranked, rel, k)
			mrr += eval.MRR(ranked, rel)
		}
		n := float64(len(cases))
		t.Logf("%s  recall@%d=%.3f  mrr=%.3f  broke-correct-top1=%d  fixed-wrong-top1=%d",
			p.name, k, recall/n, mrr/n, broke, fixed)
	}
}

func isRelevant(r Retrieved, rel map[string]struct{}) bool {
	_, ok := rel[r.Chunk.DocID]
	return ok
}

// boundGen binds a model to the Ollama client so it satisfies Generator.
type boundGen struct {
	c     *ollama.Client
	model string
}

func (g boundGen) Generate(ctx context.Context, system, prompt string) (string, error) {
	return g.c.Generate(ctx, g.model, system, prompt)
}

func docRankOf(res []Retrieved) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(res))
	for _, r := range res {
		if _, ok := seen[r.Chunk.DocID]; ok {
			continue
		}
		seen[r.Chunk.DocID] = struct{}{}
		out = append(out, r.Chunk.DocID)
	}
	return out
}

func relevantOf(ids []string) map[string]struct{} {
	m := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		m[id] = struct{}{}
	}
	return m
}

// rerankCorpus is a labelled corpus with distractors that are lexically close to
// the queries, which is what gives a reranker something to get wrong.
func rerankCorpus() ([]Document, []eval.Case) {
	docs := []Document{
		{ID: "hnsw", Text: "HNSW is the vector index turbograph uses for approximate nearest neighbour search. It builds a layered proximity graph so a query reaches its closest chunk embeddings in logarithmic time."},
		{ID: "bm25", Text: "BM25 is the lexical ranking function in turbograph's hybrid retrieval. It scores a chunk by term frequency and inverse document frequency, rewarding an exact keyword match."},
		{ID: "pagerank", Text: "Personalized PageRank spreads relevance across the chunk similarity graph, lifting a chunk one hop from a strong match even when the query does not mention it."},
		{ID: "quant", Text: "TurboQuant compresses each embedding into a compact code while keeping a low variance score estimate for ranking."},
		{ID: "communities", Text: "Label propagation detects communities of closely related chunks, and each community can be summarized once for corpus wide questions."},
		{ID: "entity", Text: "The entity knowledge graph extracts typed entities and relationships from each chunk so two passages connect because they mention the same thing."},
		{ID: "chunking", Text: "Documents are split into chunks before embedding, with recursive, markdown, sentence and word strategies available."},
		{ID: "contextual", Text: "Contextual retrieval prepends a generated sentence situating each chunk in its document, which helps when later chunks lose the entity name."},
	}
	// Distractors that share vocabulary with the queries but do not answer them.
	distract := []struct{ id, text string }{
		{"d_index", "An index is a data structure that speeds up lookups. Many systems build an index over their records to avoid a full scan."},
		{"d_graph", "A graph is a set of nodes joined by edges. Graphs appear throughout computer science, from routing to dependency resolution."},
		{"d_search", "Search is the task of finding items matching a request. Search quality is usually measured with recall and precision."},
		{"d_score", "A score ranks candidates against each other. Scores are often normalized before they are compared or combined."},
		{"d_vector", "A vector is an ordered list of numbers. Vectors can be added, scaled, and compared by the angle between them."},
		{"d_word", "A word is a unit of language. Word frequency varies enormously across a corpus, which complicates naive counting."},
		{"d_model", "A model approximates a process. Models are fit to data and then used to make predictions about unseen cases."},
		{"d_query", "A query expresses an information need. Users often phrase the same need in very different words."},
		{"d_text", "Text is a sequence of characters. Text processing includes tokenization, normalization, and segmentation."},
		{"d_memory", "Memory holds data a program is working on. Access patterns determine whether a workload is cache friendly."},
		{"d_disk", "Disk stores data durably. Sequential reads are far faster than random reads on spinning media."},
		{"d_cpu", "A CPU executes instructions. Modern cores use vector instructions to process several values at once."},
	}
	for _, d := range distract {
		docs = append(docs, Document{ID: d.id, Text: d.text})
	}
	cases := []eval.Case{
		{Query: "which index structure finds nearest neighbours in logarithmic time", Relevant: []string{"hnsw"}},
		{Query: "how does the lexical ranking function reward an exact keyword match", Relevant: []string{"bm25"}},
		{Query: "what spreads relevance to a chunk one hop from a strong match", Relevant: []string{"pagerank"}},
		{Query: "how are embeddings compressed into a compact code", Relevant: []string{"quant"}},
		{Query: "how are clusters of related chunks detected and summarized", Relevant: []string{"communities"}},
		{Query: "what connects two passages that mention the same thing", Relevant: []string{"entity"}},
		{Query: "what strategies split a document before embedding", Relevant: []string{"chunking"}},
		{Query: "what situates a chunk in its document to help retrieval", Relevant: []string{"contextual"}},
	}
	return docs, cases
}

var _ = fmt.Sprintf // keep fmt imported if the logging above changes
