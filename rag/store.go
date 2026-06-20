// Package rag implements a fast graph RAG store. Chunks are embedded, quantized
// with TurboQuant, and indexed in an HNSW graph for sublinear nearest-neighbor
// search and a BM25 index for lexical matching. Retrieval fuses dense and sparse
// hits, seeds Personalized PageRank over a chunk similarity graph, blends graph
// propagation with direct similarity, and optionally diversifies with MMR. The
// result is context relevant by association as well as by direct match.
package rag

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/graph"
	"github.com/Gaurav-Gosain/turbograph/index"
	"github.com/Gaurav-Gosain/turbograph/lexical"
	"github.com/Gaurav-Gosain/turbograph/quant"
)

// Embedder produces embeddings for a batch of texts, preserving order. These are
// document embeddings: the source-of-truth vectors that are indexed.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// QueryEmbedder is an optional extension of Embedder for asymmetric, instruction
// tuned models that encode a query differently from a document. When the store's
// embedder implements it, retrieval embeds the query through EmbedQuery instead
// of Embed; otherwise the same Embed path is used for both, so plain embedders
// keep working unchanged.
type QueryEmbedder interface {
	EmbedQuery(ctx context.Context, texts []string) ([][]float32, error)
}

// embedQuery routes a query through EmbedQuery when the embedder supports it.
func embedQuery(ctx context.Context, e Embedder, texts []string) ([][]float32, error) {
	if qe, ok := e.(QueryEmbedder); ok {
		return qe.EmbedQuery(ctx, texts)
	}
	return e.Embed(ctx, texts)
}

// Config parameterizes the store. Zero values select sensible defaults.
type Config struct {
	Chunk ChunkConfig
	// Chunker, if set, overrides Chunk.Strategy with a caller-supplied splitter
	// (bring your own). It is not persisted; after loading a store you must set it
	// again to ingest further documents with a custom chunker.
	Chunker Chunker

	// Quantization.
	Bits         int
	ResidualDims int
	Seed         uint64

	// Vector index (HNSW).
	HNSW     index.HNSWConfig
	EfSearch int // query-time search width (default 64)

	// Graph construction.
	GraphKNN         int
	MinSimilarity    float32
	SequentialWeight float32

	// Hybrid lexical fusion. BM25 + RRF is on by default because it reliably
	// improves recall on exact and rare terms; set DisableLexical to turn it off.
	DisableLexical bool
	RRFK           int // reciprocal rank fusion constant (default 60)
}

func (c *Config) withDefaults() {
	if c.Bits == 0 {
		c.Bits = 4
	}
	if c.ResidualDims == 0 {
		c.ResidualDims = 32
	}
	if c.EfSearch == 0 {
		c.EfSearch = 64
	}
	if c.GraphKNN == 0 {
		c.GraphKNN = 12
	}
	if c.MinSimilarity == 0 {
		c.MinSimilarity = 0.5
	}
	if c.SequentialWeight == 0 {
		c.SequentialWeight = 0.6
	}
	if c.RRFK == 0 {
		c.RRFK = lexical.DefaultRRFK
	}
	if c.Chunk.TargetWords == 0 {
		c.Chunk = DefaultChunkConfig()
	}
}

// Store holds the indexed corpus, its vector and lexical indexes, the similarity
// graph, and its community structure. It is safe for concurrent retrieval.
type Store struct {
	mu sync.RWMutex

	cfg      Config
	embedder Embedder

	dim    int
	q      *quant.Quantizer
	hnsw   *index.HNSW
	bm25   *lexical.Index
	chunks []Chunk
	embeds [][]float32         // raw embeddings, the source of truth for rebuilds and MMR
	docSet map[string]struct{} // document ids already ingested, for dedup and resume
	hashes map[[32]byte]string // content hash -> owning doc id, for content-level dedup
	idHash map[string][32]byte // doc id -> content hash, persisted so dedup survives reload

	versions    map[string][]docVersion    // doc id -> content history, oldest first
	docMeta     map[string]json.RawMessage // doc id -> arbitrary user metadata (raw JSON)
	commSummary map[int]string             // community label -> generated summary (global queries)

	edges        []edgeRec
	indexedUpTo  int  // chunks for which similarity edges have been discovered
	needsRebuild bool // vector and lexical indexes are stale (a document was removed)
	g            *graph.Graph
	comm         *graph.Communities

	// Optional entity-relationship knowledge graph (GraphRAG style). Built on
	// demand from an LLM extractor; nil until BuildEntityGraph runs.
	eg       *entity.Graph
	entList  []entity.Entity // sorted, index == node id
	entIndex map[string]int  // canonical entity name -> node id
	entCSR   *graph.Graph
	entComm  *graph.Communities
}

type edgeRec struct {
	a, b int
	w    float32
}

// Document is an input document.
type Document struct {
	ID   string
	Text string
	// Meta is arbitrary user metadata attached to the document. It is stored as
	// canonical JSON, propagated to every chunk of the document, and returned with
	// each retrieved result, so callers can decide how to use it (parse it, filter
	// on it, or feed selected fields to the model). nil means no metadata.
	Meta map[string]any
	// Kind and ImageRef mark an image-derived document: Text is then a caption of
	// the image, Kind is "image", and ImageRef is the asset id of the source image.
	// Both are empty for an ordinary text document.
	Kind     string
	ImageRef string
}

// New creates an empty store.
func New(embedder Embedder, cfg Config) *Store {
	cfg.withDefaults()
	return &Store{cfg: cfg, embedder: embedder}
}

// Len returns the number of indexed chunks.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

// Chunk returns the chunk at ordinal i.
func (s *Store) Chunk(i int) Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.chunks[i]
}

// Communities returns the detected community structure (may be nil before build).
func (s *Store) Communities() *graph.Communities {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.comm
}

// Build indexes the documents from scratch, replacing any previous contents.
func (s *Store) Build(ctx context.Context, docs []Document) error {
	chunks := make([]Chunk, 0, len(docs))
	for _, d := range docs {
		chunks = append(chunks, s.chunkDoc(d)...)
	}
	if len(chunks) == 0 {
		return fmt.Errorf("rag: no chunks produced from %d documents", len(docs))
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	vecs, err := s.embedder.Embed(ctx, texts)
	if err != nil {
		return err
	}
	if len(vecs) != len(chunks) || len(vecs[0]) == 0 {
		return fmt.Errorf("rag: embedder returned %d vectors of dim %d", len(vecs), dimOf(vecs))
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.dim = len(vecs[0])
	s.initIndexes()
	s.chunks = s.chunks[:0]
	s.embeds = s.embeds[:0]
	s.edges = s.edges[:0]
	s.indexedUpTo = 0
	s.docSet = make(map[string]struct{})
	s.hashes = make(map[[32]byte]string)
	s.idHash = make(map[string][32]byte)
	s.versions = make(map[string][]docVersion)
	s.docMeta = make(map[string]json.RawMessage)
	s.commSummary = nil // a fresh build invalidates any community summaries
	if err := s.appendChunksLocked(chunks, vecs); err != nil {
		return err
	}
	perDoc := make(map[string]int, len(docs))
	for _, c := range chunks {
		perDoc[c.DocID]++
	}
	for _, d := range docs {
		h := contentHash(d.Text)
		s.recordHashLocked(d.ID, h)
		s.recordVersionLocked(d.ID, h, d.Text, perDoc[d.ID])
		s.recordMetaLocked(d.ID, d.Meta)
	}
	s.reindexLocked()
	return nil
}

// AddDocuments incrementally indexes documents. Re-adding identical content is a
// no-op (deduped by content hash). A document whose id already exists but whose
// content has changed is updated in place: it is re-chunked and only the chunks
// whose text actually changed are re-embedded, the rest reuse their existing
// embeddings. New documents are added directly. The graph and communities are
// then refreshed.
func (s *Store) AddDocuments(ctx context.Context, docs []Document) error {
	docs = s.newDocuments(docs)
	if len(docs) == 0 {
		return nil
	}
	preps := make([]prepared, 0, len(docs))
	for _, d := range docs {
		p, err := s.prepareDoc(ctx, d)
		if err != nil {
			return err
		}
		if len(p.chunks) > 0 {
			preps = append(preps, p)
		}
	}
	if len(preps) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	changed := false
	for _, p := range preps {
		if s.applyPreparedLocked(p) {
			changed = true
		}
	}
	if changed {
		s.commSummary = nil // graph changed; community summaries are now stale
	}
	s.reindexLocked()
	return nil
}

// contentHash is the SHA-256 of a document's text, used for content-level dedup.
func contentHash(text string) [32]byte { return sha256.Sum256([]byte(text)) }

// newDocuments returns the documents worth processing: those whose content is not
// already present. Identical content (same hash) is skipped as a duplicate
// whether the id matches or not. A document whose id exists but whose content has
// changed is kept, so it can be applied as an update. Within-batch duplicates are
// collapsed, keeping the first occurrence of each id.
func (s *Store) newDocuments(docs []Document) []Document {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := docs[:0:0]
	seenID := make(map[string]struct{}, len(docs))
	seenHash := make(map[[32]byte]struct{}, len(docs))
	for _, d := range docs {
		h := contentHash(d.Text)
		if _, ok := s.hashes[h]; ok {
			continue // this exact content is already indexed
		}
		if _, dup := seenHash[h]; dup {
			continue
		}
		if _, dup := seenID[d.ID]; dup {
			continue
		}
		seenID[d.ID] = struct{}{}
		seenHash[h] = struct{}{}
		out = append(out, d)
	}
	return out
}

// recordHashLocked associates a document id with its content hash. The caller
// must hold the write lock.
func (s *Store) recordHashLocked(id string, h [32]byte) {
	if s.hashes == nil {
		s.hashes = make(map[[32]byte]string)
		s.idHash = make(map[string][32]byte)
	}
	s.hashes[h] = id
	s.idHash[id] = h
}

// ContentOwner returns the id of the document with the given content hash, if any.
func (s *Store) ContentOwner(h [32]byte) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	id, ok := s.hashes[h]
	return id, ok
}

// HasDoc reports whether a document id has been ingested.
func (s *Store) HasDoc(id string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.docSet[id]
	return ok
}

// DocCount returns the number of distinct documents ingested.
func (s *Store) DocCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docSet)
}

// DocInfo summarizes one ingested document.
type DocInfo struct {
	ID     string `json:"id"`
	Chunks int    `json:"chunks"`
	Bytes  int    `json:"bytes"` // total chunk text length
}

// Documents lists the ingested documents in first-seen order, with each one's
// chunk count and size. It lets a client reconstruct the document list after
// loading a store from disk, where the in-memory client has no record of what was
// ingested in a previous session.
func (s *Store) Documents() []DocInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	idx := make(map[string]int, len(s.docSet))
	var out []DocInfo
	for _, c := range s.chunks {
		i, ok := idx[c.DocID]
		if !ok {
			idx[c.DocID] = len(out)
			out = append(out, DocInfo{ID: c.DocID})
			i = len(out) - 1
		}
		out[i].Chunks++
		out[i].Bytes += len(c.Text)
	}
	return out
}

// Embedder returns the embedder the store was created with.
func (s *Store) Embedder() Embedder { return s.embedder }

// ChunkDocument splits a document using the store's chunk configuration. It is
// exposed so an ingestion engine can chunk and embed off the write path.
func (s *Store) ChunkDocument(d Document) []Chunk {
	return s.chunkDoc(d)
}

// Config returns the store's configuration (the custom Chunker, if any, is
// omitted as it does not round-trip).
func (s *Store) Config() Config {
	c := s.cfg
	c.Chunker = nil
	return c
}

// AddEmbedded indexes already-embedded chunks without rebuilding the graph. The
// caller is expected to call Reindex once after a batch of AddEmbedded calls.
// This is the low-level entry point used by the bulk ingestion engine so that
// embedding (the slow part) happens off the write lock and graph reconstruction
// is deferred to the end.
func (s *Store) AddEmbedded(chunks []Chunk, vecs [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hnsw == nil {
		s.dim = len(vecs[0])
		s.initIndexes()
	}
	if len(vecs[0]) != s.dim {
		return fmt.Errorf("rag: embedding dim %d does not match store dim %d", len(vecs[0]), s.dim)
	}
	return s.appendChunksLocked(chunks, vecs)
}

// Reindex discovers similarity edges for any chunks added since the last reindex
// and rebuilds the graph and communities. It is cheap to call once after a bulk
// ingestion and idempotent if nothing changed.
func (s *Store) Reindex() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reindexLocked()
}

func (s *Store) initIndexes() {
	s.q = quant.New(quant.Config{
		Dim:          s.dim,
		Bits:         s.cfg.Bits,
		ResidualDims: s.cfg.ResidualDims,
		Seed:         s.cfg.Seed,
	})
	s.hnsw = index.NewHNSW(s.dim, s.q, s.cfg.HNSW)
	s.bm25 = lexical.New(lexical.DefaultConfig())
}

// appendChunksLocked adds chunks and their embeddings to every index and records
// the document ids. It does NOT rebuild the graph; call reindexLocked afterward.
// The caller must hold the write lock.
func (s *Store) appendChunksLocked(chunks []Chunk, vecs [][]float32) error {
	if s.docSet == nil {
		s.docSet = make(map[string]struct{})
	}
	for i, c := range chunks {
		if len(vecs[i]) != s.dim {
			return fmt.Errorf("rag: inconsistent embedding dim at %d", i)
		}
		s.hnsw.Add(c.ID, vecs[i])
		s.bm25.Add(c.ID, c.Text)
		s.chunks = append(s.chunks, c)
		s.embeds = append(s.embeds, vecs[i])
		s.docSet[c.DocID] = struct{}{}
	}
	// Finalize the lexical index under the write lock so concurrent readers never
	// trigger its lazy finalization (which mutates shared state).
	s.bm25.Finalize()
	return nil
}

// reindexLocked brings the graph up to date. If a document was removed or updated
// the vector and lexical indexes are rebuilt from the source-of-truth arrays
// first, then all edges are rediscovered. Otherwise only the edges for newly
// added chunks are discovered. The caller must hold the write lock.
func (s *Store) reindexLocked() {
	if s.needsRebuild {
		s.rebuildIndexesLocked()
		s.needsRebuild = false
		s.edges = s.edges[:0]
		s.indexedUpTo = 0
	}
	if s.indexedUpTo >= len(s.chunks) {
		if s.g == nil || s.indexedUpTo == 0 {
			s.rebuildGraph()
		}
		return
	}
	s.discoverEdges(s.indexedUpTo)
	s.indexedUpTo = len(s.chunks)
	s.rebuildGraph()
}

// discoverEdges finds similarity neighbors for every chunk from ordinal `from`
// onward and records undirected edges. Only the new nodes are searched, so an
// incremental add does not re-scan the whole corpus. The per-node searches are
// independent and read-only on the index, so they run in parallel.
func (s *Store) discoverEdges(from int) {
	n := len(s.chunks)
	count := n - from
	if count <= 0 {
		return
	}
	k := s.cfg.GraphKNN + 1 // +1 because a chunk matches itself
	ef := maxInt(s.cfg.EfSearch, k)

	workers := runtime.GOMAXPROCS(0)
	if workers > count {
		workers = count
	}
	perWorker := make([][]edgeRec, workers)
	var wg sync.WaitGroup
	chunkSize := (count + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := from + w*chunkSize
		hi := minInt(lo+chunkSize, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			var local []edgeRec
			for i := lo; i < hi; i++ {
				hits := s.hnsw.Search(s.embeds[i], k, ef)
				for _, h := range hits {
					j, ok := s.hnsw.Ord(h.ID)
					if !ok || j == i || h.Score < s.cfg.MinSimilarity {
						continue
					}
					local = append(local, edgeRec{a: i, b: j, w: h.Score})
				}
			}
			perWorker[w] = local
		}(w, lo, hi)
	}
	wg.Wait()
	for _, e := range perWorker {
		s.edges = append(s.edges, e...)
	}
}

// rebuildGraph reconstructs the CSR graph and community structure from the
// recorded edges plus structural document-order edges.
func (s *Store) rebuildGraph() {
	n := len(s.chunks)
	b := graph.NewBuilder(n)
	for _, e := range s.edges {
		b.AddEdge(e.a, e.b, e.w)
	}
	for i := 1; i < n; i++ {
		if s.chunks[i].DocID == s.chunks[i-1].DocID {
			b.AddEdge(i-1, i, s.cfg.SequentialWeight)
		}
	}
	s.g = b.Build()
	s.comm = graph.DetectCommunities(s.g, graph.CommunityOpts{Seed: s.cfg.Seed})
}

// DefaultLexicalWeight is the BM25 weight used when RetrieveParams.LexicalWeight
// is unset and the store indexes BM25. It is deliberately small: a light lexical
// boost on top of the dense ranking improves keyword and entity matching while
// staying near-neutral on dense-dominant corpora. See docs/benchmarks.md.
const DefaultLexicalWeight = 0.25

// RetrieveParams controls a retrieval.
type RetrieveParams struct {
	TopK  int // number of chunks to return
	SeedK int // dense/sparse hits used to seed PageRank (default 3*TopK)
	// GraphMix is how strongly the personalized-PageRank graph signal is added on
	// top of direct relevance. The score is relevance + GraphMix*pagerank, so the
	// graph can lift an associated chunk (one hop from a strong hit) into the
	// results without demoting a strong direct match. It is off by default (0):
	// graph reranking measurably lowers precision on standard retrieval, so it is
	// opt-in for thematic or associative queries. Negative is clamped to 0.
	GraphMix float32
	// LexicalWeight is how strongly the BM25 score is added to the dense cosine in
	// the direct relevance term (relevance = dense + LexicalWeight*bm25, both
	// normalized to their per-query max). It preserves the dense ranking and lets
	// an exact lexical match lift a chunk, which helps on keyword- and entity-heavy
	// corpora and is near-neutral where dense already dominates. 0 is pure dense; a
	// negative value is treated as 0. Ignored when the store has lexical disabled.
	LexicalWeight float32
	MMRLambda     float32 // MMR relevance/diversity tradeoff; 0 disables diversification
	EntityMix     float32 // weight of the entity-graph signal in [0,1]; 0 ignores it
	// PRF enables pseudo-relevance feedback: an initial dense search of this many
	// chunks is run, their vectors are averaged into the query (Rocchio in
	// embedding space), and the expanded query drives retrieval. It surfaces
	// chunks that share the topic's vocabulary but not the query's exact words,
	// which helps recall on multi-hop and underspecified queries. 0 disables it.
	PRF int
	// PRFWeight is how strongly the feedback centroid is mixed into the query
	// (Rocchio beta). The original query is always kept at full weight so feedback
	// refines rather than replaces it. Defaults to 0.5 when PRF is set.
	PRFWeight float32
	Filter    func(Chunk) bool // optional metadata filter
	PPR       graph.PPRParams
}

// Retrieved is a scored chunk.
type Retrieved struct {
	Chunk      Chunk
	Score      float32         // blended retrieval score
	Similarity float32         // direct cosine similarity to the query (0 if not a seed)
	Meta       json.RawMessage // the source document's metadata, if any
}

// Retrieve runs hybrid graph retrieval for the query.
func (s *Store) Retrieve(ctx context.Context, query string, p RetrieveParams) ([]Retrieved, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.hnsw == nil || len(s.chunks) == 0 {
		return nil, fmt.Errorf("rag: store is empty")
	}
	if p.TopK <= 0 {
		p.TopK = 8
	}
	if p.SeedK <= 0 {
		p.SeedK = 3 * p.TopK
	}
	// The graph is opt-in. Benchmarks (see docs/benchmarks.md) show similarity-graph
	// reranking lowers retrieval precision on standard single-hop and multi-hop
	// tasks, so the default (zero) ranks by hybrid relevance alone; a positive
	// GraphMix adds the graph boost for thematic or associative use.
	if p.GraphMix < 0 {
		p.GraphMix = 0
	}
	if p.PPR.Damping == 0 {
		p.PPR = graph.DefaultPPR()
	}

	qv, err := embedQuery(ctx, s.embedder, []string{query})
	if err != nil {
		return nil, err
	}

	// Pseudo-relevance feedback: expand the query vector toward the centroid of
	// its top hits before the main retrieval. Lexical (BM25) seeding still uses the
	// original query text, so feedback only nudges the dense side.
	if p.PRF > 0 {
		qv[0] = s.prfExpand(qv[0], p.PRF, p.PRFWeight)
	}

	// Direct relevance blends the dense cosine (magnitude preserved) with the BM25
	// score additively: relevance = dense + LexicalWeight * bm25. This keeps the
	// strong dense ranking that rank fusion would flatten, while letting an exact
	// lexical match lift a chunk. It is independent of the graph, so lexical
	// matching contributes even with the graph off. seeds (RRF) is used only to
	// seed PageRank; sim is also the Similarity field and abstention signal.
	seeds, sim, bm25 := s.seedScores(query, qv[0], p.SeedK)
	// Lexical fusion is on by default when the store indexes BM25: an unset (zero)
	// LexicalWeight uses DefaultLexicalWeight, a modest value that helps keyword and
	// entity-heavy corpora and is near-neutral where dense already dominates. A
	// negative value forces pure dense; a store built with DisableLexical has no
	// BM25 scores, so the term is inert regardless.
	switch {
	case p.LexicalWeight == 0 && !s.cfg.DisableLexical:
		p.LexicalWeight = DefaultLexicalWeight
	case p.LexicalWeight < 0:
		p.LexicalWeight = 0
	}
	var simMax, bm25Max float32
	for _, v := range sim {
		if v > simMax {
			simMax = v
		}
	}
	for _, v := range bm25 {
		if v > bm25Max {
			bm25Max = v
		}
	}

	// The graph boost is computed only when asked for: PageRank over the whole
	// graph is the expensive part of retrieval, and the default path does not use
	// it. The entity signal is likewise optional.
	var ppr []float32
	var pprMax float32
	if p.GraphMix > 0 {
		ppr = s.g.PersonalizedPageRank(seeds, p.PPR)
		pprMax = maxF(ppr)
	}
	var escore map[int]float32
	if p.EntityMix > 0 && s.entCSR != nil {
		escore = s.entityChunkScores(query)
	}

	type sc struct {
		ord int
		val float32
	}
	var scored []sc
	rel := func(ord int) float32 {
		var d float32
		if simMax > 0 {
			d = sim[ord] / simMax
		}
		if p.LexicalWeight > 0 && bm25Max > 0 {
			d += p.LexicalWeight * bm25[ord] / bm25Max
		}
		return d
	}
	if ppr == nil && escore == nil {
		// Fast path: pure hybrid. Only seeds (dense or BM25 hits) have a nonzero
		// score, so rank those directly and skip the full-corpus scan and the
		// PageRank pass.
		scored = make([]sc, 0, len(seeds))
		for ord := range seeds {
			if p.Filter != nil && !p.Filter(s.chunks[ord]) {
				continue
			}
			if v := rel(ord); v > 0 {
				scored = append(scored, sc{ord, v})
			}
		}
	} else {
		// Graph and/or entity signal active: every chunk can receive propagated
		// mass, so scan the corpus and add the boosts on top of relevance.
		scored = make([]sc, 0, len(s.chunks))
		for i := range s.chunks {
			if p.Filter != nil && !p.Filter(s.chunks[i]) {
				continue
			}
			var g float32
			if pprMax > 0 {
				g = ppr[i] / pprMax
			}
			// Additive, not convex: the graph adds an associative boost on top of
			// direct relevance rather than trading relevance away for it. A strong
			// direct hit keeps its rank; a graph-associated chunk (low relevance, high
			// g) is lifted from the tail. A convex blend let high-centrality chunks
			// displace genuinely relevant ones, which collapsed precision.
			val := rel(i) + p.GraphMix*g
			if escore != nil {
				val = (1-p.EntityMix)*val + p.EntityMix*escore[i]
			}
			if val > 0 {
				scored = append(scored, sc{i, val})
			}
		}
	}
	sort.Slice(scored, func(a, b int) bool { return scored[a].val > scored[b].val })

	// Optional MMR diversification over a candidate pool a few times TopK.
	if p.MMRLambda > 0 && p.MMRLambda < 1 && len(scored) > p.TopK {
		pool := minInt(len(scored), maxInt(p.TopK*4, p.TopK))
		rel := make([]float32, pool)
		vecs := make([][]float32, pool)
		// MMR trades relevance against a cosine redundancy term in [0,1]. Normalize
		// the relevance to the same scale so MMRLambda means the same thing
		// regardless of the blend's absolute magnitude (the additive graph boost can
		// push val above 1).
		relMax := scored[0].val
		for i := 0; i < pool; i++ {
			r := scored[i].val
			if relMax > 0 {
				r /= relMax
			}
			rel[i] = r
			vecs[i] = s.hnsw.Vector(scored[i].ord) // normalized
		}
		order := mmrRerank(rel, vecs, p.MMRLambda, p.TopK)
		out := make([]Retrieved, len(order))
		for i, idx := range order {
			ord := scored[idx].ord
			c := s.chunks[ord]
			out[i] = Retrieved{Chunk: c, Score: scored[idx].val, Similarity: sim[ord], Meta: s.docMeta[c.DocID]}
		}
		return out, nil
	}

	if len(scored) > p.TopK {
		scored = scored[:p.TopK]
	}
	out := make([]Retrieved, len(scored))
	for i, e := range scored {
		c := s.chunks[e.ord]
		out[i] = Retrieved{Chunk: c, Score: e.val, Similarity: sim[e.ord], Meta: s.docMeta[c.DocID]}
	}
	return out, nil
}

// seedScores runs the dense and sparse searches and returns three things: the
// RRF-fused seed vector for PageRank (ordinal -> mass), the per-ordinal dense
// cosine similarity (the magnitude signal, kept for the relevance score and the
// Similarity field), and the per-ordinal BM25 score (the lexical signal). Keeping
// the dense cosine and BM25 separate lets the caller blend them by score, which
// preserves the dense magnitude that rank fusion alone would discard.
func (s *Store) seedScores(query string, qv []float32, seedK int) (seeds, sim, bm25 map[int]float32) {
	dense := s.hnsw.Search(qv, seedK, maxInt(s.cfg.EfSearch, seedK))
	sim = make(map[int]float32, len(dense))
	for _, h := range dense {
		if ord, ok := s.hnsw.Ord(h.ID); ok {
			sim[ord] = h.Score
		}
	}

	var ranked []lexical.Result
	bm25 = map[int]float32{}
	if !s.cfg.DisableLexical {
		denseR := make([]lexical.Result, len(dense))
		for i, h := range dense {
			denseR[i] = lexical.Result{ID: h.ID, Score: h.Score}
		}
		sparse := s.bm25.Search(query, seedK)
		for _, r := range sparse {
			if ord, ok := s.hnsw.Ord(r.ID); ok {
				bm25[ord] = r.Score
			}
		}
		ranked = lexical.RRF(s.cfg.RRFK, denseR, sparse)
	} else {
		ranked = make([]lexical.Result, len(dense))
		for i, h := range dense {
			ranked[i] = lexical.Result{ID: h.ID, Score: h.Score}
		}
	}

	seeds = make(map[int]float32, len(ranked))
	for _, r := range ranked {
		if ord, ok := s.hnsw.Ord(r.ID); ok && r.Score > 0 {
			seeds[ord] = r.Score
		}
	}
	return seeds, sim, bm25
}

// prfExpand returns a unit query vector moved toward the centroid of its top-k
// nearest chunks (Rocchio with the original query fixed at weight 1). It is a
// pure vector operation, so it adds one extra ANN search and no model call. The
// original query dominates unless weight is large, which keeps query drift in
// check; a noisy top-k can still mislead it, so callers enable it deliberately.
func (s *Store) prfExpand(q []float32, k int, weight float32) []float32 {
	if weight <= 0 {
		weight = 0.5
	}
	hits := s.hnsw.Search(q, k, maxInt(s.cfg.EfSearch, k))
	if len(hits) == 0 {
		return q
	}
	out := make([]float32, len(q))
	copy(out, q)
	scale := weight / float32(len(hits))
	for _, h := range hits {
		ord, ok := s.hnsw.Ord(h.ID)
		if !ok {
			continue
		}
		v := s.hnsw.Vector(ord) // normalized document vector
		for i := range out {
			out[i] += scale * v[i]
		}
	}
	var n float64
	for _, x := range out {
		n += float64(x) * float64(x)
	}
	if n > 0 {
		inv := float32(1 / math.Sqrt(n))
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

func maxF(v []float32) float32 {
	var m float32
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}

func dimOf(v [][]float32) int {
	if len(v) == 0 {
		return 0
	}
	return len(v[0])
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
