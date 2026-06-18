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
	"fmt"
	"runtime"
	"sort"
	"sync"

	"github.com/Gaurav-Gosain/turbograph/graph"
	"github.com/Gaurav-Gosain/turbograph/index"
	"github.com/Gaurav-Gosain/turbograph/lexical"
	"github.com/Gaurav-Gosain/turbograph/quant"
)

// Embedder produces embeddings for a batch of texts, preserving order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Config parameterizes the store. Zero values select sensible defaults.
type Config struct {
	Chunk ChunkConfig

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

	edges        []edgeRec
	indexedUpTo  int  // chunks for which similarity edges have been discovered
	needsRebuild bool // vector and lexical indexes are stale (a document was removed)
	g            *graph.Graph
	comm         *graph.Communities
}

type edgeRec struct {
	a, b int
	w    float32
}

// Document is an input document.
type Document struct {
	ID   string
	Text string
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
		chunks = append(chunks, chunkDocument(d.ID, d.Text, s.cfg.Chunk)...)
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
	if err := s.appendChunksLocked(chunks, vecs); err != nil {
		return err
	}
	for _, d := range docs {
		s.recordHashLocked(d.ID, contentHash(d.Text))
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
	for _, p := range preps {
		s.applyPreparedLocked(p)
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

// Embedder returns the embedder the store was created with.
func (s *Store) Embedder() Embedder { return s.embedder }

// ChunkDocument splits a document using the store's chunk configuration. It is
// exposed so an ingestion engine can chunk and embed off the write path.
func (s *Store) ChunkDocument(d Document) []Chunk {
	return chunkDocument(d.ID, d.Text, s.cfg.Chunk)
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

// RetrieveParams controls a retrieval.
type RetrieveParams struct {
	TopK      int              // number of chunks to return
	SeedK     int              // dense/sparse hits used to seed PageRank (default 3*TopK)
	GraphMix  float32          // weight of PageRank vs direct similarity in [0,1] (default 0.6)
	MMRLambda float32          // MMR relevance/diversity tradeoff; 0 disables diversification
	Filter    func(Chunk) bool // optional metadata filter
	PPR       graph.PPRParams
}

// Retrieved is a scored chunk.
type Retrieved struct {
	Chunk      Chunk
	Score      float32 // blended retrieval score
	Similarity float32 // direct cosine similarity to the query (0 if not a seed)
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
	if p.GraphMix == 0 {
		p.GraphMix = 0.6
	}
	if p.PPR.Damping == 0 {
		p.PPR = graph.DefaultPPR()
	}

	qv, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}

	seeds, sim := s.seedScores(query, qv[0], p.SeedK)
	ppr := s.g.PersonalizedPageRank(seeds, p.PPR)

	pprMax := maxF(ppr)
	var simMax float32
	for _, v := range sim {
		if v > simMax {
			simMax = v
		}
	}

	type sc struct {
		ord int
		val float32
	}
	scored := make([]sc, 0, len(s.chunks))
	for i := range s.chunks {
		if p.Filter != nil && !p.Filter(s.chunks[i]) {
			continue
		}
		var g float32
		if pprMax > 0 {
			g = ppr[i] / pprMax
		}
		var d float32
		if simMax > 0 {
			d = sim[i] / simMax
		}
		val := p.GraphMix*g + (1-p.GraphMix)*d
		if val > 0 {
			scored = append(scored, sc{i, val})
		}
	}
	sort.Slice(scored, func(a, b int) bool { return scored[a].val > scored[b].val })

	// Optional MMR diversification over a candidate pool a few times TopK.
	if p.MMRLambda > 0 && p.MMRLambda < 1 && len(scored) > p.TopK {
		pool := minInt(len(scored), maxInt(p.TopK*4, p.TopK))
		rel := make([]float32, pool)
		vecs := make([][]float32, pool)
		for i := 0; i < pool; i++ {
			rel[i] = scored[i].val
			vecs[i] = s.hnsw.Vector(scored[i].ord) // normalized
		}
		order := mmrRerank(rel, vecs, p.MMRLambda, p.TopK)
		out := make([]Retrieved, len(order))
		for i, idx := range order {
			ord := scored[idx].ord
			out[i] = Retrieved{Chunk: s.chunks[ord], Score: scored[idx].val, Similarity: sim[ord]}
		}
		return out, nil
	}

	if len(scored) > p.TopK {
		scored = scored[:p.TopK]
	}
	out := make([]Retrieved, len(scored))
	for i, e := range scored {
		out[i] = Retrieved{Chunk: s.chunks[e.ord], Score: e.val, Similarity: sim[e.ord]}
	}
	return out, nil
}

// seedScores produces the PageRank seed vector (ordinal -> mass) and a per-ordinal
// direct similarity map. With hybrid enabled, dense and sparse rankings are fused
// by reciprocal rank fusion; otherwise dense cosine seeds directly.
func (s *Store) seedScores(query string, qv []float32, seedK int) (map[int]float32, map[int]float32) {
	dense := s.hnsw.Search(qv, seedK, maxInt(s.cfg.EfSearch, seedK))
	sim := make(map[int]float32, len(dense))
	for _, h := range dense {
		if ord, ok := s.hnsw.Ord(h.ID); ok {
			sim[ord] = h.Score
		}
	}

	var ranked []lexical.Result
	if !s.cfg.DisableLexical {
		denseR := make([]lexical.Result, len(dense))
		for i, h := range dense {
			denseR[i] = lexical.Result{ID: h.ID, Score: h.Score}
		}
		sparse := s.bm25.Search(query, seedK)
		ranked = lexical.RRF(s.cfg.RRFK, denseR, sparse)
	} else {
		ranked = make([]lexical.Result, len(dense))
		for i, h := range dense {
			ranked[i] = lexical.Result{ID: h.ID, Score: h.Score}
		}
	}

	seeds := make(map[int]float32, len(ranked))
	for _, r := range ranked {
		if ord, ok := s.hnsw.Ord(r.ID); ok && r.Score > 0 {
			seeds[ord] = r.Score
		}
	}
	return seeds, sim
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
