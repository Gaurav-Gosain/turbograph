package rag

import (
	"context"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/Gaurav-Gosain/turbograph/entity"
	"github.com/Gaurav-Gosain/turbograph/graph"
)

// EntityProgress reports the state of an entity-graph build.
type EntityProgress struct {
	Done      int
	Total     int
	Entities  int
	Relations int
}

// EntityBuildOptions configures BuildEntityGraph.
type EntityBuildOptions struct {
	Workers int
	// BatchSize groups this many chunks into a single model call when the extractor
	// implements entity.BatchExtractor, cutting the number of LLM round trips by
	// roughly this factor. 0 or 1 extracts one chunk per call. Large batches are
	// faster but can dilute a small model's accuracy; 4 to 8 is a good range.
	BatchSize  int
	OnProgress func(EntityProgress)
}

// HasEntityGraph reports whether an entity-relationship graph has been built.
func (s *Store) HasEntityGraph() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.entCSR != nil
}

// EntityCount returns the number of entities in the knowledge graph.
func (s *Store) EntityCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entList)
}

// BuildEntityGraph extracts an entity-relationship knowledge graph from every
// chunk using ex (typically an LLM), replacing any previous entity graph. This is
// the GraphRAG-style alternative to the chunk-similarity graph: it is more
// expensive to build but connects passages by shared entities and typed
// relationships rather than by similarity. Extraction runs in parallel; the
// accumulation and graph construction happen once at the end.
func (s *Store) BuildEntityGraph(ctx context.Context, ex entity.Extractor, opt EntityBuildOptions) error {
	if opt.Workers <= 0 {
		opt.Workers = runtime.GOMAXPROCS(0)
	}
	s.mu.RLock()
	type chunkRef struct{ id, text string }
	refs := make([]chunkRef, len(s.chunks))
	for i, c := range s.chunks {
		refs[i] = chunkRef{c.ID, c.Text}
	}
	s.mu.RUnlock()

	// Group chunks into batches; a batch extractor handles a whole batch in one
	// model call, otherwise each chunk is extracted individually.
	batchSize := opt.BatchSize
	if batchSize < 1 {
		batchSize = 1
	}
	batcher, _ := ex.(entity.BatchExtractor)

	type res struct {
		id string
		ex entity.Extraction
	}
	jobs := make(chan []chunkRef)
	out := make(chan res, opt.Workers)
	var wg sync.WaitGroup
	wg.Add(opt.Workers)
	for w := 0; w < opt.Workers; w++ {
		go func() {
			defer wg.Done()
			for b := range jobs {
				exs := make([]entity.Extraction, len(b))
				if batcher != nil && len(b) > 1 {
					texts := make([]string, len(b))
					for i, r := range b {
						texts[i] = r.text
					}
					// A failed batch drops its chunks' entities but does not abort.
					if got, err := batcher.ExtractBatch(ctx, texts); err == nil && len(got) == len(b) {
						exs = got
					}
				} else {
					for i, r := range b {
						if e, err := ex.Extract(ctx, r.text); err == nil {
							exs[i] = e
						}
					}
				}
				for i, r := range b {
					out <- res{r.id, exs[i]}
				}
			}
		}()
	}
	go func() {
		defer close(jobs)
		for i := 0; i < len(refs); i += batchSize {
			end := min(i+batchSize, len(refs))
			select {
			case <-ctx.Done():
				return
			case jobs <- refs[i:end]:
			}
		}
	}()
	go func() { wg.Wait(); close(out) }()

	eg := entity.NewGraph()
	prog := EntityProgress{Total: len(refs)}
	for r := range out {
		eg.Add(r.id, r.ex)
		prog.Done++
		prog.Entities = eg.Len()
		prog.Relations = len(eg.Relations())
		if opt.OnProgress != nil {
			opt.OnProgress(prog)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	s.eg = eg
	s.rebuildEntityLocked()
	s.mu.Unlock()

	// Embed the entities (name plus description) so retrieval can seed PPR by
	// semantic similarity to the query, not just literal token overlap. This is a
	// best-effort enhancement: if embedding fails, seeding falls back to lexical.
	s.embedEntities(ctx)
	return nil
}

// embedEntities computes and stores an embedding per entity from its display name
// and description, off the write lock. The entity count is small relative to the
// chunk count, so this is one extra embedding batch at build time.
func (s *Store) embedEntities(ctx context.Context) {
	s.mu.RLock()
	texts := make([]string, len(s.entList))
	for i, e := range s.entList {
		name := e.Display
		if name == "" {
			name = e.Name
		}
		if e.Description != "" {
			texts[i] = name + ": " + e.Description
		} else {
			texts[i] = name
		}
	}
	s.mu.RUnlock()
	if len(texts) == 0 {
		return
	}
	vecs, err := s.embedder.Embed(ctx, texts)
	if err != nil || len(vecs) != len(texts) {
		return
	}
	s.mu.Lock()
	// Guard against a concurrent rebuild changing the entity list underneath us.
	if len(vecs) == len(s.entList) {
		s.entVec = vecs
	}
	s.mu.Unlock()
}

// rebuildEntityLocked materializes the entity list, CSR graph, and communities
// from s.eg. The caller must hold the write lock.
func (s *Store) rebuildEntityLocked() {
	s.entVec = nil // entity list is changing; stale embeddings must not be used
	if s.eg == nil {
		s.entList, s.entIndex, s.entCSR, s.entComm = nil, nil, nil, nil
		return
	}
	s.entList = s.eg.Entities()
	s.entIndex = make(map[string]int, len(s.entList))
	for i, e := range s.entList {
		s.entIndex[e.Name] = i
	}
	b := graph.NewBuilder(len(s.entList))
	for _, r := range s.eg.Relations() {
		si, ok1 := s.entIndex[r.Source]
		ti, ok2 := s.entIndex[r.Target]
		if ok1 && ok2 {
			w := r.Weight
			if w <= 0 {
				w = 1
			}
			b.AddEdge(si, ti, w)
		}
	}
	s.entCSR = b.Build()
	s.entComm = graph.DetectCommunities(s.entCSR, graph.CommunityOpts{Seed: s.cfg.Seed})
}

// EntityGraphView exports the entity graph for visualization, reusing the chunk
// graph view shape: id is the entity name, doc_id carries its type, and snippet
// carries its description.
func (s *Store) EntityGraphView() GraphView {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.entCSR == nil {
		return GraphView{}
	}
	view := GraphView{Nodes: make([]GraphNode, len(s.entList))}
	for i, e := range s.entList {
		community := -1
		if s.entComm != nil {
			community = s.entComm.Label(i)
		}
		display := e.Display
		if display == "" {
			display = e.Name
		}
		view.Nodes[i] = GraphNode{
			Index:     i,
			ID:        display,
			DocID:     e.Type,
			Community: community,
			Degree:    s.entCSR.Degree(i),
			Snippet:   snippet(e.Description, 200),
		}
	}
	for _, r := range s.eg.Relations() {
		si, ok1 := s.entIndex[r.Source]
		ti, ok2 := s.entIndex[r.Target]
		if ok1 && ok2 && si < ti {
			view.Edges = append(view.Edges, GraphEdge{Source: si, Target: ti, Weight: r.Weight})
		}
	}
	return view
}

// entityChunkScores propagates query-matched entities over the entity graph with
// Personalized PageRank and projects the resulting entity scores onto chunks,
// returning a normalized score per chunk ordinal. The caller must hold the read
// lock. It returns nil when there is no entity graph or nothing matched.
func (s *Store) entityChunkScores(query string, qv []float32) map[int]float32 {
	if s.entCSR == nil || len(s.entList) == 0 {
		return nil
	}
	seeds := s.entitySeeds(query, qv)
	if len(seeds) == 0 {
		return nil
	}
	ppr := s.entCSR.PersonalizedPageRank(seeds, graph.DefaultPPR())

	idToOrd := make(map[string]int, len(s.chunks))
	for i, c := range s.chunks {
		idToOrd[c.ID] = i
	}
	scores := make(map[int]float32)
	for i, e := range s.entList {
		p := ppr[i]
		if p <= 0 {
			continue
		}
		for _, cid := range e.Chunks {
			if ord, ok := idToOrd[cid]; ok {
				scores[ord] += p
			}
		}
	}
	var max float32
	for _, v := range scores {
		if v > max {
			max = v
		}
	}
	if max > 0 {
		for k := range scores {
			scores[k] /= max
		}
	}
	return scores
}

// entitySeeds picks the entity nodes to seed Personalized PageRank for a query.
// It scores every entity by the cosine similarity of its embedded name and
// description to the query vector (so paraphrases match, "the CEO" finds "chief
// executive"), keeps the top matches above a floor, and adds a lexical bonus for
// entities whose name tokens literally appear in the query (an exact match is a
// strong prior). When no entity embeddings are available (an older store, or
// embedding failed) it falls back to the lexical-only seeding. The caller holds
// the read lock and passes the already-embedded query vector.
func (s *Store) entitySeeds(query string, qv []float32) map[int]float32 {
	lex := s.lexicalEntityHits(query)
	if len(s.entVec) != len(s.entList) || len(qv) == 0 {
		return lex // no embeddings: lexical only (backward compatible)
	}
	const (
		topK  = 24   // seed at most this many entities
		floor = 0.30 // minimum cosine to be considered a match
	)
	type cand struct {
		idx int
		sim float32
	}
	cands := make([]cand, 0, len(s.entList))
	for i, ev := range s.entVec {
		if len(ev) == 0 {
			continue
		}
		if c := cosine(qv, ev); c >= floor {
			cands = append(cands, cand{i, c})
		}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].sim > cands[b].sim })
	if len(cands) > topK {
		cands = cands[:topK]
	}
	seeds := make(map[int]float32, len(cands)+len(lex))
	for _, c := range cands {
		seeds[c.idx] = c.sim
	}
	// A literal name match is a strong signal; add it on top of the dense score.
	for i, hits := range lex {
		seeds[i] += 0.5 * hits
	}
	if len(seeds) == 0 {
		return lex
	}
	return seeds
}

// lexicalEntityHits returns entities whose name tokens appear in the query,
// weighted by the number of matching tokens.
func (s *Store) lexicalEntityHits(query string) map[int]float32 {
	qtokens := map[string]struct{}{}
	for _, t := range strings.FieldsFunc(strings.ToLower(query), notAlnum) {
		if len(t) >= 2 {
			qtokens[t] = struct{}{}
		}
	}
	if len(qtokens) == 0 {
		return nil
	}
	seeds := map[int]float32{}
	for i, e := range s.entList {
		var hits float32
		for _, t := range strings.FieldsFunc(strings.ToLower(e.Name), notAlnum) {
			if _, ok := qtokens[t]; ok {
				hits++
			}
		}
		if hits > 0 {
			seeds[i] = hits
		}
	}
	return seeds
}

func notAlnum(r rune) bool {
	return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9')
}
