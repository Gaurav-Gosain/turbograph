package rag

import (
	"context"
	"crypto/sha256"
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
	// Cached counts the chunks answered from the extraction cache rather than the
	// model. They complete immediately, so a caller can say how much of the build was
	// already paid for instead of showing a progress bar that jumps.
	Cached int
	// New lists the entities this chunk surfaced that had not been seen before, so a
	// caller can show the graph populating as it is extracted rather than staring at
	// a counter. These are the extractor's raw names: the final graph canonicalizes
	// and prunes, so it is usually smaller than the number streamed here, and that
	// difference is worth showing rather than hiding.
	New []EntityBrief
	// NewRelations lists the relationships this chunk asserted, so the caller can draw
	// the graph as it is discovered rather than only its nodes.
	NewRelations []RelationBrief
}

// RelationBrief is one newly asserted relationship, enough to draw an edge.
type RelationBrief struct {
	Source      string `json:"source"`
	Target      string `json:"target"`
	Description string `json:"description,omitempty"`
}

// EntityBrief is a newly discovered entity, enough to display it live.
type EntityBrief struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// EntityBuildOptions configures BuildEntityGraph.
type EntityBuildOptions struct {
	Workers int
	// BatchSize groups this many chunks into a single model call when the extractor
	// implements entity.BatchExtractor, cutting the number of LLM round trips by
	// roughly this factor. 0 or 1 extracts one chunk per call.
	//
	// Larger is NOT better: a small local model loses track of which passage it is
	// reading and drops most of what is in them. Measured with qwen3.5:4b, a batch of
	// 4 found 6 entities and 4 relationships where a batch of 2 found 17 and 8. Keep
	// it low (1 or 2) unless a large model has been shown to hold up.
	BatchSize int
	// Model names the model doing the extraction. It is part of the cache key: what
	// a 4b local model said about a chunk is not what a frontier model would say, so
	// switching models must re-extract rather than reuse.
	Model string
	// Refresh ignores the cache and re-extracts every chunk. Use it after changing
	// something the cache key does not cover, or to retry a bad extraction.
	Refresh    bool
	OnProgress func(EntityProgress)
}

// extractPromptVersion salts the cache key. Bump it whenever the extraction prompt
// changes, so a new turbograph does not silently reuse answers the old prompt got.
const extractPromptVersion = "v1"

// cachedExtraction is one memoized answer. It carries the hash of the text it came
// from so the cache can be pruned against the live corpus without knowing which
// models produced which entries.
type cachedExtraction struct {
	Text [32]byte          `json:"text"` // sha256 of the chunk text alone
	Ex   entity.Extraction `json:"ex"`
}

// extractKey identifies a cached extraction: the chunk text, the model that read it,
// and the prompt that was used. Any of the three changing means the cached answer is
// no longer the answer to the question being asked.
func extractKey(model, text string) [32]byte {
	h := sha256.New()
	h.Write([]byte(extractPromptVersion))
	h.Write([]byte{0})
	h.Write([]byte(model))
	h.Write([]byte{0})
	h.Write([]byte(text))
	var k [32]byte
	copy(k[:], h.Sum(nil))
	return k
}

// CachedExtractions reports how many chunk extractions are memoized, so a caller can
// tell how much of the next build is already paid for.
func (s *Store) CachedExtractions() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.extractCache)
}

// pruneExtractCacheLocked drops entries whose chunk text is no longer in the corpus.
// Without it the cache would keep an answer for every chunk that ever existed,
// including every superseded version of an edited document, and persist them forever.
// Entries for a model that is not the current one are kept: switching model and back
// should not mean paying for the extraction twice.
func (s *Store) pruneExtractCacheLocked() {
	if len(s.extractCache) == 0 {
		return
	}
	live := make(map[[32]byte]struct{}, len(s.chunks))
	for _, c := range s.chunks {
		live[sha256.Sum256([]byte(c.Text))] = struct{}{}
	}
	for k, e := range s.extractCache {
		if _, ok := live[e.Text]; !ok {
			delete(s.extractCache, k)
		}
	}
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

// RelationCount reports how many relationships the entity graph holds.
func (s *Store) RelationCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.eg == nil {
		return 0
	}
	return len(s.eg.Relations())
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
	type chunkRef struct {
		id, text string
		key      [32]byte
	}
	s.mu.RLock()
	refs := make([]chunkRef, len(s.chunks))
	for i, c := range s.chunks {
		refs[i] = chunkRef{id: c.ID, text: c.Text, key: extractKey(opt.Model, c.Text)}
	}
	// Take the cached answers under the same lock that read the chunks.
	cached := make(map[[32]byte]entity.Extraction, len(s.extractCache))
	if !opt.Refresh {
		for _, r := range refs {
			if e, ok := s.extractCache[r.key]; ok {
				cached[r.key] = e.Ex
			}
		}
	}
	s.mu.RUnlock()

	// Only the chunks nobody has asked the model about go to the model. This is the
	// whole game: extraction is ~1s of GPU per call and does not get faster with more
	// workers, so the only way to make a rebuild quick is not to make the calls.
	todo := make([]chunkRef, 0, len(refs))
	for _, r := range refs {
		if _, hit := cached[r.key]; !hit {
			todo = append(todo, r)
		}
	}

	// Group chunks into batches; a batch extractor handles a whole batch in one
	// model call, otherwise each chunk is extracted individually.
	batchSize := opt.BatchSize
	if batchSize < 1 {
		batchSize = 1
	}
	batcher, _ := ex.(entity.BatchExtractor)

	type res struct {
		id       string
		key      [32]byte
		textHash [32]byte
		ex       entity.Extraction
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
					out <- res{r.id, r.key, sha256.Sum256([]byte(r.text)), exs[i]}
				}
			}
		}()
	}
	go func() {
		defer close(jobs)
		for i := 0; i < len(todo); i += batchSize {
			end := min(i+batchSize, len(todo))
			select {
			case <-ctx.Done():
				return
			case jobs <- todo[i:end]:
			}
		}
	}()
	go func() { wg.Wait(); close(out) }()

	eg := entity.NewGraph()
	prog := EntityProgress{Total: len(refs), Cached: len(cached)}
	fresh := make(map[[32]byte]entity.Extraction, len(todo))
	freshText := make(map[[32]byte][32]byte, len(todo))
	seen := make(map[string]struct{})

	// absorb folds one chunk's extraction into the graph and reports what it added.
	// The report is the raw extraction, not the finished graph: the graph is
	// canonicalized and pruned at the end, so what streams here is what the model
	// said, and the reconciliation between the two is shown rather than hidden.
	absorb := func(id string, ex entity.Extraction) {
		// Clean first, and stream what was cleaned. The graph drops relations whose
		// endpoints are malformed, so reporting them would draw an edge, and a node for
		// the relation's own verb, that the finished graph never contains.
		ex = entity.Clean(ex)
		eg.Add(id, ex)
		prog.Done++
		prog.Entities = eg.Len()
		prog.Relations = len(eg.Relations())
		prog.New = prog.New[:0]
		prog.NewRelations = prog.NewRelations[:0]
		// Report the entities this chunk surfaced for the first time, so the caller can
		// show the graph filling in as it is extracted. Relationship endpoints count:
		// an extractor routinely names an entity only inside a fact about it, and those
		// become real nodes, so listing only ex.Entities under-reports the graph.
		surface := func(name, typ string) {
			name = strings.TrimSpace(name)
			key := strings.ToLower(name)
			if key == "" {
				return
			}
			if _, dup := seen[key]; dup {
				return
			}
			seen[key] = struct{}{}
			prog.New = append(prog.New, EntityBrief{Name: name, Type: typ})
		}
		for _, e := range ex.Entities {
			surface(e.Name, e.Type)
		}
		for _, rel := range ex.Relations {
			surface(rel.Source, "")
			surface(rel.Target, "")
			src, tgt := strings.TrimSpace(rel.Source), strings.TrimSpace(rel.Target)
			if src == "" || tgt == "" || strings.EqualFold(src, tgt) {
				continue
			}
			prog.NewRelations = append(prog.NewRelations, RelationBrief{
				Source: src, Target: tgt, Description: rel.Description,
			})
		}
		if opt.OnProgress != nil {
			opt.OnProgress(prog)
		}
	}

	// The chunks already extracted go in first and cost nothing, so a rebuild of a
	// corpus that has only grown a little draws almost the whole graph immediately and
	// then fills in the new part.
	for _, r := range refs {
		if e, hit := cached[r.key]; hit {
			absorb(r.id, e)
		}
	}
	for r := range out {
		fresh[r.key] = r.ex
		freshText[r.key] = r.textHash
		absorb(r.id, r.ex)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Clean up the raw extraction before materializing the graph: merge surface
	// variants of the same entity (and rewrite the edges through the merge), then
	// drop generic and malformed ghost nodes. This concentrates PageRank mass and
	// reconnects the subgraph, improving entity-graph retrieval.
	eg.Canonicalize()
	eg.Prune()

	s.mu.Lock()
	s.eg = eg
	// Remember what the model said, so the next build does not ask again. The cache is
	// keyed by content, so a chunk that survives an edit unchanged keeps its answer.
	if s.extractCache == nil {
		s.extractCache = make(map[[32]byte]cachedExtraction, len(fresh))
	}
	for k, e := range fresh {
		s.extractCache[k] = cachedExtraction{Text: freshText[k], Ex: e}
	}
	s.pruneExtractCacheLocked()
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
	s.entVec = nil                                  // entity list is changing; stale embeddings must not be used
	s.factVec, s.factSrc, s.factTgt = nil, nil, nil // the fact index refers to node indices that are about to move
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
func (s *Store) entityChunkScores(query string, qv []float32, link string) map[int]float32 {
	if s.entCSR == nil || len(s.entList) == 0 {
		return nil
	}
	// Fact-linking is the default: seeding PageRank from the relationships a query
	// matches beats seeding from entity names it matches, measured on MultiHop-RAG (fact
	// vs node, higher recall and nDCG at every mix level), which is what HippoRAG 2's
	// ablation predicts. "node" selects the older name-matching behavior for comparison.
	var seeds map[int]float32
	if link == "node" {
		seeds = s.entitySeeds(query, qv)
	} else {
		seeds = s.factSeeds(qv)
	}
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

// RelationContext returns up to maxRels knowledge-graph relationships grounded in
// the given retrieved chunk ids, rendered as short "Subject -> Object: fact"
// lines, most salient first. A relationship is fully grounded when both of its
// endpoints are mentioned in the retrieved chunks: such a fact ties two retrieved
// passages together and is exactly what a multi-hop answer needs but a single
// chunk may not state. When too few are fully grounded, relationships with one
// grounded endpoint are appended (ranked by weight) so a bridge to a neighbouring
// entity is not lost. The result is deterministic and LLM-free; it returns nil
// when no entity graph has been built. This is turbograph's analogue of feeding
// the retrieved subgraph triplets into the prompt, not just the text passages.
func (s *Store) RelationContext(chunkIDs []string, maxRels int) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.eg == nil || len(s.entList) == 0 || maxRels <= 0 {
		return nil
	}
	want := make(map[string]struct{}, len(chunkIDs))
	for _, id := range chunkIDs {
		want[id] = struct{}{}
	}
	// An entity is present when any chunk that mentions it was retrieved.
	present := make(map[string]bool, len(s.entList))
	display := make(map[string]string, len(s.entList))
	for _, e := range s.entList {
		display[e.Name] = e.Display
		for _, c := range e.Chunks {
			if _, ok := want[c]; ok {
				present[e.Name] = true
				break
			}
		}
	}
	type scored struct {
		rel  entity.Relation
		both bool
	}
	var cands []scored
	for _, r := range s.eg.Relations() {
		ps, pt := present[r.Source], present[r.Target]
		if !ps && !pt {
			continue // unrelated to anything retrieved
		}
		cands = append(cands, scored{r, ps && pt})
	}
	// Fully grounded facts first, then bridges; ties broken by edge weight.
	sort.SliceStable(cands, func(a, b int) bool {
		if cands[a].both != cands[b].both {
			return cands[a].both
		}
		return cands[a].rel.Weight > cands[b].rel.Weight
	})
	if len(cands) > maxRels {
		cands = cands[:maxRels]
	}
	out := make([]string, 0, len(cands))
	for _, c := range cands {
		src, tgt := display[c.rel.Source], display[c.rel.Target]
		if src == "" {
			src = c.rel.Source
		}
		if tgt == "" {
			tgt = c.rel.Target
		}
		line := src + " -> " + tgt
		if d := strings.TrimSpace(c.rel.Description); d != "" {
			line += ": " + d
		}
		out = append(out, line)
	}
	return out
}

// ensureFactIndex builds the fact index if it is missing: one embedding per relation,
// where a relation is rendered as the short fact "<source> <how they relate> <target>".
// It is the substrate for query->fact linking, and it is built lazily because a store
// loaded from disk has the relations but not their embeddings, and computing them costs
// one embedding call over a few hundred short strings.
func (s *Store) ensureFactIndex(ctx context.Context) {
	s.mu.RLock()
	built := s.factVec != nil
	haveGraph := s.eg != nil && len(s.entList) > 0
	s.mu.RUnlock()
	if built || !haveGraph {
		return
	}

	s.mu.RLock()
	gen := s.entGen
	rels := s.eg.Relations()
	texts := make([]string, 0, len(rels))
	src := make([]int, 0, len(rels))
	tgt := make([]int, 0, len(rels))
	display := func(name string) string {
		if i, ok := s.entIndex[name]; ok {
			if d := s.entList[i].Display; d != "" {
				return d
			}
		}
		return name
	}
	for _, r := range rels {
		si, ok1 := s.entIndex[r.Source]
		ti, ok2 := s.entIndex[r.Target]
		if !ok1 || !ok2 {
			continue
		}
		fact := display(r.Source) + " " + r.Description + " " + display(r.Target)
		texts = append(texts, fact)
		src = append(src, si)
		tgt = append(tgt, ti)
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
	// Reject the result if the graph was rebuilt while we embedded off the lock: src/tgt
	// index the entity list as it was then, and a rebuild reorders it. Checking factVec ==
	// nil is not enough, because a rebuild sets it nil, which would let stale indices in.
	if s.entGen == gen && s.factVec == nil && len(vecs) == len(src) {
		s.factVec, s.factSrc, s.factTgt = vecs, src, tgt
	}
	s.mu.Unlock()
}

// factSeeds links the query to the entity graph through its relationships. It scores
// every fact by cosine to the query, takes the strongest, and seeds PageRank from those
// facts' endpoint entities.
//
// Two things make it different from seeding on entity names. The linking unit is a
// fact, so "who runs Acme" matches the relation "Jane Doe is the CEO of Acme" directly
// rather than hoping the query vector lands near the bare token "Acme". And each
// endpoint's seed is divided by how many chunks it appears in, an IDF-like penalty that
// stops a hub entity named in half the corpus from swallowing the walk. The caller holds
// the read lock and passes the embedded query.
func (s *Store) factSeeds(qv []float32) map[int]float32 {
	if len(s.factVec) == 0 || len(qv) == 0 {
		return nil
	}
	const (
		topFacts = 12   // seed from at most this many facts
		floor    = 0.30 // minimum cosine for a fact to be a match
		capNodes = 8    // keep at most this many seed entities
	)
	type fc struct {
		i   int
		sim float32
	}
	cands := make([]fc, 0, len(s.factVec))
	for i, fv := range s.factVec {
		if len(fv) == 0 {
			continue
		}
		if c := cosine(qv, fv); c >= floor {
			cands = append(cands, fc{i, c})
		}
	}
	if len(cands) == 0 {
		return nil
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].sim > cands[b].sim })
	if len(cands) > topFacts {
		cands = cands[:topFacts]
	}

	seeds := make(map[int]float32, 2*len(cands))
	add := func(node int, sim float32) {
		chunks := 1
		if node >= 0 && node < len(s.entList) {
			if n := len(s.entList[node].Chunks); n > 1 {
				chunks = n
			}
		}
		seeds[node] += sim / float32(chunks)
	}
	for _, c := range cands {
		add(s.factSrc[c.i], c.sim)
		add(s.factTgt[c.i], c.sim)
	}
	// Keep only the strongest endpoints, so one query does not seed the whole graph.
	if len(seeds) > capNodes {
		type ns struct {
			n int
			s float32
		}
		all := make([]ns, 0, len(seeds))
		for n, v := range seeds {
			all = append(all, ns{n, v})
		}
		sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
		seeds = make(map[int]float32, capNodes)
		for _, x := range all[:capNodes] {
			seeds[x.n] = x.s
		}
	}
	return seeds
}
