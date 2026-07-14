package rag

import (
	"context"
	"math"
	"runtime"
	"sort"
	"sync"
)

// cosine is the cosine similarity of two vectors, used to rank community
// summaries against a query. The embeddings may not be unit length, so it
// normalizes rather than assuming it.
func cosine(a, b []float32) float32 {
	d := dot32(a, b)
	na, nb := dot32(a, a), dot32(b, b)
	if na == 0 || nb == 0 {
		return 0
	}
	return d / float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
}

// CommunitySummary is a natural-language description of one community of chunks,
// generated once at index time so global, thematic, corpus-wide questions can be
// answered from the summaries instead of from raw passages. The communities are
// the existing label-propagation partitions of the chunk similarity graph; the
// summary is the missing piece that lets turbograph answer "what are the main
// themes" style questions (the GraphRAG community-report idea), without giving up
// its cheap default: summaries are built only when asked for.
type CommunitySummary struct {
	Label   int      `json:"label"`
	Size    int      `json:"size"`    // number of chunks in the community
	Summary string   `json:"summary"` // the generated thematic summary
	Chunks  []string `json:"chunks"`  // member chunk ids
	DocIDs  []string `json:"doc_ids"` // distinct source documents
}

// Summarizer turns a community's member passages into a short thematic summary,
// typically by calling a language model. It is supplied by the caller so the rag
// package stays free of any model dependency.
type Summarizer func(ctx context.Context, passages []string) (string, error)

// CommunityOptions configures BuildCommunitySummaries.
type CommunityOptions struct {
	Workers     int // concurrent summarizers; 0 uses GOMAXPROCS
	MaxPassages int // cap member passages per community in the prompt (0 = 12)
	MinSize     int // skip communities smaller than this (0 = 1, summarize all)
	OnProgress  func(done, total int)
}

// HasCommunitySummaries reports whether community summaries have been generated.
func (s *Store) HasCommunitySummaries() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.commSummary) > 0
}

// BuildCommunitySummaries generates a summary for every community of the chunk
// similarity graph using summarize, replacing any previous summaries. It is the
// global-query counterpart to the entity graph: more expensive than the embedding
// only default (one model call per community, far fewer than per chunk), and
// entirely opt-in.
func (s *Store) BuildCommunitySummaries(ctx context.Context, summarize Summarizer, opt CommunityOptions) error {
	s.ensureGraph()
	if opt.Workers <= 0 {
		opt.Workers = runtime.GOMAXPROCS(0)
	}
	if opt.MaxPassages <= 0 {
		opt.MaxPassages = 12
	}
	if opt.MinSize <= 0 {
		opt.MinSize = 1
	}

	// Snapshot the communities and their member passages off the write lock.
	type job struct {
		label    int
		passages []string
	}
	s.mu.RLock()
	var jobs []job
	if s.comm != nil {
		for c := 0; c < s.comm.NumCommunities(); c++ {
			members := s.comm.Members(c)
			if len(members) < opt.MinSize {
				continue
			}
			passages := make([]string, 0, len(members))
			for _, ord := range members {
				if ord >= 0 && ord < len(s.chunks) {
					passages = append(passages, s.chunks[ord].Text)
					if len(passages) >= opt.MaxPassages {
						break
					}
				}
			}
			if len(passages) > 0 {
				jobs = append(jobs, job{label: c, passages: passages})
			}
		}
	}
	s.mu.RUnlock()
	if len(jobs) == 0 {
		return nil
	}

	type result struct {
		label   int
		summary string
	}
	in := make(chan job)
	out := make(chan result, opt.Workers)
	var wg sync.WaitGroup
	wg.Add(opt.Workers)
	for w := 0; w < opt.Workers; w++ {
		go func() {
			defer wg.Done()
			for j := range in {
				sum, err := summarize(ctx, j.passages)
				if err == nil {
					out <- result{j.label, sum}
				} else {
					out <- result{j.label, ""}
				}
			}
		}()
	}
	go func() {
		defer close(in)
		for _, j := range jobs {
			select {
			case <-ctx.Done():
				return
			case in <- j:
			}
		}
	}()
	go func() { wg.Wait(); close(out) }()

	summaries := make(map[int]string, len(jobs))
	done := 0
	for r := range out {
		done++
		if r.summary != "" {
			summaries[r.label] = r.summary
		}
		if opt.OnProgress != nil {
			opt.OnProgress(done, len(jobs))
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	s.commSummary = summaries
	s.mu.Unlock()
	return nil
}

// CommunitySummaries returns the generated summaries, largest community first,
// each enriched with its current member chunk and document ids.
func (s *Store) CommunitySummaries() []CommunitySummary {
	s.ensureGraph()
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.communitySummariesLocked()
}

func (s *Store) communitySummariesLocked() []CommunitySummary {
	if len(s.commSummary) == 0 || s.comm == nil {
		return nil
	}
	out := make([]CommunitySummary, 0, len(s.commSummary))
	for label, summary := range s.commSummary {
		members := s.comm.Members(label)
		cs := CommunitySummary{Label: label, Size: len(members), Summary: summary}
		seenDoc := map[string]struct{}{}
		for _, ord := range members {
			if ord < 0 || ord >= len(s.chunks) {
				continue
			}
			cs.Chunks = append(cs.Chunks, s.chunks[ord].ID)
			if d := s.chunks[ord].DocID; d != "" {
				if _, ok := seenDoc[d]; !ok {
					seenDoc[d] = struct{}{}
					cs.DocIDs = append(cs.DocIDs, d)
				}
			}
		}
		out = append(out, cs)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Size > out[j].Size })
	return out
}

// RelevantCommunities ranks the community summaries by similarity of their text
// to the query and returns the top k, the seed set for a global, map-reduce style
// answer. It embeds the query and the summaries with the store's embedder.
func (s *Store) RelevantCommunities(ctx context.Context, query string, k int) ([]CommunitySummary, error) {
	s.ensureGraph()
	all := s.CommunitySummaries()
	if len(all) == 0 {
		return nil, nil
	}
	if k <= 0 || k > len(all) {
		k = len(all)
	}
	texts := make([]string, len(all))
	for i, c := range all {
		texts[i] = c.Summary
	}
	qvs, err := embedQuery(ctx, s.embedder, []string{query})
	if err != nil {
		return nil, err
	}
	sv, err := s.embedder.Embed(ctx, texts)
	if err != nil {
		return nil, err
	}
	qv := qvs[0]
	type scored struct {
		idx int
		sim float32
	}
	ranked := make([]scored, len(all))
	for i := range all {
		ranked[i] = scored{i, cosine(qv, sv[i])}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].sim > ranked[j].sim })
	out := make([]CommunitySummary, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, all[ranked[i].idx])
	}
	return out, nil
}
