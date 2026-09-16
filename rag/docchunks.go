package rag

import "sort"

// DocChunkSample summarizes how one document was chunked after an ingest: the
// chunk count plus the first and last chunk's body. The first/last pair is the
// cheapest honest preview that the text really was split (and where it begins and
// ends), which is what the UI surfaces after an ingest.
type DocChunkSample struct {
	ID     string `json:"id"`
	Chunks int    `json:"chunks"`
	First  string `json:"first,omitempty"`
	Last   string `json:"last,omitempty"`
}

// DocChunkSamples reports the chunk count and first/last chunk body for each of
// the given document ids. Unknown ids are omitted; a document whose chunks were
// not retained reports nothing. The samples are independent of the order of ids.
func (s *Store) DocChunkSamples(ids []string) []DocChunkSample {
	s.mu.RLock()
	defer s.mu.RUnlock()
	samples := make([]DocChunkSample, 0, len(ids))
	for _, id := range ids {
		var pos []struct {
			pos  int
			text string
		}
		for i := range s.chunks {
			c := s.chunks[i]
			if c.DocID == id {
				pos = append(pos, struct {
					pos  int
					text string
				}{c.Pos, c.Text})
			}
		}
		if len(pos) == 0 {
			continue
		}
		sort.Slice(pos, func(a, b int) bool { return pos[a].pos < pos[b].pos })
		sample := DocChunkSample{ID: id, Chunks: len(pos), First: pos[0].text}
		if len(pos) > 1 {
			sample.Last = pos[len(pos)-1].text
		}
		samples = append(samples, sample)
	}
	return samples
}

// ChunksByID resolves chunk ids to their verbatim chunks, preserving the
// requested order and skipping ids that are unknown to the store. It is
// intended for surfacing a community's member passages as real source
// fragments (global queries) instead of the generated summary.
func (s *Store) ChunksByID(ids []string) []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()
	idx := make(map[string]int, len(s.chunks))
	for i, c := range s.chunks {
		idx[c.ID] = i
	}
	out := make([]Chunk, 0, len(ids))
	for _, id := range ids {
		if n, ok := idx[id]; ok && n >= 0 && n < len(s.chunks) {
			out = append(out, s.chunks[n])
		}
	}
	return out
}
