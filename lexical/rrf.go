package lexical

// DefaultRRFK is the conventional rank-fusion constant. It dampens the
// influence of top ranks so that agreement across lists, not a single list's
// confidence, drives the fused order.
const DefaultRRFK = 60

// RRF fuses several ranked lists with Reciprocal Rank Fusion. A document's
// fused score is the sum over the lists it appears in of 1/(k + rank), where
// rank is its 0-based position in that list. Because the contribution depends
// only on rank and not on each retriever's raw score, RRF combines lists whose
// scores live on incomparable scales (BM25 versus cosine similarity) without
// any normalization.
//
// A non-positive k falls back to DefaultRRFK. The output is ranked best-first
// and is deterministic: ties break by document ID. Documents are deduplicated
// across lists by ID, and duplicate IDs within a single list count only at
// their first (best) position.
func RRF(k int, rankings ...[]Result) []Result {
	if k <= 0 {
		k = DefaultRRFK
	}

	scores := make(map[string]float64)
	for _, list := range rankings {
		seen := make(map[string]struct{}, len(list))
		for rank, r := range list {
			if _, dup := seen[r.ID]; dup {
				continue
			}
			seen[r.ID] = struct{}{}
			scores[r.ID] += 1.0 / float64(k+rank)
		}
	}
	if len(scores) == 0 {
		return nil
	}

	fused := make([]Result, 0, len(scores))
	for id, s := range scores {
		fused = append(fused, Result{ID: id, Score: float32(s)})
	}
	sortResults(fused)
	return fused
}
