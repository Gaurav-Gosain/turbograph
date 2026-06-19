package rag

import "encoding/json"

// locateSpan finds sub within text (both rune-indexed), ignoring differences in
// whitespace runs, scanning forward from rune index from. It returns the
// [start,end) rune offsets of the match, or (-1,-1) if sub is not present. The
// chunkers normalize whitespace when they build a piece (trim and join with single
// spaces), so an exact substring match would fail; this matches the piece's
// non-whitespace runes in order, treating any whitespace in either side as a
// flexible separator. That makes the document-to-chunk mapping robust across all
// the built-in strategies without threading offsets through each one.
func locateSpan(text []rune, sub string, from int) (int, int) {
	s := []rune(sub)
	// Trim leading and trailing whitespace in the needle.
	lo, hi := 0, len(s)
	for lo < hi && isSpace(s[lo]) {
		lo++
	}
	for hi > lo && isSpace(s[hi-1]) {
		hi--
	}
	if lo >= hi {
		return -1, -1
	}
	if from < 0 {
		from = 0
	}
	for begin := from; begin < len(text); begin++ {
		if isSpace(text[begin]) {
			continue
		}
		if text[begin] != s[lo] {
			continue
		}
		if start, end, ok := matchFrom(text, s, begin, lo, hi); ok {
			return start, end
		}
	}
	return -1, -1
}

// matchFrom attempts to align s[lo:hi] against text starting at ti, allowing any
// run of whitespace in text to satisfy a whitespace position in s and skipping
// extra whitespace in text between tokens. It returns the matched [start,end).
func matchFrom(text, s []rune, ti, lo, hi int) (start, end int, ok bool) {
	start = -1
	for sj := lo; sj < hi; {
		if isSpace(s[sj]) {
			for sj < hi && isSpace(s[sj]) {
				sj++
			}
			for ti < len(text) && isSpace(text[ti]) {
				ti++
			}
			continue
		}
		for ti < len(text) && isSpace(text[ti]) {
			ti++
		}
		if ti >= len(text) || text[ti] != s[sj] {
			return -1, -1, false
		}
		if start < 0 {
			start = ti
		}
		ti++
		sj++
	}
	return start, ti, true
}

// ChunkSpan locates one chunk inside its document for highlighting.
type ChunkSpan struct {
	ID    string `json:"id"`
	Pos   int    `json:"pos"`
	Start int    `json:"start"` // rune offset, -1 if the chunk could not be located
	End   int    `json:"end"`   // rune offset (exclusive)
}

// DocView is a document with everything needed to preview it and highlight the
// chunks a query retrieved: the original text, the document's metadata, and the
// span of every chunk within the text.
type DocView struct {
	ID    string          `json:"id"`
	Text  string          `json:"text"`
	Meta  json.RawMessage `json:"meta,omitempty"`
	Spans []ChunkSpan     `json:"spans"`
}

// DocumentView returns the original text of a document, its metadata, and the
// span of each of its chunks, so a caller can render the whole document with the
// retrieved chunks highlighted. It reports false for an unknown document or one
// whose original text was not retained (ingested before text was tracked).
func (s *Store) DocumentView(id string) (DocView, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	text, ok := s.currentTextLocked(id)
	if !ok {
		return DocView{}, false
	}
	view := DocView{ID: id, Text: text, Meta: s.docMeta[id]}
	for i := range s.chunks {
		c := s.chunks[i]
		if c.DocID == id {
			view.Spans = append(view.Spans, ChunkSpan{ID: c.ID, Pos: c.Pos, Start: c.Start, End: c.End})
		}
	}
	return view, true
}

// currentTextLocked returns a document's current full text. The version history
// keeps the text of every version, so the live document is the newest one. The
// caller must hold at least the read lock.
func (s *Store) currentTextLocked(id string) (string, bool) {
	vs := s.versions[id]
	if len(vs) == 0 {
		return "", false
	}
	return vs[len(vs)-1].Text, true
}

// DocMeta returns the raw JSON metadata attached to a document, or nil if none.
func (s *Store) DocMeta(id string) json.RawMessage {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.docMeta[id]
}

// SetDocMeta replaces the metadata for a document. A nil or empty map clears it.
// Metadata is independent of content, so this updates it without re-ingesting.
func (s *Store) SetDocMeta(id string, meta map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recordMetaLocked(id, meta)
}

// recordMetaLocked stores a document's metadata as canonical JSON, or clears it
// when empty. The caller must hold the write lock.
func (s *Store) recordMetaLocked(id string, meta map[string]any) error {
	if len(meta) == 0 {
		delete(s.docMeta, id)
		return nil
	}
	b, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	if s.docMeta == nil {
		s.docMeta = make(map[string]json.RawMessage)
	}
	s.docMeta[id] = b
	return nil
}
