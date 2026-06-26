package rag

import "strings"

// Chunk is a unit of retrievable text with provenance.
type Chunk struct {
	ID    string `json:"id"` // stable identifier, "doc#pos"
	DocID string `json:"doc_id"`
	Pos   int    `json:"pos"` // ordinal within the document
	Text  string `json:"text"`
	// Start and End are the [start,end) rune offsets of this chunk's body within
	// the original document text, giving an exact document-to-chunk mapping that
	// callers use to preview a document with its retrieved chunks highlighted.
	// They are best-effort: both are -1 when a chunk's text cannot be located
	// verbatim in the source (for example a custom Chunker that rewrites text).
	Start int `json:"start"`
	End   int `json:"end"`
	// Kind labels non-text chunks. "" (text) is the default; "image" marks a chunk
	// whose Text is a model-written caption of an image, figure, or table.
	Kind string `json:"kind,omitempty"`
	// ImageRef is the asset id of the source image for an image chunk, served by
	// the host application (for example GET /api/asset/<ref>). Empty for text.
	ImageRef string `json:"image_ref,omitempty"`
	// Context is an optional short, LLM-generated sentence that situates this chunk
	// within its document (Anthropic's "contextual retrieval"). It is prepended to
	// the body only for embedding and lexical indexing, never shown to the user or
	// fed to the generator, so it sharpens retrieval without altering answers. It
	// is empty unless a Contextualizer was set on the store at ingest time.
	Context string `json:"context,omitempty"`
}

// IndexText is the text used for embedding and lexical indexing: the optional
// contextual prefix followed by the chunk body. The body alone (Text) is what is
// returned to callers and fed to the generator. When no context was generated
// this is exactly Text, so the default pipeline is byte-for-byte unchanged.
func (c Chunk) IndexText() string {
	if c.Context == "" {
		return c.Text
	}
	return c.Context + "\n\n" + c.Text
}

// ChunkConfig controls how documents are split.
type ChunkConfig struct {
	// Strategy names the built-in chunker: "recursive" (default), "word",
	// "markdown", or "sentence". See NewChunker and the Strategy* constants.
	Strategy string
	// TargetWords is the desired chunk size in whitespace-delimited tokens.
	TargetWords int
	// OverlapWords is how many tokens consecutive chunks share, preserving
	// context across boundaries.
	OverlapWords int
}

// DefaultChunkConfig returns balanced defaults for prose: the recursive splitter
// at a modest size, which keeps paragraphs and sentences intact.
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{Strategy: StrategyRecursive, TargetWords: 120, OverlapWords: 24}
}

// chunkDoc splits a document with the store's configured chunker (Config.Chunker
// if a custom one was supplied, otherwise the strategy named in Config.Chunk),
// turning the pieces into chunks and prepending any heading breadcrumb so the
// embedded and lexically-indexed text carries its context.
func (s *Store) chunkDoc(d Document) []Chunk {
	ch := s.cfg.Chunker
	if ch == nil {
		ch = NewChunker(s.cfg.Chunk)
	}
	pieces := ch.Split(d.Text)
	runes := []rune(d.Text)
	out := make([]Chunk, 0, len(pieces))
	cursor := 0
	for i, p := range pieces {
		t := p.Text
		if len(p.Headings) > 0 {
			t = strings.Join(p.Headings, " > ") + "\n" + t
		}
		// Map the piece body back to its span in the original text. The chunkers
		// normalize whitespace, so the match is whitespace-insensitive; the search
		// runs forward from the previous piece's start so overlapping pieces resolve
		// in document order.
		start, end := locateSpan(runes, p.Text, cursor)
		if start >= 0 {
			cursor = start
		}
		out = append(out, Chunk{
			ID:       d.ID + "#" + itoa(i),
			DocID:    d.ID,
			Pos:      i,
			Text:     t,
			Start:    start,
			End:      end,
			Kind:     d.Kind,
			ImageRef: d.ImageRef,
		})
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
