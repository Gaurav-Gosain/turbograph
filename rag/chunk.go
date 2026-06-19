package rag

import "strings"

// Chunk is a unit of retrievable text with provenance.
type Chunk struct {
	ID    string // stable identifier, "doc#pos"
	DocID string
	Pos   int // ordinal within the document
	Text  string
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
func (s *Store) chunkDoc(docID, text string) []Chunk {
	ch := s.cfg.Chunker
	if ch == nil {
		ch = NewChunker(s.cfg.Chunk)
	}
	pieces := ch.Split(text)
	out := make([]Chunk, 0, len(pieces))
	for i, p := range pieces {
		t := p.Text
		if len(p.Headings) > 0 {
			t = strings.Join(p.Headings, " > ") + "\n" + t
		}
		out = append(out, Chunk{
			ID:    docID + "#" + itoa(i),
			DocID: docID,
			Pos:   i,
			Text:  t,
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
