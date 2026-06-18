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
	// TargetWords is the desired chunk size in whitespace-delimited tokens.
	TargetWords int
	// OverlapWords is how many tokens consecutive chunks share, preserving
	// context across boundaries.
	OverlapWords int
}

// DefaultChunkConfig returns balanced defaults for prose.
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{TargetWords: 120, OverlapWords: 24}
}

// chunkDocument splits text into overlapping word windows. It is deterministic
// and allocation-light, and never emits empty chunks.
func chunkDocument(docID, text string, cfg ChunkConfig) []Chunk {
	if cfg.TargetWords <= 0 {
		cfg = DefaultChunkConfig()
	}
	if cfg.OverlapWords < 0 || cfg.OverlapWords >= cfg.TargetWords {
		cfg.OverlapWords = cfg.TargetWords / 4
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	stride := cfg.TargetWords - cfg.OverlapWords
	var chunks []Chunk
	pos := 0
	for start := 0; start < len(words); start += stride {
		end := min(start+cfg.TargetWords, len(words))
		chunks = append(chunks, Chunk{
			ID:    docID + "#" + itoa(pos),
			DocID: docID,
			Pos:   pos,
			Text:  strings.Join(words[start:end], " "),
		})
		pos++
		if end == len(words) {
			break
		}
	}
	return chunks
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
