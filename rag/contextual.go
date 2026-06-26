package rag

import (
	"context"
	"runtime"
	"strings"
	"sync"
)

// Contextualizer turns a (document, chunk) pair into a short sentence situating
// the chunk within the document. An *ollama.Client bound to a model satisfies it
// (via a thin adapter), as does any LLM wrapper with this one method. It is the
// only dependency the contextual-retrieval feature adds, and it is injected, so
// the rag package stays model-agnostic.
type Contextualizer interface {
	Generate(ctx context.Context, system, prompt string) (string, error)
}

// SetContextualizer enables Anthropic-style contextual retrieval: at ingest time
// each chunk is prefixed, for indexing only, with a short generated sentence that
// situates it in its document. This fixes the fragmentation that hurts a flat
// chunker most ("it grew 3%" with no nearby subject becomes retrievable) and
// strengthens both the dense and the lexical index at once, for zero query-time
// cost. Pass nil to disable. The generator is used only during Build and
// AddDocuments; it is never persisted.
func (s *Store) SetContextualizer(c Contextualizer) {
	s.mu.Lock()
	s.contextualizer = c
	s.mu.Unlock()
}

const contextSystem = "You situate a passage within its source document to improve search retrieval. " +
	"Reply with one short sentence and nothing else: no preamble, no quotes, no markdown."

// contextPrompt builds the situating-context request. The document is clipped to
// a window so the prompt stays cheap on large documents; the chunk is sent whole.
func contextPrompt(doc, chunk string) string {
	return "Document:\n" + clipRunes(doc, 6000) +
		"\n\nPassage:\n" + chunk +
		"\n\nIn one short sentence, state what this passage is about and how it fits in the " +
		"document, naming the key entity, section, or time it concerns. Answer with only the sentence."
}

// contextualize fills each text chunk's Context field by asking the configured
// Contextualizer to situate it within its document. It runs before embedding,
// off the write lock, with bounded concurrency. Failures and image chunks are
// left with an empty Context, so the chunk simply indexes its body as before;
// the feature can only help or no-op, never error out an ingest.
func (s *Store) contextualize(ctx context.Context, docs []Document, chunks []Chunk) {
	c := s.contextualizer
	if c == nil {
		return
	}
	docText := make(map[string]string, len(docs))
	for _, d := range docs {
		docText[d.ID] = d.Text
	}
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}
	if workers < 1 {
		workers = 1
	}
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for i := range chunks {
		if chunks[i].Kind != "" { // only situate plain text, not image captions
			continue
		}
		doc := docText[chunks[i].DocID]
		if doc == "" || strings.TrimSpace(chunks[i].Text) == "" {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, doc string) {
			defer wg.Done()
			defer func() { <-sem }()
			out, err := c.Generate(ctx, contextSystem, contextPrompt(doc, chunks[i].Text))
			if err != nil {
				return
			}
			cxt := strings.TrimSpace(firstLine(out))
			// Guard against a model that echoes the whole passage or rambles: a
			// situating sentence is short. Cap to keep the index prefix lean.
			if cxt != "" && len(cxt) < len(chunks[i].Text)+len(doc) {
				chunks[i].Context = clipRunes(cxt, 400)
			}
		}(i, doc)
	}
	wg.Wait()
}

// firstLine returns the first non-empty line of s, the situating sentence; a
// chatty model sometimes adds trailing lines we do not want in the index.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			return t
		}
	}
	return strings.TrimSpace(s)
}

// clipRunes truncates s to at most n runes, on a rune boundary so multi-byte
// text is never split mid-character.
func clipRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
