package rag

import (
	"strings"
	"unicode"
)

// Chunker splits a document's text into ordered pieces. It is the seam for
// document segmentation: the built-in strategies implement it with pure string
// operations, and a caller can supply their own by implementing this one method
// (set Config.Chunker). Dependencies for richer strategies are injected at
// construction, so the interface stays minimal.
type Chunker interface {
	// Split returns the document's pieces in order. It must never return empty
	// pieces and must be deterministic for a given input.
	Split(text string) []Piece
}

// Piece is one unit produced by a Chunker: the text plus an optional heading
// breadcrumb (for example ["Title", "Section"]). When headings are present the
// store prepends them to the chunk before embedding and lexical indexing, a
// cheap "contextual chunk header" that situates the passage and measurably
// improves retrieval on structured documents.
type Piece struct {
	Text     string
	Headings []string
}

// Chunking strategy names accepted by ChunkConfig.Strategy.
const (
	StrategyRecursive = "recursive" // separator hierarchy, the default
	StrategyWord      = "word"      // fixed overlapping word windows
	StrategyMarkdown  = "markdown"  // split on headings, attach breadcrumbs
	StrategySentence  = "sentence"  // pack whole sentences to the budget
)

// NewChunker builds the chunker named by cfg.Strategy, sized by cfg.TargetWords
// and cfg.OverlapWords. An unknown or empty strategy uses the recursive splitter,
// the pragmatic default that keeps paragraphs and sentences intact.
func NewChunker(cfg ChunkConfig) Chunker {
	target, overlap := chunkBounds(cfg)
	switch cfg.Strategy {
	case StrategyWord:
		return wordChunker{target, overlap}
	case StrategyMarkdown:
		return markdownChunker{target, overlap}
	case StrategySentence:
		return sentenceChunker{target, overlap}
	default:
		return recursiveChunker{target, overlap}
	}
}

func chunkBounds(cfg ChunkConfig) (target, overlap int) {
	target, overlap = cfg.TargetWords, cfg.OverlapWords
	if target <= 0 {
		d := DefaultChunkConfig()
		target, overlap = d.TargetWords, d.OverlapWords
	}
	if overlap < 0 || overlap >= target {
		overlap = target / 4
	}
	return target, overlap
}

// wordChunker is the original strategy: fixed overlapping windows over the word
// stream, ignoring structure. Fast and uniform; the weakest at boundaries.
type wordChunker struct{ target, overlap int }

func (c wordChunker) Split(text string) []Piece {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	stride := c.target - c.overlap
	if stride < 1 {
		stride = 1
	}
	var out []Piece
	for start := 0; start < len(words); start += stride {
		end := min(start+c.target, len(words))
		out = append(out, Piece{Text: strings.Join(words[start:end], " ")})
		if end == len(words) {
			break
		}
	}
	return out
}

// recursiveChunker splits on a hierarchy of separators (paragraph, line,
// sentence, word), descending to the next only when a piece is still over budget,
// then greedily packs the resulting fragments up to the word target with overlap.
// This keeps whole paragraphs and sentences together far better than blind word
// windows and is the recommended default.
type recursiveChunker struct{ target, overlap int }

var recursiveSeparators = []string{"\n\n", "\n", ". ", " "}

func (c recursiveChunker) Split(text string) []Piece {
	frags := splitRecursive(text, recursiveSeparators, c.target)
	return packFragments(frags, c.target, c.overlap)
}

// splitRecursive breaks text down the separator list until each fragment fits the
// word target (or no separators remain), preserving the separators' text.
func splitRecursive(text string, seps []string, target int) []string {
	if wordCount(text) <= target || len(seps) == 0 {
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return []string{text}
	}
	var out []string
	for _, part := range strings.SplitAfter(text, seps[0]) {
		if strings.TrimSpace(part) == "" {
			continue
		}
		if wordCount(part) <= target {
			out = append(out, part)
		} else {
			out = append(out, splitRecursive(part, seps[1:], target)...)
		}
	}
	return out
}

// packFragments greedily merges adjacent fragments into chunks up to target
// words, seeding each new chunk with the tail overlap words of the previous one.
func packFragments(frags []string, target, overlap int) []Piece {
	var out []Piece
	var cur []string
	curWords := 0
	flush := func() {
		if curWords == 0 {
			return
		}
		t := strings.TrimSpace(strings.Join(cur, " "))
		if t != "" {
			out = append(out, Piece{Text: t})
		}
	}
	for _, f := range frags {
		w := wordCount(f)
		if curWords > 0 && curWords+w > target {
			tail := ""
			if overlap > 0 {
				tail = lastWords(strings.Join(cur, " "), overlap)
			}
			flush()
			cur, curWords = nil, 0
			if tail != "" {
				cur = append(cur, tail)
				curWords = wordCount(tail)
			}
		}
		cur = append(cur, strings.TrimSpace(f))
		curWords += w
	}
	flush()
	return out
}

// sentenceChunker packs whole sentences up to the word budget, never cutting a
// sentence in half, with a sentence-aligned overlap.
type sentenceChunker struct{ target, overlap int }

func (c sentenceChunker) Split(text string) []Piece {
	sents := splitSentences(text)
	if len(sents) == 0 {
		return nil
	}
	return packFragments(sents, c.target, c.overlap)
}

// markdownChunker splits on ATX headings, tracks the active heading path, and
// recursively size-splits each section's body, attaching the heading breadcrumb
// to every piece so retrieval sees where the passage sits.
type markdownChunker struct{ target, overlap int }

func (c markdownChunker) Split(text string) []Piece {
	type head struct {
		level int
		title string
	}
	var stack []head
	var body []string
	var out []Piece
	inFence := false

	flush := func() {
		if len(body) == 0 {
			return
		}
		path := make([]string, len(stack))
		for i, h := range stack {
			path[i] = h.title
		}
		for _, p := range packFragments(splitRecursive(strings.Join(body, "\n"), recursiveSeparators, c.target), c.target, c.overlap) {
			if len(path) > 0 {
				p.Headings = append([]string(nil), path...)
			}
			out = append(out, p)
		}
		body = body[:0]
	}

	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			inFence = !inFence
			body = append(body, line)
			continue
		}
		if !inFence {
			if lvl, title, ok := atxHeading(line); ok {
				flush()
				for len(stack) > 0 && stack[len(stack)-1].level >= lvl {
					stack = stack[:len(stack)-1]
				}
				stack = append(stack, head{lvl, title})
				continue
			}
		}
		body = append(body, line)
	}
	flush()
	return out
}

// atxHeading parses a Markdown ATX heading line ("## Title"), returning its level
// (1-6) and title.
func atxHeading(line string) (level int, title string, ok bool) {
	s := strings.TrimSpace(line)
	n := 0
	for n < len(s) && s[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n >= len(s) || s[n] != ' ' {
		return 0, "", false
	}
	return n, strings.TrimSpace(s[n:]), true
}

// splitSentences segments text on sentence-ending punctuation followed by
// whitespace. It is a heuristic forward scan (Go's RE2 has no lookbehind), with a
// short guard for common abbreviations, not a trained model.
func splitSentences(text string) []string {
	var out []string
	start := 0
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		c := runes[i]
		if c != '.' && c != '!' && c != '?' {
			continue
		}
		// Must be followed by whitespace (or end) to count as a boundary.
		if i+1 < len(runes) && !isSpace(runes[i+1]) {
			continue
		}
		if c == '.' && isAbbrev(runes[:i+1]) {
			continue
		}
		// A period is only a real boundary when the next non-space character
		// starts a new sentence (uppercase or a digit). This avoids splitting on
		// decimals ("3.14"), lowercased abbreviations, and initials the abbrev
		// guard missed. Idea from cognee's is_real_paragraph_end lookahead.
		if c == '.' {
			if nx := nextNonSpace(runes, i+1); nx != 0 && !isUpper(nx) && !isDigit(nx) {
				continue
			}
		}
		seg := strings.TrimSpace(string(runes[start : i+1]))
		if seg != "" {
			out = append(out, seg)
		}
		start = i + 1
	}
	if tail := strings.TrimSpace(string(runes[start:])); tail != "" {
		out = append(out, tail)
	}
	return out
}

var abbrevs = map[string]bool{
	"mr": true, "mrs": true, "ms": true, "dr": true, "prof": true, "st": true,
	"vs": true, "etc": true, "e.g": true, "i.e": true, "fig": true, "no": true,
	"al": true, "inc": true, "ltd": true, "co": true, "jr": true, "sr": true,
}

// isAbbrev reports whether the word ending at the period in s is a known
// abbreviation, so "Dr." does not end a sentence.
func isAbbrev(s []rune) bool {
	// take the alphabetic run immediately before the trailing period
	end := len(s) - 1 // the '.'
	i := end - 1
	for i >= 0 && (isLetter(s[i]) || s[i] == '.') {
		i--
	}
	word := strings.ToLower(strings.TrimRight(string(s[i+1:end]), "."))
	return abbrevs[word]
}

func wordCount(s string) int { return len(strings.Fields(s)) }

func lastWords(s string, n int) string {
	f := strings.Fields(s)
	if n >= len(f) {
		return strings.Join(f, " ")
	}
	return strings.Join(f[len(f)-n:], " ")
}

// isSpace reports Unicode whitespace, matching strings.Fields (which the chunkers
// use to normalize text). Aligning the two is what lets locateSpan map a
// whitespace-normalized chunk body back to a source that contains non-ASCII
// whitespace such as a non-breaking space.
func isSpace(r rune) bool  { return unicode.IsSpace(r) }
func isLetter(r rune) bool { return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') }
func isUpper(r rune) bool  { return unicode.IsUpper(r) }
func isDigit(r rune) bool  { return r >= '0' && r <= '9' }

// nextNonSpace returns the first non-space rune at or after index i, or 0 at end.
func nextNonSpace(runes []rune, i int) rune {
	for ; i < len(runes); i++ {
		if !isSpace(runes[i]) {
			return runes[i]
		}
	}
	return 0
}
