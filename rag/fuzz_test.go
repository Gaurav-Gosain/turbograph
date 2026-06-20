package rag

import (
	"strings"
	"testing"
	"unicode"
)

// strategies under test. The empty string exercises the default (recursive)
// dispatch path in NewChunker.
var fuzzStrategies = []string{
	StrategyRecursive,
	StrategyWord,
	StrategyMarkdown,
	StrategySentence,
	"", // unknown/empty -> recursive default
}

// adversarialTexts is the shared seed corpus for the text-processing fuzzers.
// It deliberately covers the cases that break naive splitters and offset math:
// empty/whitespace-only input, long separator runs, unicode/emoji/CJK, a single
// enormous "word", markdown headings and code fences, mixed newline styles, and
// abbreviation-laden prose.
var adversarialTexts = []string{
	"",
	" ",
	"\n\n\n\n",
	"\t \t \t",
	"\r\n\r\n",
	"a",
	"word",
	"two words",
	"First paragraph one two three.\n\nSecond paragraph four five six seven.\n\nThird paragraph eight nine ten.",
	strings.Repeat("\n", 200),
	strings.Repeat(" ", 500) + "x" + strings.Repeat(" ", 500),
	strings.Repeat("a", 5000),   // one enormous word, no separators
	strings.Repeat("la ", 1000), // many tiny words
	"emoji 😀😀😀 and CJK 你好世界 mixed in",
	"héllo wörld naïve café résumé",
	"Dr. Smith met Mr. Jones vs. Ms. Lee at 3 p.m. etc. and left.",
	"No boundary here because no space after.dot still going on and on",
	"# Title\n\nIntro words here.\n\n## Section A\n\nBody of section A with several words to chunk.\n\n## Section B\n\nBody of section B content here.",
	"# Real\n\n```\n# not a heading\nsome code\n```\n\nafter the fence.",
	"###### deep heading\ntext\n####### too deep not a heading\ntext",
	"#no space so not a heading\n# yes heading",
	"```unterminated fence\n# inside\nmore",
	"a.b.c.d.e.f.g.",
	"Sentence one! Sentence two? Sentence three. Done",
	"line1\nline2\nline3\nline4\nline5",
	"para\n\n\n\n\npara two",
	"  nbsp and unicode space words",
	"tab\there\tand\tthere",
	"...",
	"!?.",
	"a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng\n\nh\n\ni\n\nj",
}

// trimSpaceUnicode mirrors the chunkers' whitespace handling closely enough for
// assertions: the chunkers use strings.TrimSpace / strings.Fields, which treat
// unicode whitespace as separators.
func nonWhitespaceRunes(s string) []rune {
	var out []rune
	for _, r := range s {
		if !unicode.IsSpace(r) {
			out = append(out, r)
		}
	}
	return out
}

// FuzzNewChunkerSplit drives every strategy with fuzzed text and sizing, and
// asserts the Chunker contract: no panic, no empty/whitespace-only pieces, and
// determinism (identical output across two calls).
func FuzzNewChunkerSplit(f *testing.F) {
	for _, t := range adversarialTexts {
		// vary target/overlap across seeds, including degenerate sizings.
		f.Add(t, 0, 0)
		f.Add(t, 8, 2)
		f.Add(t, 1, 0)
		f.Add(t, 120, 24)
		f.Add(t, 3, 5) // overlap >= target, normalized by chunkBounds
		f.Add(t, -4, -9)
	}

	f.Fuzz(func(t *testing.T, text string, target, overlap int) {
		for _, strategy := range fuzzStrategies {
			cfg := ChunkConfig{Strategy: strategy, TargetWords: target, OverlapWords: overlap}
			ch := NewChunker(cfg)

			pieces := ch.Split(text)
			again := ch.Split(text)

			// Determinism: same input, same output (same chunker instance).
			if len(pieces) != len(again) {
				t.Fatalf("strategy %q non-deterministic length: %d vs %d for %q",
					strategy, len(pieces), len(again), text)
			}
			for i := range pieces {
				if pieces[i].Text != again[i].Text {
					t.Fatalf("strategy %q non-deterministic text at %d: %q vs %q",
						strategy, i, pieces[i].Text, again[i].Text)
				}
				if strings.Join(pieces[i].Headings, "\x00") != strings.Join(again[i].Headings, "\x00") {
					t.Fatalf("strategy %q non-deterministic headings at %d", strategy, i)
				}
			}

			// Contract: pieces are never empty after whitespace trimming. The
			// chunkers all flush only non-empty, TrimSpace'd joins.
			for i, p := range pieces {
				if strings.TrimSpace(p.Text) == "" {
					t.Fatalf("strategy %q produced an empty piece at %d for input %q",
						strategy, i, text)
				}
			}
		}
	})
}

// FuzzLocateSpan checks locateSpan never panics and always returns either a
// not-found sentinel or a valid, from-respecting [start,end) range. It also
// asserts the key positive property the document mapping relies on: a
// whitespace-normalized slice taken from the text is locatable.
func FuzzLocateSpan(f *testing.F) {
	seeds := []struct {
		text string
		sub  string
		from int
	}{
		{"hello world", "world", 0},
		{"hello   world", "hello world", 0},
		{"  leading and trailing  ", "leading and trailing", 0},
		{"abc abc abc", "abc", 1},
		{"abc abc abc", "abc", 100},
		{"", "x", 0},
		{"x", "", 0},
		{"   ", "   ", 0},
		{"line1\nline2", "line1 line2", 0},
		{"你好 世界", "你好 世界", 0},
		{"emoji 😀 here", "😀 here", 0},
		{"tabs\tand\nnewlines", "tabs and newlines", 0},
		{"a b c", "b c", -5},
		{"repeat repeat", "repeat", 7},
	}
	for _, s := range seeds {
		f.Add(s.text, s.sub, s.from)
	}

	f.Fuzz(func(t *testing.T, text, sub string, from int) {
		runes := []rune(text)
		start, end := locateSpan(runes, sub, from)

		if start == -1 && end == -1 {
			// not-found sentinel is always acceptable.
			return
		}

		// Any non-sentinel result must be a valid range.
		if start < 0 || end < 0 {
			t.Fatalf("partial sentinel: start=%d end=%d (text=%q sub=%q from=%d)",
				start, end, text, sub, from)
		}
		if !(start < end) {
			t.Fatalf("non-positive span: start=%d end=%d (text=%q sub=%q)", start, end, text, sub)
		}
		if start > len(runes) || end > len(runes) {
			t.Fatalf("span out of range: start=%d end=%d len=%d (text=%q sub=%q)",
				start, end, len(runes), text, sub)
		}
		// from-respecting when from is a valid index.
		if from >= 0 && from < len(runes) && start < from {
			t.Fatalf("span started before from: start=%d from=%d (text=%q sub=%q)",
				start, from, text, sub)
		}
		// The located runes, ignoring whitespace, must equal the needle's
		// non-whitespace runes. This is the exact guarantee matchFrom provides.
		got := nonWhitespaceRunes(string(runes[start:end]))
		want := nonWhitespaceRunes(sub)
		if string(got) != string(want) {
			t.Fatalf("located runes mismatch: got %q want %q (text=%q sub=%q span=[%d,%d))",
				string(got), string(want), text, sub, start, end)
		}
	})
}

// asciiWhitespaceOnly reports whether every whitespace rune in s is one of the
// four ASCII whitespace characters that locateSpan's isSpace recognizes.
// locateSpan does NOT treat Unicode whitespace (NBSP U+00A0, EM SPACE U+2003,
// etc.) as a separator, but the chunkers normalize on strings.Fields, which
// does. So a piece whose source spanned Unicode whitespace is legitimately not
// locatable. See the note in the file header comment near FuzzLocateSpanRoundTrip.
func asciiWhitespaceOnly(s string) bool {
	for _, r := range s {
		if unicode.IsSpace(r) && r != ' ' && r != '\t' && r != '\n' && r != '\r' {
			return false
		}
	}
	return true
}

// FuzzLocateSpanRoundTrip asserts the property chunkDoc depends on: a piece body
// produced by a real chunker can be located back in the original text. For every
// chunker we run, each piece must resolve to a valid span (or, acceptably, the
// not-found sentinel which the Chunk doc comment documents as best-effort).
//
// KNOWN NON-TEST QUIRK (reported, not fixed per instructions): the chunkers
// normalize whitespace with strings.Fields, which splits on ALL Unicode
// whitespace, but locateSpan.isSpace only recognizes ASCII space/tab/CR/LF.
// A document containing Unicode whitespace (e.g. NBSP U+00A0, EM SPACE U+2003)
// inside or around a chunk therefore yields the (-1,-1) sentinel rather than a
// real highlight span, silently losing the document-to-chunk mapping for those
// chunks. We tolerate the sentinel only when the source text contains non-ASCII
// whitespace; for pure-ASCII-whitespace text we still require a real location.
func FuzzLocateSpanRoundTrip(f *testing.F) {
	for _, t := range adversarialTexts {
		f.Add(t)
	}
	f.Fuzz(func(t *testing.T, text string) {
		runes := []rune(text)
		for _, strategy := range fuzzStrategies {
			ch := NewChunker(ChunkConfig{Strategy: strategy, TargetWords: 8, OverlapWords: 2})
			cursor := 0
			for _, p := range ch.Split(text) {
				start, end := locateSpan(runes, p.Text, cursor)
				if start == -1 && end == -1 {
					// A miss is only acceptable when Unicode whitespace is in play
					// (the known quirk above) or for an empty needle. Otherwise a
					// non-empty piece's runes do appear in the source, so surface
					// the miss as a failure.
					if strings.TrimSpace(p.Text) != "" && asciiWhitespaceOnly(text) {
						t.Fatalf("strategy %q: non-empty piece %q not locatable in %q",
							strategy, p.Text, text)
					}
					continue
				}
				if start < 0 || end < start || end > len(runes) {
					t.Fatalf("strategy %q: bad span [%d,%d) len=%d piece=%q",
						strategy, start, end, len(runes), p.Text)
				}
				cursor = start
			}
		}
	})
}

// FuzzChunkOffsets is the highest-value fuzzer: it replicates chunkDoc's exact
// offset logic (NewChunker -> Split -> locateSpan, forward cursor) and asserts
// that every produced offset is either the (-1,-1) sentinel or a real,
// in-range span whose runes match the piece body ignoring whitespace. This
// guards the document-to-chunk highlighting mapping against corruption.
func FuzzChunkOffsets(f *testing.F) {
	for _, t := range adversarialTexts {
		f.Add(t, 8, 2)
		f.Add(t, 1, 0)
		f.Add(t, 120, 24)
	}

	f.Fuzz(func(t *testing.T, text string, target, overlap int) {
		runes := []rune(text)
		for _, strategy := range fuzzStrategies {
			ch := NewChunker(ChunkConfig{Strategy: strategy, TargetWords: target, OverlapWords: overlap})
			pieces := ch.Split(text)

			cursor := 0
			for i, p := range pieces {
				// This mirrors rag/chunk.go chunkDoc exactly.
				start, end := locateSpan(runes, p.Text, cursor)
				if start >= 0 {
					cursor = start
				}

				if start == -1 && end == -1 {
					// Sentinel: acceptable per the Chunk doc comment (best-effort).
					continue
				}

				// Otherwise the span must be real and in range.
				if start < 0 || end < 0 {
					t.Fatalf("strategy %q chunk %d: partial sentinel start=%d end=%d (text=%q)",
						strategy, i, start, end, text)
				}
				if !(start < end) {
					t.Fatalf("strategy %q chunk %d: empty span [%d,%d) piece=%q",
						strategy, i, start, end, p.Text)
				}
				if start > len(runes) || end > len(runes) {
					t.Fatalf("strategy %q chunk %d: span out of range [%d,%d) len=%d",
						strategy, i, start, end, len(runes))
				}
				// The located text must be real, non-empty, and match the piece
				// body once whitespace is ignored.
				body := string(runes[start:end])
				if strings.TrimSpace(body) == "" {
					t.Fatalf("strategy %q chunk %d: located span is whitespace-only", strategy, i)
				}
				got := nonWhitespaceRunes(body)
				want := nonWhitespaceRunes(p.Text)
				if string(got) != string(want) {
					t.Fatalf("strategy %q chunk %d: located runes %q != piece runes %q (text=%q span=[%d,%d))",
						strategy, i, string(got), string(want), text, start, end)
				}
			}
		}
	})
}

// FuzzChunkOffsetsStore drives the same property through a real Store and the
// keyword test embedder, exercising the public ChunkDocument path end to end
// (heading prepending included). It asserts no panic and that every non-sentinel
// chunk offset is in range.
func FuzzChunkOffsetsStore(f *testing.F) {
	for _, t := range adversarialTexts {
		f.Add(t)
	}
	f.Fuzz(func(t *testing.T, text string) {
		for _, strategy := range fuzzStrategies {
			s := New(newKeywordEmbedder(32), Config{
				Seed:  1,
				Chunk: ChunkConfig{Strategy: strategy, TargetWords: 8, OverlapWords: 2},
			})
			chunks := s.ChunkDocument(Document{ID: "d", Text: text})
			runes := []rune(text)
			for i, c := range chunks {
				if c.Start == -1 && c.End == -1 {
					continue
				}
				if c.Start < 0 || c.End < 0 || c.Start >= c.End ||
					c.Start > len(runes) || c.End > len(runes) {
					t.Fatalf("strategy %q chunk %d: invalid offsets [%d,%d) len=%d",
						strategy, i, c.Start, c.End, len(runes))
				}
			}
		}
	})
}

// FuzzAtxHeading fuzzes the ATX heading parser: no panic, and when ok the level
// is within 1..6 and the title is the trimmed remainder.
func FuzzAtxHeading(f *testing.F) {
	seeds := []string{
		"# Title", "## Section", "###### Deep", "####### TooDeep",
		"#NoSpace", "   ## Indented heading  ", "#", "# ", "#  spaced  ",
		"not a heading", "", "#######", "## 你好", "## 😀 emoji",
		"#\t tab not space", "###    multiple spaces",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, line string) {
		level, title, ok := atxHeading(line)
		if !ok {
			if level != 0 || title != "" {
				t.Fatalf("not-ok heading returned level=%d title=%q for %q", level, title, line)
			}
			return
		}
		if level < 1 || level > 6 {
			t.Fatalf("heading level out of range: %d for %q", level, line)
		}
		if title != strings.TrimSpace(title) {
			t.Fatalf("title not trimmed: %q for %q", title, line)
		}
		if title == "" {
			t.Fatalf("ok heading with empty title for %q", line)
		}
	})
}

// FuzzSplitSentences fuzzes the sentence segmenter: no panic, no empty segments,
// and every segment is a trimmed, non-whitespace string whose non-whitespace
// runes are a subset (in order) of the source.
func FuzzSplitSentences(f *testing.F) {
	seeds := []string{
		"One. Two. Three.",
		"Dr. Smith left. Mr. Jones stayed.",
		"No terminator here",
		"Bang! Question? Period.",
		"a.b.c.d.",
		"你好。世界！再见？",
		"Spaces.    Many.   Sentences.",
		"", "   ", "...", "!?.",
		"Trailing space after period. ",
		"e.g. this and i.e. that.",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		sents := splitSentences(text)
		var rebuilt []rune
		for i, s := range sents {
			if s == "" {
				t.Fatalf("empty sentence at %d for %q", i, text)
			}
			if s != strings.TrimSpace(s) {
				t.Fatalf("sentence not trimmed at %d: %q", i, s)
			}
			rebuilt = append(rebuilt, nonWhitespaceRunes(s)...)
		}
		// The concatenated non-whitespace content of all sentences must equal the
		// source's non-whitespace content: splitting drops only whitespace.
		src := nonWhitespaceRunes(text)
		if string(rebuilt) != string(src) {
			t.Fatalf("sentence split lost/changed content: got %q want %q (text=%q)",
				string(rebuilt), string(src), text)
		}
	})
}
