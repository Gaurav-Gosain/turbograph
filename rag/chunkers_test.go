package rag

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestNewChunkerDispatch(t *testing.T) {
	cases := map[string]string{
		StrategyRecursive: "recursiveChunker",
		StrategyWord:      "wordChunker",
		StrategyMarkdown:  "markdownChunker",
		StrategySentence:  "sentenceChunker",
		"":                "recursiveChunker", // unknown/empty -> recursive default
	}
	for strategy, want := range cases {
		got := typeName(NewChunker(ChunkConfig{Strategy: strategy, TargetWords: 50}))
		if !strings.Contains(got, want) {
			t.Errorf("strategy %q: got %s, want %s", strategy, got, want)
		}
	}
}

func typeName(c Chunker) string {
	switch c.(type) {
	case wordChunker:
		return "wordChunker"
	case recursiveChunker:
		return "recursiveChunker"
	case markdownChunker:
		return "markdownChunker"
	case sentenceChunker:
		return "sentenceChunker"
	}
	return "unknown"
}

func TestRecursiveKeepsParagraphsAndSize(t *testing.T) {
	// Three short paragraphs; a small target should keep each roughly intact and
	// never exceed the target by much.
	text := "First paragraph one two three.\n\nSecond paragraph four five six seven.\n\nThird paragraph eight nine ten."
	ch := recursiveChunker{target: 8, overlap: 2}
	pieces := ch.Split(text)
	if len(pieces) < 2 {
		t.Fatalf("expected multiple pieces, got %d", len(pieces))
	}
	for i, p := range pieces {
		if p.Text == "" {
			t.Fatalf("piece %d empty", i)
		}
		if wordCount(p.Text) > 12 { // target + overlap slack
			t.Errorf("piece %d unexpectedly large (%d words): %q", i, wordCount(p.Text), p.Text)
		}
	}
}

func TestSentenceChunkerDoesNotCutSentences(t *testing.T) {
	text := "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu. Nu xi omicron pi."
	ch := sentenceChunker{target: 8, overlap: 0}
	pieces := ch.Split(text)
	if len(pieces) == 0 {
		t.Fatal("no pieces")
	}
	// Every emitted piece must end at a sentence terminator.
	for i, p := range pieces {
		last := strings.TrimSpace(p.Text)
		if last == "" {
			t.Fatalf("piece %d empty", i)
		}
		c := last[len(last)-1]
		if c != '.' && c != '!' && c != '?' {
			t.Errorf("piece %d does not end at a sentence boundary: %q", i, p.Text)
		}
	}
}

func TestMarkdownChunkerAttachesBreadcrumbs(t *testing.T) {
	text := "# Title\n\nIntro words here.\n\n## Section A\n\nBody of section A with several words to chunk.\n\n## Section B\n\nBody of section B content here."
	ch := markdownChunker{target: 20, overlap: 0}
	pieces := ch.Split(text)
	if len(pieces) == 0 {
		t.Fatal("no pieces")
	}
	var sawSectionA, sawNested bool
	for _, p := range pieces {
		if len(p.Headings) > 0 && p.Headings[len(p.Headings)-1] == "Section A" {
			sawSectionA = true
			// the Title should be the breadcrumb root for a nested heading
			if p.Headings[0] != "Title" {
				t.Errorf("expected Title as breadcrumb root, got %v", p.Headings)
			}
			if len(p.Headings) == 2 {
				sawNested = true
			}
		}
	}
	if !sawSectionA {
		t.Error("no piece carried the Section A breadcrumb")
	}
	if !sawNested {
		t.Error("nested heading path (Title > Section A) not produced")
	}
}

func TestMarkdownIgnoresHeadingsInCodeFences(t *testing.T) {
	text := "# Real\n\n```\n# not a heading\nsome code\n```\n\nafter the fence."
	ch := markdownChunker{target: 50, overlap: 0}
	pieces := ch.Split(text)
	for _, p := range pieces {
		for _, h := range p.Headings {
			if h == "not a heading" {
				t.Errorf("a fenced '# not a heading' line was treated as a heading: %v", p.Headings)
			}
		}
	}
}

func TestChunkDocPrependsBreadcrumb(t *testing.T) {
	// Through the store: a markdown doc's chunks should carry the heading path in
	// their text (the contextual chunk header).
	s := New(newKeywordEmbedder(64), Config{Seed: 1, Chunk: ChunkConfig{Strategy: StrategyMarkdown, TargetWords: 30}})
	chunks := s.ChunkDocument(Document{ID: "d", Text: "# Guide\n\n## Setup\n\nInstall the thing and run it twice."})
	found := false
	for _, c := range chunks {
		if strings.Contains(c.Text, "Guide > Setup") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a chunk to carry the 'Guide > Setup' breadcrumb, got %+v", chunks)
	}
}

func TestBringYourOwnChunker(t *testing.T) {
	// A custom Chunker set on Config must be used instead of the built-ins.
	s := New(newKeywordEmbedder(64), Config{Seed: 1, Chunker: fixedTwoChunker{}})
	chunks := s.ChunkDocument(Document{ID: "d", Text: "anything at all here"})
	if len(chunks) != 2 || chunks[0].Text != "one" || chunks[1].Text != "two" {
		t.Fatalf("custom chunker not used: %+v", chunks)
	}
}

type fixedTwoChunker struct{}

func (fixedTwoChunker) Split(string) []Piece { return []Piece{{Text: "one"}, {Text: "two"}} }

func TestStrategySurvivesReload(t *testing.T) {
	ctx := context.Background()
	s := New(newKeywordEmbedder(64), Config{Seed: 1, Chunk: ChunkConfig{Strategy: StrategySentence, TargetWords: 50}})
	if err := s.Build(ctx, []Document{{ID: "a", Text: "One sentence here. Another sentence there."}}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		t.Fatal(err)
	}
	r, err := Load(newKeywordEmbedder(64), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.Config().Chunk.Strategy; got != StrategySentence {
		t.Errorf("strategy did not survive reload: %q", got)
	}
}

func TestSentenceLookahead(t *testing.T) {
	// A decimal and a lowercase continuation must not end a sentence; a real
	// uppercase start must.
	got := splitSentences("Pi is 3.14 and it is irrational. Newton studied it. e.g. this stays.")
	for _, s := range got {
		if strings.HasPrefix(s, "14") || strings.HasPrefix(s, "and") {
			t.Fatalf("split inside a decimal/clause: %q in %v", s, got)
		}
	}
	// The real boundary after "irrational." (next char 'N') should produce a
	// segment starting with "Newton".
	var hasNewton bool
	for _, s := range got {
		if strings.HasPrefix(s, "Newton") {
			hasNewton = true
		}
	}
	if !hasNewton {
		t.Fatalf("did not split at a real boundary: %v", got)
	}
}
