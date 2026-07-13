package main

import (
	"strings"
	"testing"
)

func TestSliceLines(t *testing.T) {
	text := "one\ntwo\nthree\nfour\nfive"
	cases := []struct{ spec, want string }{
		{"", text},            // no spec: whole document
		{"2:3", "two\nthree"}, // inclusive 1-based range
		{"4", "four\nfive"},   // start-to-end
		{"1:99", text},        // end clamped to the document
		{"3:1", "three"},      // reversed range collapses to the start line
		{"nonsense", text},    // malformed spec falls back to the whole text
		{"99:100", ""},        // start past the end
		{"1:1", "one"},        // single line
	}
	for _, c := range cases {
		if got := sliceLines(text, c.spec); got != c.want {
			t.Errorf("sliceLines(%q) = %q, want %q", c.spec, got, c.want)
		}
	}
}

func TestClipBytesAndBudget(t *testing.T) {
	if got, cut := clipBytes("hello", 100); got != "hello" || cut {
		t.Errorf("under budget should pass through untouched, got %q cut=%v", got, cut)
	}
	got, cut := clipBytes("hello world", 5)
	if got != "hello" || !cut {
		t.Errorf("clipBytes = %q cut=%v, want %q true", got, cut, "hello")
	}
	// Multi-byte runes must not be cut mid-character.
	s := strings.Repeat("é", 10) // 2 bytes each
	out, cut := clipBytes(s, 5)  // 5 bytes lands mid-rune
	if !cut {
		t.Error("expected truncation")
	}
	if !isValidUTF8(out) {
		t.Errorf("clipBytes produced invalid UTF-8: %q", out)
	}
	// Budget defaults when unset or negative.
	if budget(0) != 20000 || budget(-5) != 20000 {
		t.Error("unset budget should default")
	}
	if budget(123) != 123 {
		t.Error("explicit budget should be honoured")
	}
}

func isValidUTF8(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}
