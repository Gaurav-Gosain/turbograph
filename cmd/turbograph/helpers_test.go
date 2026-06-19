package main

import (
	"strings"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/rag"
)

func TestUIURL(t *testing.T) {
	cases := map[string]string{
		":8080":                 "http://localhost:8080",
		"localhost:9000":        "http://localhost:9000",
		"https://x.example.com": "https://x.example.com",
	}
	for in, want := range cases {
		if got := uiURL(in); got != want {
			t.Errorf("uiURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHumanRate(t *testing.T) {
	cases := map[float64]string{
		950:     "950",
		12000:   "12k",
		1500000: "1.5M",
	}
	for in, want := range cases {
		if got := humanRate(in); got != want {
			t.Errorf("humanRate(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestSplitCmd(t *testing.T) {
	got := splitCmd("pdftotext -q {in} -")
	want := []string{"pdftotext", "-q", "{in}", "-"}
	if len(got) != len(want) {
		t.Fatalf("splitCmd len %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("splitCmd[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestTruncate(t *testing.T) {
	long := strings.Repeat("word ", 100)
	got := truncate(long, 20)
	if len(got) > 23 { // 20 + "..."
		t.Errorf("truncate did not bound length: %d", len(got))
	}
	if !strings.HasSuffix(got, "...") {
		t.Errorf("truncate should mark elision: %q", got)
	}
	short := "a few words"
	if truncate(short, 200) != short {
		t.Errorf("short text should be unchanged")
	}
}

func TestResolvedVersion(t *testing.T) {
	// An explicit build-time version is reported verbatim.
	old := version
	defer func() { version = old }()
	version = "v1.2.3"
	if got := resolvedVersion(); got != "v1.2.3" {
		t.Errorf("resolvedVersion() = %q, want v1.2.3", got)
	}
}

func TestBuildPrompt(t *testing.T) {
	res := []rag.Retrieved{
		{Chunk: rag.Chunk{ID: "d#0", Text: "alpha"}},
		{Chunk: rag.Chunk{ID: "d#1", Text: "beta"}},
	}
	p := buildPrompt("what is alpha?", res)
	for _, want := range []string{"alpha", "beta", "what is alpha?", "Context:", "Answer:"} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt missing %q:\n%s", want, p)
		}
	}
}
