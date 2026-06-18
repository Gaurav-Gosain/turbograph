package extract

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestTextExtractorReturnsInputUnchanged(t *testing.T) {
	in := []byte("# Title\n\nSome text with trailing space   \nand a final newline\n")
	got, err := TextExtractor{}.Extract(context.Background(), "doc.md", in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != string(in) {
		t.Fatalf("text mutated:\nwant %q\ngot  %q", string(in), got)
	}
}

func TestTextExtractorEmptyInput(t *testing.T) {
	got, err := TextExtractor{}.Extract(context.Background(), "empty.txt", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Fatalf("want empty string, got %q", got)
	}
}

func TestCommandExtractorStdout(t *testing.T) {
	if _, err := exec.LookPath("cat"); err != nil {
		t.Skip("cat not available")
	}
	want := "hello turbograph"
	ce := CommandFromTemplate([]string{"cat", inToken})
	got, err := ce.Extract(context.Background(), "x.txt", []byte(want+"\n\n"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Fatalf("want %q, got %q (trailing whitespace should be trimmed)", want, got)
	}
}

func TestCommandExtractorTransforms(t *testing.T) {
	if _, err := exec.LookPath("tr"); err != nil {
		t.Skip("tr not available")
	}
	// tr reads stdin, so feed via a shell-free pipeline is awkward; instead use
	// sh to redirect the {in} file into tr. Guarded below.
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not available")
	}
	ce := CommandFromTemplate([]string{"sh", "-c", "tr a b < " + inToken})
	got, err := ce.Extract(context.Background(), "x.txt", []byte("banana"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "bbnbnb" {
		t.Fatalf("want %q, got %q", "bbnbnb", got)
	}
}

func TestCommandExtractorOutFile(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not available")
	}
	ce := CommandFromTemplate([]string{"sh", "-c", "cat " + inToken + " > " + outToken})
	want := "routed through an output file"
	got, err := ce.Extract(context.Background(), "x.txt", []byte(want+"\n"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Fatalf("want %q, got %q", want, got)
	}
}

func TestCommandExtractorPreservesExtension(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not available")
	}
	// Echo the basename of the input temp file so we can assert the original
	// extension was preserved (tools that sniff by extension rely on this).
	ce := CommandFromTemplate([]string{"sh", "-c", "basename " + inToken})
	got, err := ce.Extract(context.Background(), "report.pdf", []byte("data"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasSuffix(got, ".pdf") {
		t.Fatalf("temp input file did not preserve .pdf extension, got %q", got)
	}
}

func TestCommandExtractorNonZeroExitIncludesStderr(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not available")
	}
	ce := CommandFromTemplate([]string{"sh", "-c", "echo boom-on-stderr 1>&2; exit 3"})
	_, err := ce.Extract(context.Background(), "x.txt", []byte("ignored"))
	if err == nil {
		t.Fatal("expected error on non-zero exit, got nil")
	}
	if !strings.Contains(err.Error(), "boom-on-stderr") {
		t.Fatalf("error should include stderr tail, got: %v", err)
	}
}

func TestCommandExtractorEmptyOutput(t *testing.T) {
	if _, err := exec.LookPath("true"); err != nil {
		t.Skip("true not available")
	}
	ce := CommandFromTemplate([]string{"true"})
	_, err := ce.Extract(context.Background(), "x.txt", []byte("anything"))
	if !errors.Is(err, ErrEmptyOutput) {
		t.Fatalf("want ErrEmptyOutput, got: %v", err)
	}
}

func TestCommandExtractorWhitespaceOnlyIsEmpty(t *testing.T) {
	if _, err := exec.LookPath("printf"); err != nil {
		t.Skip("printf not available")
	}
	ce := CommandFromTemplate([]string{"printf", "\n\n   \t\n"})
	_, err := ce.Extract(context.Background(), "x.txt", []byte("anything"))
	if !errors.Is(err, ErrEmptyOutput) {
		t.Fatalf("whitespace-only output should be empty, got: %v", err)
	}
}

func TestCommandExtractorEmptyTemplate(t *testing.T) {
	ce := CommandExtractor{}
	_, err := ce.Extract(context.Background(), "x.txt", []byte("data"))
	if err == nil {
		t.Fatal("expected error for empty template")
	}
}

func TestCommandExtractorAlreadyCanceledContext(t *testing.T) {
	if _, err := exec.LookPath("cat"); err != nil {
		t.Skip("cat not available")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ce := CommandFromTemplate([]string{"cat", inToken})
	_, err := ce.Extract(ctx, "x.txt", []byte("hello"))
	if err == nil {
		t.Fatal("expected error from canceled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got: %v", err)
	}
}

func TestCommandExtractorTimeout(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not available")
	}
	ce := CommandExtractor{
		Template: []string{"sleep", "30"},
		Timeout:  50 * time.Millisecond,
	}
	start := time.Now()
	_, err := ce.Extract(context.Background(), "x.txt", []byte("data"))
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want context.DeadlineExceeded, got: %v", err)
	}
	// Must return well before the 30s sleep would have finished.
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("timeout not honored, took %v", elapsed)
	}
}

func TestCommandExtractorCallerDeadlineRespected(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not available")
	}
	// A short caller deadline should win even though the extractor's own Timeout
	// is large, proving we do not override an existing deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	ce := CommandExtractor{
		Template: []string{"sleep", "30"},
		Timeout:  10 * time.Minute,
	}
	start := time.Now()
	_, err := ce.Extract(ctx, "x.txt", []byte("data"))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want context.DeadlineExceeded, got: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("caller deadline not honored, took %v", elapsed)
	}
}

func TestCommandExtractorCleansUpTempFiles(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not available")
	}
	// Point temp creation at an isolated dir so we can count residue precisely.
	dir := t.TempDir()
	t.Setenv("TMPDIR", dir)

	before := countFiles(t, dir)

	ce := CommandFromTemplate([]string{"sh", "-c", "cat " + inToken + " > " + outToken})
	if _, err := ce.Extract(context.Background(), "x.txt", []byte("payload")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if after := countFiles(t, dir); after != before {
		t.Fatalf("temp files leaked: before=%d after=%d", before, after)
	}
}

func TestCommandExtractorCleansUpTempFilesOnError(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("sh not available")
	}
	dir := t.TempDir()
	t.Setenv("TMPDIR", dir)
	before := countFiles(t, dir)

	ce := CommandFromTemplate([]string{"sh", "-c", "echo nope 1>&2; exit 1"})
	if _, err := ce.Extract(context.Background(), "x.pdf", []byte("payload")); err == nil {
		t.Fatal("expected error")
	}

	if after := countFiles(t, dir); after != before {
		t.Fatalf("temp files leaked on error: before=%d after=%d", before, after)
	}
}

func countFiles(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read temp dir: %v", err)
	}
	return len(entries)
}

func TestSubstituteTemplateDoesNotMutate(t *testing.T) {
	tmpl := []string{"tool", inToken, outToken}
	got := substituteTemplate(tmpl, "/tmp/in", "/tmp/out")
	if tmpl[1] != inToken || tmpl[2] != outToken {
		t.Fatalf("original template mutated: %v", tmpl)
	}
	want := []string{"tool", "/tmp/in", "/tmp/out"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("substitution wrong at %d: want %q got %q", i, want[i], got[i])
		}
	}
}

func TestTrimStderrCapsLength(t *testing.T) {
	long := strings.Repeat("x", maxStderrTail*2)
	got := trimStderr(long)
	if len(got) > maxStderrTail+3 {
		t.Fatalf("stderr tail not capped: len=%d", len(got))
	}
	if !strings.HasPrefix(got, "...") {
		t.Fatalf("truncated tail should be marked with ellipsis: %q", got[:10])
	}
}

func TestPDFViaPdftotextTemplate(t *testing.T) {
	ce, ok := PDFViaPdftotext().(CommandExtractor)
	if !ok {
		t.Fatal("PDFViaPdftotext should return a CommandExtractor")
	}
	if len(ce.Template) == 0 || ce.Template[0] != "pdftotext" {
		t.Fatalf("unexpected template: %v", ce.Template)
	}
	if !templateContains(ce.Template, inToken) {
		t.Fatalf("template must contain %q: %v", inToken, ce.Template)
	}
}
