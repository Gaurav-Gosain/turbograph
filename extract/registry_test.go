package extract

import (
	"context"
	"errors"
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

func TestNewRegistryHasTextExtensions(t *testing.T) {
	r := NewRegistry()
	for _, ext := range []string{"txt", "md", "markdown", "text"} {
		if !r.Has(ext) {
			t.Errorf("expected %q to be registered", ext)
		}
	}
	if r.Has("pdf") {
		t.Error("NewRegistry should not register pdf")
	}
}

func TestRegistryRoutingByExtension(t *testing.T) {
	r := NewRegistry()
	got, err := r.Extract(context.Background(), "notes.md", []byte("# hi"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "# hi" {
		t.Fatalf("want %q, got %q", "# hi", got)
	}
}

func TestRegistryCaseInsensitive(t *testing.T) {
	r := NewRegistry()
	// Uppercase extension on the file and mixed-case lookups should all resolve.
	got, err := r.Extract(context.Background(), "README.MD", []byte("body"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "body" {
		t.Fatalf("want %q, got %q", "body", got)
	}
	if !r.Has(".TXT") {
		t.Error("Has should accept a leading dot and uppercase")
	}
	if !r.Has("Md") {
		t.Error("Has should be case-insensitive")
	}
}

func TestRegistryRegisterReplaces(t *testing.T) {
	r := NewRegistry()
	sentinel := stubExtractor{out: "from-stub"}
	r.Register(".TXT", sentinel)
	got, err := r.Extract(context.Background(), "a.txt", []byte("original"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "from-stub" {
		t.Fatalf("Register did not replace existing extractor, got %q", got)
	}
}

func TestRegistryUnknownExtension(t *testing.T) {
	r := NewRegistry()
	_, err := r.Extract(context.Background(), "scan.pdf", []byte("data"))
	if err == nil {
		t.Fatal("expected error for unregistered extension")
	}
	if !strings.Contains(err.Error(), "pdf") {
		t.Errorf("error should name the missing extension: %v", err)
	}
	// Error should list what IS registered to aid debugging.
	if !strings.Contains(err.Error(), "txt") {
		t.Errorf("error should list registered extensions: %v", err)
	}
}

func TestRegistryNoExtension(t *testing.T) {
	r := NewRegistry()
	_, err := r.Extract(context.Background(), "LICENSE", []byte("data"))
	if err == nil {
		t.Fatal("expected error for file with no extension")
	}
	if !strings.Contains(err.Error(), "LICENSE") {
		t.Errorf("error should name the file: %v", err)
	}
}

func TestRegistryExtensionsSorted(t *testing.T) {
	r := NewRegistry()
	r.Register("pdf", stubExtractor{})
	got := r.Extensions()
	want := []string{"markdown", "md", "pdf", "text", "txt"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Extensions not sorted/complete:\nwant %v\ngot  %v", want, got)
	}
}

func TestRegistryPropagatesExtractorError(t *testing.T) {
	r := NewRegistry()
	boom := errors.New("downstream failure")
	r.Register("pdf", stubExtractor{err: boom})
	_, err := r.Extract(context.Background(), "x.pdf", []byte("data"))
	if !errors.Is(err, boom) {
		t.Fatalf("want downstream error, got: %v", err)
	}
}

func TestDefaultRegistryDoesNotErrorAtConstruction(t *testing.T) {
	r := DefaultRegistry()
	if r == nil {
		t.Fatal("DefaultRegistry returned nil")
	}
	// Text extractors are always present regardless of host tooling.
	if !r.Has("txt") {
		t.Error("text extractors should always be registered")
	}
}

func TestDefaultRegistryPDFMatchesPdftotextPresence(t *testing.T) {
	r := DefaultRegistry()
	_, lookErr := exec.LookPath("pdftotext")
	pdftotextPresent := lookErr == nil

	if pdftotextPresent {
		if !r.Has("pdf") {
			t.Error("pdftotext is on PATH; pdf should be registered")
		}
	} else {
		// Skip the positive assertion when the tool is absent.
		if r.Has("pdf") {
			t.Error("pdftotext absent; pdf should not be registered")
		}
		t.Skip("pdftotext not installed; skipping positive pdf assertion")
	}
}

func TestDefaultRegistryRealPDFExtraction(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		t.Skip("pdftotext not installed")
	}
	r := DefaultRegistry()
	got, err := r.Extract(context.Background(), "tiny.pdf", tinyPDF())
	if err != nil {
		t.Fatalf("extracting tiny pdf: %v", err)
	}
	if !strings.Contains(got, "Hello") {
		t.Fatalf("expected extracted text to contain %q, got %q", "Hello", got)
	}
}

// stubExtractor is a deterministic in-memory Extractor for registry tests.
type stubExtractor struct {
	out string
	err error
}

func (s stubExtractor) Extract(_ context.Context, _ string, _ []byte) (string, error) {
	if s.err != nil {
		return "", s.err
	}
	return s.out, nil
}

// tinyPDF returns the bytes of a minimal but valid single-page PDF that draws the
// text "Hello PDF". It is hand-built with correct xref offsets so pdftotext can
// parse it without external assets.
func tinyPDF() []byte {
	objs := []string{
		"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
		"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
		"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
		"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 20 100 Td (Hello PDF) Tj ET\nendstream\nendobj\n",
		"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
	}

	var b strings.Builder
	b.WriteString("%PDF-1.4\n")

	// Offsets for the xref table; index 0 is the free object header entry.
	offsets := make([]int, len(objs)+1)
	for i, o := range objs {
		offsets[i+1] = b.Len()
		b.WriteString(o)
	}

	xrefStart := b.Len()
	b.WriteString("xref\n")
	b.WriteString("0 ")
	b.WriteString(itoa(len(objs) + 1))
	b.WriteString("\n")
	b.WriteString("0000000000 65535 f \n")
	for i := 1; i <= len(objs); i++ {
		b.WriteString(pad10(offsets[i]))
		b.WriteString(" 00000 n \n")
	}
	b.WriteString("trailer\n<< /Size ")
	b.WriteString(itoa(len(objs) + 1))
	b.WriteString(" /Root 1 0 R >>\nstartxref\n")
	b.WriteString(itoa(xrefStart))
	b.WriteString("\n%%EOF")

	return []byte(b.String())
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

func pad10(n int) string {
	s := itoa(n)
	for len(s) < 10 {
		s = "0" + s
	}
	return s
}
