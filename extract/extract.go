// Package extract provides pluggable document text extraction for turbograph.
//
// turbograph is a graph/RAG engine: it indexes and reasons over text. Documents
// arrive in many formats, and Go cannot natively parse most of them (PDF, scanned
// images, office formats). Rather than vendor a fragile parser for every format,
// this package routes a file to a user-configured Extractor by file extension and,
// where needed, shells out to a battle-tested external tool.
//
// Two extractor implementations ship here:
//
//   - TextExtractor returns already-plain-text bytes unchanged.
//   - CommandExtractor runs an external command (for example pdftotext) that
//     converts the input bytes to plain text or markdown.
//
// Because CommandExtractor is just "run a command and read its text output", it is
// the integration point for arbitrary tooling. OCR engines such as PaddleOCR
// PP-OCRv6 can be wired in by supplying a small wrapper command that takes the
// input file and emits plain text or markdown to stdout (or to a {out} file). The
// user owns that wrapper; this package only orchestrates it.
package extract

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Extractor turns the raw bytes of a document into plain text. The filename is
// provided so implementations can preserve the original extension (some tools
// sniff format by extension) and produce clearer errors.
type Extractor interface {
	Extract(ctx context.Context, filename string, data []byte) (string, error)
}

// ErrEmptyOutput is returned when extraction succeeds at the process level but
// yields no text. Callers can use this to distinguish "the file genuinely has no
// extractable text" (for example a scanned PDF that needs OCR) from a real
// failure, and route such files to an OCR-backed Extractor instead.
var ErrEmptyOutput = errors.New("extract: no text extracted")

// TextExtractor treats the input as plain text and returns it verbatim. It exists
// so plain formats flow through the same Registry as binary formats, keeping the
// caller's dispatch logic uniform.
type TextExtractor struct{}

// Extract returns the input bytes as a string without modification.
func (TextExtractor) Extract(_ context.Context, _ string, data []byte) (string, error) {
	return string(data), nil
}

// defaultTimeout bounds a single external command when the incoming context has
// no deadline of its own. External converters can hang on malformed input, so we
// never want to block a RAG ingestion pipeline indefinitely.
const defaultTimeout = 120 * time.Second

// inToken and outToken are placeholders in a command template. inToken is required
// and is replaced with the path of a temp file holding the input bytes. outToken
// is optional; when present, extracted text is read from that file instead of the
// command's stdout.
const (
	inToken  = "{in}"
	outToken = "{out}"
)

// CommandExtractor converts bytes to text by invoking an external command.
//
// Template is the argv to run, with the first element being the program. The
// special token "{in}" is replaced by a temp file containing the input bytes; the
// optional token "{out}" is replaced by a temp output file path, in which case the
// result is read from that file rather than stdout.
type CommandExtractor struct {
	// Template is the command and its arguments, for example
	// {"pdftotext", "-q", "-enc", "UTF-8", "{in}", "-"}.
	Template []string

	// Timeout caps a single invocation when the caller's context carries no
	// deadline. Zero means use the package default.
	Timeout time.Duration
}

// Extract writes data to a temp file, runs the configured command, and returns the
// produced text. Temp files always get cleaned up, even on error.
func (c CommandExtractor) Extract(ctx context.Context, filename string, data []byte) (string, error) {
	if len(c.Template) == 0 {
		return "", errors.New("extract: command template is empty")
	}

	// Apply our own timeout only when the caller has not already bounded the work.
	// This respects a deadline the pipeline may have set while still protecting
	// against hangs in the common no-deadline case.
	if _, ok := ctx.Deadline(); !ok {
		timeout := c.Timeout
		if timeout <= 0 {
			timeout = defaultTimeout
		}
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// Preserve the original extension so extension-sniffing tools behave. A missing
	// extension is fine; the file simply has none.
	ext := filepath.Ext(filename)
	inFile, err := os.CreateTemp("", "turbograph-extract-in-*"+ext)
	if err != nil {
		return "", fmt.Errorf("extract: create input temp file: %w", err)
	}
	inPath := inFile.Name()
	defer os.Remove(inPath)

	if _, err := inFile.Write(data); err != nil {
		inFile.Close()
		return "", fmt.Errorf("extract: write input temp file: %w", err)
	}
	if err := inFile.Close(); err != nil {
		return "", fmt.Errorf("extract: close input temp file: %w", err)
	}

	// Resolve {out} before building argv so we know where to read the result from.
	var outPath string
	useOutFile := templateContains(c.Template, outToken)
	if useOutFile {
		outFile, err := os.CreateTemp("", "turbograph-extract-out-*"+ext)
		if err != nil {
			return "", fmt.Errorf("extract: create output temp file: %w", err)
		}
		outPath = outFile.Name()
		// We only need the path; the command writes the contents.
		outFile.Close()
		defer os.Remove(outPath)
	}

	argv := substituteTemplate(c.Template, inPath, outPath)

	// CommandContext kills the process if ctx is canceled or times out.
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	// Surface context errors plainly so callers can detect timeout/cancellation,
	// which Run() otherwise reports as a generic "signal: killed".
	if ctxErr := ctx.Err(); ctxErr != nil {
		return "", fmt.Errorf("extract: command %q: %w", argv[0], ctxErr)
	}
	if runErr != nil {
		tail := trimStderr(stderr.String())
		if tail != "" {
			return "", fmt.Errorf("extract: command %q failed: %w: %s", argv[0], runErr, tail)
		}
		return "", fmt.Errorf("extract: command %q failed: %w", argv[0], runErr)
	}

	var out []byte
	if useOutFile {
		out, err = os.ReadFile(outPath)
		if err != nil {
			return "", fmt.Errorf("extract: read output temp file: %w", err)
		}
	} else {
		out = stdout.Bytes()
	}

	// Trim trailing whitespace; many converters append a trailing newline or form
	// feed that is not meaningful text.
	result := strings.TrimRight(string(out), " \t\r\n\f\v")
	if result == "" {
		return "", ErrEmptyOutput
	}
	return result, nil
}

// templateContains reports whether any argument equals or embeds the token.
func templateContains(tmpl []string, token string) bool {
	for _, arg := range tmpl {
		if strings.Contains(arg, token) {
			return true
		}
	}
	return false
}

// substituteTemplate returns a copy of tmpl with {in} and {out} tokens replaced.
// It does not mutate the caller's slice, so a single CommandExtractor is safe for
// concurrent reuse.
func substituteTemplate(tmpl []string, inPath, outPath string) []string {
	argv := make([]string, len(tmpl))
	for i, arg := range tmpl {
		arg = strings.ReplaceAll(arg, inToken, inPath)
		arg = strings.ReplaceAll(arg, outToken, outPath)
		argv[i] = arg
	}
	return argv
}

// maxStderrTail caps how much of a failing command's stderr we echo, so a chatty
// tool cannot bloat the error message or leak large internal output.
const maxStderrTail = 2048

// trimStderr returns a trimmed tail of stderr suitable for an error message.
func trimStderr(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > maxStderrTail {
		s = "..." + s[len(s)-maxStderrTail:]
	}
	return s
}

// CommandFromTemplate builds a CommandExtractor from a command template using the
// default timeout.
func CommandFromTemplate(tmpl []string) Extractor {
	return CommandExtractor{Template: tmpl}
}

// PDFViaPdftotext returns an Extractor that uses poppler's pdftotext to extract
// text from PDFs. "-" tells pdftotext to write to stdout.
func PDFViaPdftotext() Extractor {
	return CommandFromTemplate([]string{"pdftotext", "-q", "-enc", "UTF-8", inToken, "-"})
}
