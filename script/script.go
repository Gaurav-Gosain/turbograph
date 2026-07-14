// Package script runs operator-supplied programs as pipeline stages, so a
// document can be processed by code turbograph did not have to ship. A script is
// any executable file: a Go binary, a Python file, a shell script, anything the
// host can run. It speaks one JSON document on stdin and one on stdout, which is
// a contract every language implements in a few lines.
//
// # Trust boundary
//
// Scripts are registered by the OPERATOR, at startup, by pointing turbograph at a
// directory (Load). Callers, including the HTTP API and the web UI, may only refer
// to a script BY NAME, and only names that were discovered in that directory. A
// caller can never supply a path, an argument vector, or a shell string. This is
// deliberate: turbograph is routinely run as a server, and an endpoint that
// executed a caller-supplied command would hand remote code execution to anyone
// who could reach it. With no directory configured the feature is entirely off and
// there is nothing to attack.
//
// Programs are executed directly (no shell), so there is no command injection
// through arguments or filenames, and each run is bounded by a timeout that kills
// the process. This mirrors how the extract package already shells out for PDF and
// OCR.
package script

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// DefaultTimeout bounds a single script invocation. A script that hangs must not
// wedge an ingest.
const DefaultTimeout = 30 * time.Second

// Doc is the JSON contract a script speaks. turbograph writes one Doc to the
// script's stdin and reads one back from its stdout.
//
// On input, ID, Text and Meta describe the document as it stands at this point in
// the pipeline. On output, Text replaces the document's text and Meta, when
// present, replaces its metadata. Setting Drop tells turbograph to skip the
// document entirely, which is how a script filters out boilerplate or junk.
//
// The struct is additive: new fields can be introduced without breaking scripts
// that ignore them.
type Doc struct {
	ID   string          `json:"id"`
	Text string          `json:"text"`
	Meta json.RawMessage `json:"meta,omitempty"`
	Drop bool            `json:"drop,omitempty"`
}

// Registry is the set of scripts an operator has made available.
type Registry struct {
	scripts map[string]string // name -> absolute path
	timeout time.Duration
}

// Load discovers the executable files in dir and returns a registry of them, keyed
// by file name. A nil registry (an empty dir path) is valid and simply has no
// scripts, so callers need not special-case the feature being off.
//
// Only regular, executable files are registered. Directories, symlinked
// directories, and non-executable files are ignored, so dropping a README beside
// the scripts is harmless.
func Load(dir string, timeout time.Duration) (*Registry, error) {
	r := &Registry{scripts: map[string]string{}, timeout: timeout}
	if r.timeout <= 0 {
		r.timeout = DefaultTimeout
	}
	if strings.TrimSpace(dir) == "" {
		return r, nil
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("script: resolve %q: %w", dir, err)
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return nil, fmt.Errorf("script: read scripts dir: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil || !isExecutable(info) {
			continue
		}
		r.scripts[e.Name()] = filepath.Join(abs, e.Name())
	}
	return r, nil
}

// isExecutable reports whether a regular file carries an executable bit.
func isExecutable(fi fs.FileInfo) bool {
	return fi.Mode().IsRegular() && fi.Mode().Perm()&0o111 != 0
}

// Names returns the registered script names, sorted, for listing in a UI.
func (r *Registry) Names() []string {
	if r == nil {
		return nil
	}
	out := make([]string, 0, len(r.scripts))
	for n := range r.scripts {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Has reports whether name is a registered script. Because lookup is an exact
// match against the discovered set, a name containing a path, "..", or a shell
// metacharacter simply is not found.
func (r *Registry) Has(name string) bool {
	if r == nil {
		return false
	}
	_, ok := r.scripts[name]
	return ok
}

// Len reports how many scripts are registered.
func (r *Registry) Len() int {
	if r == nil {
		return 0
	}
	return len(r.scripts)
}

// Run pipes in to the named script and returns the document it writes back. The
// script is executed directly, with no shell, under a timeout that kills it. A
// non-zero exit, a timeout, or output that is not a Doc is an error, and the
// script's stderr is included so the author can see what went wrong. Callers
// isolate that error to the one document rather than failing the whole ingest.
func (r *Registry) Run(ctx context.Context, name string, in Doc) (Doc, error) {
	path, ok := r.scripts[name]
	if !ok {
		return Doc{}, fmt.Errorf("script: no script named %q", name)
	}
	payload, err := json.Marshal(in)
	if err != nil {
		return Doc{}, fmt.Errorf("script %s: encode input: %w", name, err)
	}

	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, path) // no shell: argv is exactly the program
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	// Killing the script on timeout is not enough on its own: Run also waits for the
	// output pipes to close, and a child the script spawned still holds them open,
	// so a hung grandchild would keep blocking the ingest long after the deadline.
	// WaitDelay bounds that wait and forces the pipes shut.
	cmd.WaitDelay = 2 * time.Second

	runErr := cmd.Run()
	// A timeout surfaces from Run as "signal: killed", so report the context error
	// plainly instead, which is what the caller can act on.
	if ctxErr := ctx.Err(); ctxErr != nil {
		return Doc{}, fmt.Errorf("script %s: %w", name, ctxErr)
	}
	if runErr != nil {
		if tail := trimStderr(stderr.String()); tail != "" {
			return Doc{}, fmt.Errorf("script %s failed: %w: %s", name, runErr, tail)
		}
		return Doc{}, fmt.Errorf("script %s failed: %w", name, runErr)
	}

	var out Doc
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &out); err != nil {
		if tail := trimStderr(stderr.String()); tail != "" {
			return Doc{}, fmt.Errorf("script %s: output is not a JSON document: %w: %s", name, err, tail)
		}
		return Doc{}, fmt.Errorf("script %s: output is not a JSON document: %w", name, err)
	}
	if !out.Drop && strings.TrimSpace(out.Text) == "" {
		return Doc{}, fmt.Errorf("script %s: returned empty text (set \"drop\":true to skip a document)", name)
	}
	return out, nil
}

// trimStderr reduces a script's stderr to a short, single-line tail suitable for
// an error message, so a chatty program cannot flood the log or the UI.
func trimStderr(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	if n := len(lines); n > 3 {
		lines = lines[n-3:]
	}
	s = strings.Join(lines, "; ")
	const max = 300
	if len(s) > max {
		s = s[:max] + "..."
	}
	return s
}
