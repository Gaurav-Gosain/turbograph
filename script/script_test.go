package script

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// write drops an executable script into dir and returns its name.
func write(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return name
}

func skipOnWindows(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell scripts are not executable on windows")
	}
}

func TestRunTransformsDocument(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	// A shell script speaking the JSON contract: uppercase-ish transform plus a
	// metadata field, showing the whole surface in one go.
	write(t, dir, "shout.sh", `#!/bin/sh
read -r line
text=$(printf '%s' "$line" | sed 's/.*"text":"\([^"]*\)".*/\1/')
printf '{"text":"%s!","meta":{"shouted":true}}' "$text"
`)
	reg, err := Load(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reg.Has("shout.sh") || reg.Len() != 1 {
		t.Fatalf("script not discovered: %v", reg.Names())
	}
	out, err := reg.Run(context.Background(), "shout.sh", Doc{ID: "d", Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if out.Text != "hello!" {
		t.Errorf("text = %q, want %q", out.Text, "hello!")
	}
	var meta map[string]any
	if err := json.Unmarshal(out.Meta, &meta); err != nil {
		t.Fatalf("meta not JSON: %v", err)
	}
	if meta["shouted"] != true {
		t.Errorf("script could not set metadata: %v", meta)
	}
}

func TestRunCanDropDocument(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	write(t, dir, "drop.sh", "#!/bin/sh\ncat >/dev/null\nprintf '{\"drop\":true}'\n")
	reg, _ := Load(dir, 0)
	out, err := reg.Run(context.Background(), "drop.sh", Doc{ID: "d", Text: "junk"})
	if err != nil {
		t.Fatalf("dropping a document must not be an error: %v", err)
	}
	if !out.Drop {
		t.Error("expected Drop to be set")
	}
}

func TestRunSurfacesFailure(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	write(t, dir, "boom.sh", "#!/bin/sh\ncat >/dev/null\necho 'the parser exploded' >&2\nexit 3\n")
	reg, _ := Load(dir, 0)
	_, err := reg.Run(context.Background(), "boom.sh", Doc{ID: "d", Text: "x"})
	if err == nil {
		t.Fatal("a failing script must be an error")
	}
	// The author needs to see why, so stderr must reach the message.
	if !strings.Contains(err.Error(), "the parser exploded") {
		t.Errorf("stderr not surfaced in error: %v", err)
	}
}

func TestRunRejectsNonJSONOutput(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	write(t, dir, "prose.sh", "#!/bin/sh\ncat >/dev/null\necho 'I am not JSON'\n")
	reg, _ := Load(dir, 0)
	if _, err := reg.Run(context.Background(), "prose.sh", Doc{ID: "d", Text: "x"}); err == nil {
		t.Fatal("non-JSON output must be an error, not silently ignored")
	}
}

func TestRunRejectsEmptyText(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	write(t, dir, "empty.sh", "#!/bin/sh\ncat >/dev/null\nprintf '{\"text\":\"\"}'\n")
	reg, _ := Load(dir, 0)
	_, err := reg.Run(context.Background(), "empty.sh", Doc{ID: "d", Text: "x"})
	if err == nil {
		t.Fatal("emptying a document silently is almost always a bug; it must error")
	}
	if !strings.Contains(err.Error(), "drop") {
		t.Errorf("the error should point at the drop field: %v", err)
	}
}

func TestRunTimesOut(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	write(t, dir, "hang.sh", "#!/bin/sh\ncat >/dev/null\nsleep 30\n")
	reg, _ := Load(dir, 150*time.Millisecond)
	start := time.Now()
	_, err := reg.Run(context.Background(), "hang.sh", Doc{ID: "d", Text: "x"})
	if err == nil {
		t.Fatal("a hanging script must be killed, not wedge the ingest")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("timeout did not kill the process promptly: %v", elapsed)
	}
}

// TestRegistryIsClosed is the security property: a caller can only reach scripts
// the operator put in the directory, and only by exact name. Anything that looks
// like a path, a traversal, or a shell string is simply not a registered name.
func TestRegistryIsClosed(t *testing.T) {
	skipOnWindows(t)
	dir := t.TempDir()
	write(t, dir, "ok.sh", "#!/bin/sh\ncat >/dev/null\nprintf '{\"text\":\"fine\"}'\n")
	reg, _ := Load(dir, 0)

	for _, bad := range []string{
		"/bin/sh",
		"../../bin/sh",
		"ok.sh; rm -rf /",
		"$(whoami)",
		"bin/../ok.sh",
		"",
	} {
		if reg.Has(bad) {
			t.Errorf("registry accepted %q as a script name", bad)
		}
		if _, err := reg.Run(context.Background(), bad, Doc{ID: "d", Text: "x"}); err == nil {
			t.Errorf("registry executed %q", bad)
		}
	}
	// The legitimate name still works.
	if _, err := reg.Run(context.Background(), "ok.sh", Doc{ID: "d", Text: "x"}); err != nil {
		t.Fatalf("registered script should run: %v", err)
	}
}

func TestLoadIgnoresNonExecutables(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("not a script"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "subdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	reg, err := Load(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	if reg.Len() != 0 {
		t.Errorf("only executable regular files should register, got %v", reg.Names())
	}
}

func TestLoadOffByDefault(t *testing.T) {
	reg, err := Load("", 0)
	if err != nil {
		t.Fatal(err)
	}
	if reg.Len() != 0 || reg.Has("anything") {
		t.Error("with no directory configured the feature must be entirely off")
	}
	if _, err := reg.Run(context.Background(), "anything", Doc{Text: "x"}); err == nil {
		t.Error("a registry with no scripts must not execute anything")
	}
}

// TestIsExecutableIsPlatformCorrect: Unix decides by permission bit, Windows by
// extension. Testing for a permission bit on Windows would register nothing and
// the feature would look silently broken.
func TestIsExecutableIsPlatformCorrect(t *testing.T) {
	dir := t.TempDir()
	name := "prog.bat"
	if runtime.GOOS != "windows" {
		name = "prog.sh"
	}
	body := "#!/bin/sh\ncat >/dev/null\nprintf '{\"text\":\"ok\"}'\n"
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	reg, err := Load(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reg.Has(name) {
		t.Fatalf("an executable program must register on %s, got %v", runtime.GOOS, reg.Names())
	}
}
