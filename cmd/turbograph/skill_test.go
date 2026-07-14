package main

import (
	"strings"
	"testing"
)

// TestSkillIsEmbedded guards the go:embed: a binary whose skill command prints
// nothing is worse than one without the command, because the failure is silent.
func TestSkillIsEmbedded(t *testing.T) {
	if len(agentSkill) < 500 {
		t.Fatalf("the embedded skill is %d bytes; it did not embed", len(agentSkill))
	}
	// Harnesses key off the front matter; without it the file is just prose. Accept CRLF
	// as well as LF: a Windows checkout can rewrite the file's line endings, and a test
	// that only knows about LF then fails for a reason that has nothing to do with the
	// skill. .gitattributes pins this too; the test does not depend on that holding.
	if !strings.HasPrefix(agentSkill, "---\n") && !strings.HasPrefix(agentSkill, "---\r\n") {
		t.Errorf("the skill has no YAML front matter; it begins %q", head(agentSkill, 8))
	}
	for _, want := range []string{"name: turbograph", "description:"} {
		if !strings.Contains(agentSkill, want) {
			t.Errorf("the skill front matter is missing %q", want)
		}
	}
	// It must document the commands it tells an agent to run. A skill that names a
	// command the binary does not have sends the agent in a circle.
	for _, cmd := range []string{
		"turbograph add", "turbograph search", "turbograph ask",
		"turbograph docs", "turbograph forget", "turbograph merge", "turbograph entities",
	} {
		if !strings.Contains(agentSkill, cmd) {
			t.Errorf("the skill never mentions %q", cmd)
		}
	}
	for _, env := range []string{"TURBOGRAPH_STORE", "TURBOGRAPH_MODEL"} {
		if !strings.Contains(agentSkill, env) {
			t.Errorf("the skill never mentions %s", env)
		}
	}
}

// head returns the leading n bytes of s, so a failure says what it actually saw.
func head(s string, n int) string {
	if len(s) < n {
		n = len(s)
	}
	return s[:n]
}
