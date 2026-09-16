package redact

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestRegistryHasBuiltinKinds pins the default rule set so the shipped patterns
// cannot silently shrink, and that Kinds is sorted for diagnostics.
func TestRegistryHasBuiltinKinds(t *testing.T) {
	r := NewRegistry()
	for _, kind := range []string{
		"private-key", "aws-access-key-id", "github-token", "github-fine-grained-token",
		"anthropic-api-key", "openai-api-key", "slack-token", "stripe-key",
		"google-api-key", "slack-webhook", "npm-token", "url-credentials",
		"jwt", "assigned-secret",
	} {
		if !r.Has(kind) {
			t.Errorf("default registry missing kind %q", kind)
		}
	}
	if kinds := r.Kinds(); !slices.IsSorted(kinds) {
		t.Errorf("Kinds() not sorted: %v", kinds)
	}
}

// TestRegistryCustomRule verifies a caller-registered rule redacts through the
// registry the same way a built-in one does.
func TestRegistryCustomRule(t *testing.T) {
	r := NewRegistry()
	r.Register(Rule{Kind: "build-secret", Re: regexp.MustCompile(`squad-\d+`), Group: 0})
	got, fs := r.Text("deploy squad-42 to production")
	if !strings.Contains(got, "[redacted:build-secret]") {
		t.Errorf("custom rule did not redact: %q", got)
	}
	if len(fs) != 1 || fs[0].Kind != "build-secret" || fs[0].Count != 1 {
		t.Fatalf("unexpected findings: %+v", fs)
	}
}

// TestRegistryRegisterReplaces pins the replace-by-kind semantics: a re-registered
// rule fully takes over its kind, and the old pattern stops matching.
func TestRegistryRegisterReplaces(t *testing.T) {
	r := NewRegistry()
	r.Register(Rule{Kind: "squad", Re: regexp.MustCompile(`squad-\d+`), Group: 0})
	r.Register(Rule{Kind: "squad", Re: regexp.MustCompile(`troop-\d+`), Group: 0})
	got, fs := r.Text("squad-9 alert troop-7")
	// The first pattern was replaced, so squad-9 survives; troop-7 redacts.
	if strings.Contains(got, "troop") {
		t.Errorf("replaced rule still matched: %q", got)
	}
	if !strings.Contains(got, "[redacted:squad]") || !strings.Contains(got, "squad-9") {
		t.Errorf("unexpected result after replace: %q", got)
	}
	if len(fs) != 1 || fs[0].Count != 1 {
		t.Fatalf("unexpected findings: %+v", fs)
	}
}

// TestRegistryPriorityPreserved pins that replacing a built-in kind keeps its slot:
// a customized generic-prefix rule must still lose to a more specific pattern that
// appears earlier in the built-in order.
func TestRegistryPriorityPreserved(t *testing.T) {
	r := NewRegistry()
	// Overwrite github-token with a pattern that would also claim the AWS token if
	// priority were not preserved.
	r.Register(Rule{Kind: "github-token", Re: regexp.MustCompile(`\b[A-Z]{4}[0-9A-Z]{16}\b`), Group: 0})
	aws := strings.Join([]string{"AKIA", "IOSFODNN7EXAMPLE"}, "")
	got, fs := r.Text("both " + aws + " here")
	for _, f := range fs {
		if f.Kind == "github-token" {
			t.Errorf("generic replacement rule jumped ahead of the more specific aws rule: %q", got)
		}
	}
	foundAWS := false
	for _, f := range fs {
		if f.Kind == "aws-access-key-id" {
			foundAWS = true
		}
	}
	if !foundAWS {
		t.Errorf("aws-access-key-id rule did not fire first: %q (%+v)", got, fs)
	}
}
