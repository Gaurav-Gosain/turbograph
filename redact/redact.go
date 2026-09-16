// Package redact strips credentials out of text before it is indexed.
//
// It exists because of how turbograph is actually used. An agent grows a knowledge
// base from whatever it has been looking at: shell output, config files, environment
// dumps, HTTP responses, CI logs. Sooner or later one of those contains an API key.
// And a .tg is a file you hand to someone else, so a secret that reaches the store
// does not merely sit there, it is packaged up and shared, and it survives in the
// version history even after the document is corrected.
//
// So redaction happens at ingest, before the text is chunked, embedded, or recorded
// as a version. The patterns are deliberately narrow: each one matches a credential
// format with a distinctive prefix or structure, because a redactor that fires on
// ordinary prose is a redactor people turn off.
package redact

import (
	"regexp"
	"strings"
)

// Finding is one credential found in a document.
type Finding struct {
	Kind  string // what it looked like, e.g. "aws-access-key-id"
	Count int    // how many times
}

// builtinRules are the credential patterns NewRegistry pre-registers, matched in
// order. Longer, more specific formats come first so that a generic assignment
// rule cannot claim a token a precise rule would have named.
var builtinRules = []Rule{
	// Private keys: the whole PEM block, which is multi-line.
	{"private-key", regexp.MustCompile(`(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----`), 0},

	// Provider tokens with distinctive prefixes.
	{"aws-access-key-id", regexp.MustCompile(`\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b`), 0},
	{"github-token", regexp.MustCompile(`\bgh[pousr]_[A-Za-z0-9]{36,}\b`), 0},
	{"github-fine-grained-token", regexp.MustCompile(`\bgithub_pat_[A-Za-z0-9_]{60,}\b`), 0},
	{"anthropic-api-key", regexp.MustCompile(`\bsk-ant-[A-Za-z0-9_\-]{20,}\b`), 0},
	{"openai-api-key", regexp.MustCompile(`\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b`), 0},
	{"slack-token", regexp.MustCompile(`\bxox[baprs]-[A-Za-z0-9\-]{10,}\b`), 0},
	{"stripe-key", regexp.MustCompile(`\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b`), 0},
	{"google-api-key", regexp.MustCompile(`\bAIza[0-9A-Za-z_\-]{35,}`), 0},
	{"slack-webhook", regexp.MustCompile(`https://hooks\.slack\.com/services/[A-Za-z0-9/+]{20,}`), 0},
	{"npm-token", regexp.MustCompile(`\bnpm_[A-Za-z0-9]{36}\b`), 0},

	// A URL carrying credentials: postgres://user:password@host. Redact the password
	// only, so the host stays readable and the passage still makes sense.
	{"url-credentials", regexp.MustCompile(`(?i)\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:([^\s@/]{3,})@`), 1},

	// JWTs: three base64url segments. The middle segment must start with "ey" (a JSON
	// object), which is what keeps this from matching dotted identifiers in prose.
	{"jwt", regexp.MustCompile(`\bey[A-Za-z0-9_\-]{8,}\.ey[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b`), 0},

	// Generic assignment, last: KEY=value / "api_key": "value" / password: value.
	// Deliberately requires a secret-ish key name AND a value with no spaces, because
	// this is the rule most likely to fire on ordinary prose.
	{"assigned-secret", regexp.MustCompile(
		`(?i)\b(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token|auth[_\-]?token|client[_\-]?secret|private[_\-]?key|passwd|password|secret)\b["']?\s*[:=]\s*["']?([^\s"',;]{8,})["']?`), 1},
}

// defaultRegistry is the single immutable set the package-level Text uses. The
// patterns are compile-time constants, so it is built once at init.
var defaultRegistry = NewRegistry()

// Text removes credentials from s using the default registry, returning the
// cleaned text and what was found. Each secret becomes a [redacted:<kind>] marker,
// so the passage stays readable and a reader can see that something was taken out
// rather than silently losing a line. For custom rule sets, build a Registry.
func Text(s string) (string, []Finding) { return defaultRegistry.Text(s) }

// placeholders are the values documentation uses where a secret would go. Reporting
// them as leaks would train people to ignore the warning, which is worse than not
// warning at all.
var placeholders = []string{
	"your", "example", "changeme", "placeholder", "redacted", "xxx", "todo",
	"<", "${", "{{", "insert", "replace", "dummy", "sample",
}

func isPlaceholder(v string) bool {
	l := strings.ToLower(v)
	for _, p := range placeholders {
		if strings.Contains(l, p) {
			return true
		}
	}
	// A value made of a single repeated character is a mask, not a key.
	if len(l) > 0 {
		same := true
		for i := 1; i < len(l); i++ {
			if l[i] != l[0] {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}

// Summary renders findings as one short line for a human or a log.
func Summary(fs []Finding) string {
	if len(fs) == 0 {
		return ""
	}
	parts := make([]string, len(fs))
	total := 0
	for i, f := range fs {
		parts[i] = f.Kind
		total += f.Count
	}
	n := "1 secret"
	if total > 1 {
		n = itoa(total) + " secrets"
	}
	return n + " redacted (" + strings.Join(parts, ", ") + ")"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
