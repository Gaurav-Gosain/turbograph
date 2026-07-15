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
	"sort"
	"strings"
)

// Finding is one credential found in a document.
type Finding struct {
	Kind  string // what it looked like, e.g. "aws-access-key-id"
	Count int    // how many times
}

// rule is one credential pattern.
type rule struct {
	kind string
	re   *regexp.Regexp
	// group is the submatch index to blank out; 0 means the whole match. It exists so a
	// rule can match "password: hunter2" and redact only "hunter2", keeping the key
	// visible so the redaction is legible to whoever reads the chunk later.
	group int
}

// rules are matched in order. Longer, more specific formats come first so that a
// generic assignment rule cannot claim a token a precise rule would have named.
var rules = []rule{
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

// Text removes credentials from s, returning the cleaned text and what was found.
// Each secret becomes a [redacted:<kind>] marker, so the passage stays readable and
// a reader can see that something was taken out rather than silently losing a line.
func Text(s string) (string, []Finding) {
	if s == "" {
		return s, nil
	}
	counts := map[string]int{}
	for _, r := range rules {
		s = r.re.ReplaceAllStringFunc(s, func(m string) string {
			if r.group == 0 {
				counts[r.kind]++
				return "[redacted:" + r.kind + "]"
			}
			// Redact only the captured group, keeping the surrounding context. Splice by
			// the group's byte offsets rather than replacing the first occurrence of its
			// text: with default credentials like postgres://postgres:postgres@host the
			// password equals the scheme or the username, and a first-occurrence replace
			// redacts the scheme and leaves the password in the clear.
			loc := r.re.FindStringSubmatchIndex(m)
			gs, ge := loc[2*r.group], loc[2*r.group+1]
			if gs < 0 || ge <= gs {
				return m
			}
			val := m[gs:ge]
			// A placeholder is not a secret. Ingesting documentation that says
			// `api_key = YOUR_API_KEY_HERE` should not report a leak.
			if isPlaceholder(val) {
				return m
			}
			counts[r.kind]++
			return m[:gs] + "[redacted:" + r.kind + "]" + m[ge:]
		})
	}
	if len(counts) == 0 {
		return s, nil
	}
	out := make([]Finding, 0, len(counts))
	for k, n := range counts {
		out = append(out, Finding{Kind: k, Count: n})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Kind < out[j].Kind })
	return s, out
}

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
