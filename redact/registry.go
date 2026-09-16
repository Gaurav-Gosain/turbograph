package redact

import (
	"regexp"
	"sort"
)

// Rule is one credential pattern.
type Rule struct {
	// Kind names what the rule matched, e.g. "aws-access-key-id".
	Kind string
	// Re is the credential pattern.
	Re *regexp.Regexp
	// Group is the submatch index to blank out; 0 means the whole match. It exists
	// so a rule can match "password: hunter2" and redact only "hunter2", keeping
	// the key visible so the redaction is legible to whoever reads the chunk later.
	Group int
}

// Registry holds a set of credential patterns, matched in registration order.
// Longer, more specific formats must be registered before generic ones so a
// generic assignment rule cannot claim a token a precise rule would have named;
// the built-in registry below follows that order. Register replaces an existing
// rule with the same Kind in place, keeping its priority position, so tuning a
// built-in pattern does not silently demote it behind the generic rules.
//
// A Registry is not safe for concurrent Register and Text calls. Build it fully
// during setup, then treat it as read-only during ingestion.
type Registry struct {
	// rules keep registration order; byKind maps a Kind to its position for
	// O(1) replace-by-kind during Register.
	rules  []Rule
	byKind map[string]int
}

// NewRegistry returns a Registry with the built-in credential patterns
// pre-registered.
func NewRegistry() *Registry {
	r := &Registry{byKind: make(map[string]int)}
	for _, rl := range builtinRules {
		r.Register(rl)
	}
	return r
}

// DefaultRegistry returns a fresh Registry with the built-in patterns. It mirrors
// extract.DefaultRegistry's shape so both dispatch layers read the same way; redact
// has nothing host-dependent to probe, so it is equivalent to NewRegistry.
func DefaultRegistry() *Registry { return NewRegistry() }

// Register adds a rule, replacing any prior entry with the same Kind. A replaced
// rule keeps its position, so its priority relative to the other patterns is
// unchanged.
func (r *Registry) Register(rl Rule) {
	if i, ok := r.byKind[rl.Kind]; ok {
		r.rules[i] = rl
		return
	}
	r.byKind[rl.Kind] = len(r.rules)
	r.rules = append(r.rules, rl)
}

// Has reports whether a rule with the Kind is registered.
func (r *Registry) Has(kind string) bool {
	_, ok := r.byKind[kind]
	return ok
}

// Kinds returns the registered rule kinds, sorted, for diagnostics.
func (r *Registry) Kinds() []string {
	kinds := make([]string, 0, len(r.rules))
	for _, rl := range r.rules {
		kinds = append(kinds, rl.Kind)
	}
	sort.Strings(kinds)
	return kinds
}

// Text removes credentials from s using this registry's rules, returning the
// cleaned text and what was found. Each secret becomes a [redacted:<kind>] marker,
// so the passage stays readable and a reader can see that something was taken out
// rather than silently losing a line.
func (r *Registry) Text(s string) (string, []Finding) {
	if s == "" {
		return s, nil
	}
	counts := map[string]int{}
	for _, rl := range r.rules {
		s = rl.Re.ReplaceAllStringFunc(s, func(m string) string {
			if rl.Group == 0 {
				counts[rl.Kind]++
				return "[redacted:" + rl.Kind + "]"
			}
			// Redact only the captured group, keeping the surrounding context. Splice by
			// the group's byte offsets rather than replacing the first occurrence of its
			// text: with default credentials like postgres://postgres:postgres@host the
			// password equals the scheme or the username, and a first-occurrence replace
			// redacts the scheme and leaves the password in the clear.
			loc := rl.Re.FindStringSubmatchIndex(m)
			gs, ge := loc[2*rl.Group], loc[2*rl.Group+1]
			if gs < 0 || ge <= gs {
				return m
			}
			val := m[gs:ge]
			// A placeholder is not a secret. Ingesting documentation that says
			// `api_key = YOUR_API_KEY_HERE` should not report a leak.
			if isPlaceholder(val) {
				return m
			}
			counts[rl.Kind]++
			return m[:gs] + "[redacted:" + rl.Kind + "]" + m[ge:]
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
