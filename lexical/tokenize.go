package lexical

import (
	"strings"
	"unicode"
)

// minTokenLen drops single-character tokens, which carry little discriminative
// signal and inflate postings lists.
const minTokenLen = 2

// stopwords is a small, high-frequency English set. Stopwords appear in nearly
// every document, so their inverse document frequency is near zero and they add
// noise without improving ranking. A compact set is deliberate: aggressive
// stopword removal can drop terms that matter for some queries (for example
// "to be or not to be"), so we trim only the most common function words.
var stopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {},
	"by": {}, "for": {}, "from": {}, "has": {}, "he": {}, "in": {}, "is": {},
	"it": {}, "its": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
	"to": {}, "was": {}, "were": {}, "will": {}, "with": {}, "this": {},
	"these": {}, "those": {}, "but": {}, "not": {}, "they": {}, "their": {},
}

// tokenize lowercases text, splits on any non-alphanumeric rune, and drops
// short tokens and stopwords. It is deterministic and allocation-conscious: the
// result slice is preallocated from a rune-count heuristic.
func tokenize(text string) []string {
	if text == "" {
		return nil
	}
	out := make([]string, 0, len(text)/8+1)
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	for _, f := range fields {
		tok := strings.ToLower(f)
		if len(tok) < minTokenLen {
			continue
		}
		if _, stop := stopwords[tok]; stop {
			continue
		}
		out = append(out, tok)
	}
	return out
}
