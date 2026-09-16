package lexical

import (
	"strings"
	"unicode"

	"github.com/Gaurav-Gosain/turbograph/stemmer"
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
// tokenizeFunc calls fn once per token, allocating nothing. When stem is true,
// tokens are routed through the stemmer package (morphological stemming plus
// language stopwords), so inflected forms collapse into one posting list.
//
// The slice-returning tokenizer builds two []string per document: one from FieldsFunc
// holding every field including the stopwords it is about to discard, and one for the
// result. Indexing a corpus only ever wants to count terms, so both are garbage as soon
// as they are made, and at 100,000 chunks they were gigabytes of it and the single
// largest CPU cost of opening a store. The tokens are substrings of the input, so the
// scan itself allocates nothing at all.
func tokenizeFunc(text string, stem bool, fn func(tok string)) {
	start := -1
	for i, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 {
			emitToken(text[start:i], stem, fn)
			start = -1
		}
	}
	if start >= 0 {
		emitToken(text[start:], stem, fn)
	}
}

// emitToken applies the same filters the slice tokenizer applies: fold case, drop the
// too-short, drop the English stopwords. With stem enabled it additionally drops the
// language stopwords and stems every "word-like" token through stemmer.StemToken;
// exact tokens (containing digits, or look like code or an acronym) are kept verbatim
// so identifiers keep matching identifiers.
func emitToken(field string, stem bool, fn func(tok string)) {
	tok := strings.ToLower(field) // returns field unchanged when it is already lower ASCII
	if len(tok) < minTokenLen {
		return
	}
	if _, stop := stopwords[tok]; stop {
		return
	}
	if stem && !isExactToken(field) {
		tok = stemmer.StemToken(tok)
		if tok == "" {
			return
		}
	}
	fn(tok)
}

// isExactToken reports whether a token must not be stemmed: identifiers with a
// digit, and run-of-the-mill acronyms (all uppercase Latin). Those are exact
// spellings whose inflected form would be a different identifier.
func isExactToken(field string) bool {
	letters := 0
	upper := 0
	for _, r := range field {
		switch {
		case unicode.IsDigit(r):
			return true
		case unicode.IsUpper(r):
			letters++
			upper++
		case unicode.IsLower(r):
			letters++
		}
	}
	return letters >= 3 && upper == letters && !stemmer.HasCyrillic(field)
}

func tokenize(text string, stem bool) []string {
	if text == "" {
		return nil
	}
	out := make([]string, 0, len(text)/8+1)
	tokenizeFunc(text, stem, func(tok string) { out = append(out, tok) })
	return out
}
