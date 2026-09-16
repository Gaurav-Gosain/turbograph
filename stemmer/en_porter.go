package stemmer

// porterStemmer реализует классический стеммер Портера (1980). Он заявляет
// только чистые строчные ASCII-токены; всё остальное (акцентированные буквы,
// цифры) передаётся насквозь через Claims, так что Stem таких токенов не
// увидит.
type porterStemmer struct{}

func init() { Register("en", porterStemmer{}) }

// Claims сообщает, что токен написан чистыми строчными латинскими буквами.
func (porterStemmer) Claims(token string) bool {
	if token == "" {
		return false
	}
	for i := 0; i < len(token); i++ {
		if token[i] < 'a' || token[i] > 'z' {
			return false
		}
	}
	return true
}

// IsStopword возвращает false: английские стоп-слова фильтрует сам
// токенизатор по своему базовому набору независимо от стемминга.
func (porterStemmer) IsStopword(string) bool { return false }

// Stem сводит слово к основе Портера.
func (porterStemmer) Stem(word string) string { return stemEN(word) }

func stemEN(word string) string {
	b := []byte(word)
	if len(b) <= 2 {
		return word
	}
	// Only pure lower ASCII words are stemmed; anything else (accented
	// letters, digits) passes through untouched.
	for _, c := range b {
		if c < 'a' || c > 'z' {
			return word
		}
	}

	hasSuf := func(s []byte, suf string) bool {
		if len(s) < len(suf) {
			return false
		}
		return string(s[len(s)-len(suf):]) == suf
	}
	repSuf := func(s []byte, suf, repl string) []byte {
		s = s[:len(s)-len(suf)]
		return append(s, repl...)
	}

	// Step 1a: plural and third-person singular -s.
	switch {
	case hasSuf(b, "sses"):
		b = repSuf(b, "sses", "ss")
	case hasSuf(b, "ies"):
		b = repSuf(b, "ies", "i")
	case hasSuf(b, "ss"):
		// unchanged
	case hasSuf(b, "s"):
		b = b[:len(b)-1]
	}

	// Step 1b: past participles and gerunds; then tidy any resulting
	// consonant cluster.
	if hasSuf(b, "eed") && len(b) > 3 && porterMeasure(b[:len(b)-3]) > 0 {
		b = repSuf(b, "eed", "ee")
	} else {
		var stem []byte
		switch {
		case hasSuf(b, "ed"):
			stem = b[:len(b)-2]
		case hasSuf(b, "ing"):
			stem = b[:len(b)-3]
		}
		if stem != nil && porterHasVowel(stem) {
			b = stem
			switch {
			case hasSuf(b, "at"), hasSuf(b, "bl"), hasSuf(b, "iz"):
				b = append(b, 'e')
			case len(b) >= 2 && b[len(b)-1] == b[len(b)-2] &&
				b[len(b)-1] != 'l' && b[len(b)-1] != 's' && b[len(b)-1] != 'z':
				b = b[:len(b)-1]
			case porterMeasure(b) == 1 && porterCVC(b):
				b = append(b, 'e')
			}
		}
	}

	// Step 1c: turn a terminal y into i after a vowel-bearing stem.
	if len(b) > 0 && b[len(b)-1] == 'y' && porterHasVowel(b[:len(b)-1]) {
		b[len(b)-1] = 'i'
	}

	// Step 2: common noun/adjective-to-noun suffixes.
	step2 := [][2]string{
		{"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"},
		{"izer", "ize"}, {"abli", "able"}, {"alli", "al"}, {"entli", "ent"},
		{"eli", "e"}, {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
		{"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
		{"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"},
		{"logi", "log"},
	}
	for _, r := range step2 {
		if hasSuf(b, r[0]) && len(b) > len(r[0]) && porterMeasure(b[:len(b)-len(r[0])]) > 0 {
			b = repSuf(b, r[0], r[1])
			break
		}
	}

	// Step 3: -icate/-ative/... -> short forms.
	step3 := [][2]string{
		{"icate", "ic"}, {"ative", ""}, {"alize", "al"}, {"iciti", "ic"},
		{"ical", "ic"}, {"ful", ""}, {"ness", ""},
	}
	for _, r := range step3 {
		if hasSuf(b, r[0]) && len(b) > len(r[0]) && porterMeasure(b[:len(b)-len(r[0])]) > 0 {
			b = repSuf(b, r[0], r[1])
			break
		}
	}

	// Step 4: delete residual suffixes when the stem is long enough.
	step4 := []string{"al", "ance", "ence", "er", "ic", "able", "ible", "ant",
		"ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"}
	for _, suf := range step4 {
		if !hasSuf(b, suf) {
			continue
		}
		if suf == "ion" {
			if len(b) < len(suf)+1 || (b[len(b)-len(suf)-1] != 's' && b[len(b)-len(suf)-1] != 't') {
				continue
			}
		}
		if porterMeasure(b[:len(b)-len(suf)]) > 1 {
			b = b[:len(b)-len(suf)]
		}
		break
	}

	// Step 5a: drop a final e unless it guards a short syllable.
	if hasSuf(b, "e") {
		stem := b[:len(b)-1]
		if porterMeasure(stem) > 1 || (porterMeasure(stem) == 1 && !porterCVC(stem)) {
			b = stem
		}
	}
	// Step 5b: collapse a doubled final l.
	if len(b) > 1 && b[len(b)-1] == 'l' && b[len(b)-2] == 'l' && porterMeasure(b) > 1 {
		b = b[:len(b)-1]
	}
	return string(b)
}

// porterConsonants classifies each byte of s as a consonant, following Porter's
// y rule: y is a consonant only when the preceding letter is also a consonant.
func porterConsonants(s []byte) []bool {
	c := make([]bool, len(s))
	for i, ch := range s {
		switch ch {
		case 'a', 'e', 'i', 'o', 'u':
			c[i] = false
		case 'y':
			c[i] = i > 0 && c[i-1]
		default:
			c[i] = true
		}
	}
	return c
}

// porterHasVowel reports whether s contains a vowel.
func porterHasVowel(s []byte) bool {
	for _, v := range porterConsonants(s) {
		if !v {
			return true
		}
	}
	return false
}

// porterMeasure is Porter's m: the number of vowel-consonant sequences in s.
func porterMeasure(s []byte) int {
	c := porterConsonants(s)
	m, i := 0, 0
	for i < len(s) {
		for i < len(s) && c[i] {
			i++
		}
		if i >= len(s) {
			break
		}
		for i < len(s) && !c[i] {
			i++
		}
		if i >= len(s) {
			break
		}
		for i < len(s) && c[i] {
			i++
		}
		m++
	}
	return m
}

// porterCVC reports whether s ends in a consonant-vowel-consonant sequence
// whose final consonant is not w, x or y (the sentential *o shape).
func porterCVC(s []byte) bool {
	if len(s) < 3 {
		return false
	}
	c := porterConsonants(s)
	i := len(s) - 3
	if !c[i] || c[i+1] || !c[i+2] {
		return false
	}
	switch s[len(s)-1] {
	case 'w', 'x', 'y':
		return false
	}
	return true
}
