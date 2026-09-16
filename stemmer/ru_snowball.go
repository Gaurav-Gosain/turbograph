package stemmer

import "strings"

// snowballRU реализует русский стеммер Snowball (https://snowballstem.org/
// algorithms/russian/stemmer.html) с нормализацией ё в е, как того требует
// референсная реализация. Тот же набор стоп-слов, что в официальном
// дистрибутиве, стоп-слова проверяются с той же нормализацией.
type snowballRU struct{}

func init() { Register("ru", snowballRU{}) }

// Claims сообщает, что в токене есть кириллические буквы.
func (snowballRU) Claims(token string) bool { return HasCyrillic(token) }

// IsStopword проверяет токен по русскому стоп-списку Snowball (с ё→е).
func (snowballRU) IsStopword(token string) bool {
	_, ok := ruStopwords[strings.ReplaceAll(token, "ё", "е")]
	return ok
}

// Stem сводит слово к основе по алгоритму Snowball.
func (snowballRU) Stem(word string) string { return stemRU(word) }

// ruStopwords is the ranked Russian stopword list from the Snowball
// distribution (https://snowballstem.org/algorithms/russian/stop.txt), with ё
// transcribed to е exactly as the stemmer normalizes it.
var ruStopwords = map[string]struct{}{
	"и": {}, "в": {}, "во": {}, "не": {}, "что": {}, "он": {}, "на": {}, "я": {},
	"с": {}, "со": {}, "как": {}, "а": {}, "то": {}, "все": {}, "она": {},
	"так": {}, "его": {}, "но": {}, "да": {}, "ты": {}, "к": {}, "у": {},
	"же": {}, "вы": {}, "за": {}, "бы": {}, "по": {}, "только": {}, "ее": {},
	"мне": {}, "было": {}, "вот": {}, "от": {}, "меня": {}, "еще": {}, "нет": {},
	"о": {}, "из": {}, "ему": {}, "теперь": {}, "когда": {}, "даже": {}, "ну": {},
	"вдруг": {}, "ли": {}, "если": {}, "уже": {}, "или": {}, "ни": {}, "быть": {},
	"был": {}, "него": {}, "до": {}, "вас": {}, "нибудь": {}, "опять": {}, "уж": {},
	"вам": {}, "сказал": {}, "ведь": {}, "там": {}, "потом": {}, "себя": {},
	"ничего": {}, "ей": {}, "может": {}, "они": {}, "тут": {}, "где": {}, "есть": {},
	"надо": {}, "ней": {}, "для": {}, "мы": {}, "тебя": {}, "их": {}, "чем": {},
	"была": {}, "сам": {}, "чтоб": {}, "без": {}, "будто": {}, "человек": {},
	"чего": {}, "раз": {}, "тоже": {}, "себе": {}, "под": {}, "жизнь": {},
	"будет": {}, "ж": {}, "тогда": {}, "кто": {}, "этот": {}, "говорил": {},
	"того": {}, "потому": {}, "этого": {}, "какой": {}, "совсем": {}, "ним": {},
	"здесь": {}, "этом": {}, "один": {}, "почти": {}, "мой": {}, "тем": {},
	"чтобы": {}, "нее": {}, "кажется": {}, "сейчас": {}, "были": {}, "куда": {},
	"зачем": {}, "сказать": {}, "всех": {}, "никогда": {}, "сегодня": {},
	"можно": {}, "при": {}, "наконец": {}, "два": {}, "об": {}, "другой": {},
	"хоть": {}, "после": {}, "над": {}, "больше": {}, "тот": {}, "через": {},
	"эти": {}, "нас": {}, "про": {}, "всего": {}, "них": {}, "какая": {},
	"много": {}, "разве": {}, "сказала": {}, "три": {}, "эту": {}, "моя": {},
	"впрочем": {}, "хорошо": {}, "свою": {}, "этой": {}, "перед": {}, "иногда": {},
	"лучше": {}, "чуть": {}, "том": {}, "нельзя": {}, "такой": {}, "им": {},
	"более": {}, "всегда": {}, "конечно": {}, "всю": {}, "между": {},
}

// ruVowel reports whether r is a Russian vowel. ё is normalized to е before
// stemming so it never appears here.
func ruVowel(r rune) bool {
	switch r {
	case 'а', 'е', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я':
		return true
	}
	return false
}

func isAOrYa(r rune) bool { return r == 'а' || r == 'я' }

// ruRegions returns the RV and R2 boundaries of the word (start indices).
// RV begins after the first vowel; R2 after the first non-vowel following a
// vowel in R1, which itself begins after the first non-vowel following the
// first vowel.
func ruRegions(rw []rune) (pV, p2 int) {
	n := len(rw)
	pV = n
	for i, r := range rw {
		if ruVowel(r) {
			pV = i + 1
			break
		}
	}
	p1 := n
	for i := 0; i+1 < n; i++ {
		if ruVowel(rw[i]) && !ruVowel(rw[i+1]) {
			p1 = i + 2
			break
		}
	}
	p2 = n
	for i := p1; i+1 < n; i++ {
		if ruVowel(rw[i]) && !ruVowel(rw[i+1]) {
			p2 = i + 2
			break
		}
	}
	return pV, p2
}

// cutEnd returns rw without a trailing suffix, or ok=false if rw does not end
// with it.
func cutEnd(rw []rune, suf string) ([]rune, bool) {
	rs := []rune(suf)
	if len(rw) < len(rs) {
		return nil, false
	}
	for i, r := range rs {
		if rw[len(rw)-len(rs)+i] != r {
			return nil, false
		}
	}
	return rw[:len(rw)-len(rs)], true
}

func stemRU(word string) string {
	w := strings.ReplaceAll(word, "ё", "е")
	rw := []rune(w)
	n := len(rw)
	if n <= 1 {
		return w
	}

	pV, p2 := ruRegions(rw)

	// Step 1: remove a perfective gerund, or a reflexive ending followed by
	// the first matching adjectival, verbal or nominal ending.
	if rest, ok := perfectiveGerund(rw, pV); ok {
		rw = rest
	} else {
		if rest, ok := cutEnd(rw, "сь"); ok && len(rest) >= pV {
			rw = rest
		} else if rest, ok := cutEnd(rw, "ся"); ok && len(rest) >= pV {
			rw = rest
		}
		if rest, ok := adjectival(rw, pV); ok {
			rw = rest
		} else if rest, ok := russianVerb(rw, pV); ok {
			rw = rest
		} else if rest, ok := russianNoun(rw, pV); ok {
			rw = rest
		}
	}

	// Step 2: a trailing и is almost always an inflection marker, not part of
	// the stem.
	if len(rw) > 0 && rw[len(rw)-1] == 'и' && len(rw)-1 >= pV {
		rw = rw[:len(rw)-1]
	}

	// Step 3: derivational suffixes, but only in R2.
	if rest, ok := cutEnd(rw, "ость"); ok && len(rest) >= p2 {
		rw = rest
	} else if rest, ok := cutEnd(rw, "ост"); ok && len(rest) >= p2 {
		rw = rest
	}

	// Step 4: superlative, doubled-н and soft-sign tidy-up.
	rw = tidyUp(rw, pV)

	return string(rw)
}

type ruEnding struct {
	suf string
	g   int // 1 = must be preceded by а/я (in RV); 2 = unconditional
}

// perfectiveGerund removes a perfective-gerund ending. Group 1 endings (в,
// вши, вшись) must follow а or я; group 2 (ив..., ыв...) delete outright. The
// longest matching ending always wins.
func perfectiveGerund(rw []rune, pV int) ([]rune, bool) {
	ends := []ruEnding{
		{"ившись", 2}, {"ывшись", 2}, {"ивши", 2}, {"ывши", 2}, {"вшись", 1},
		{"ив", 2}, {"ыв", 2}, {"вши", 1}, {"в", 1},
	}
	for _, e := range ends {
		rest, ok := cutEnd(rw, e.suf)
		if !ok || len(rest) < pV {
			continue
		}
		if e.g == 1 {
			if len(rest) == 0 || !isAOrYa(rest[len(rest)-1]) || len(rest)-1 < pV {
				continue
			}
		}
		return rest, true
	}
	return rw, false
}

// adjectival removes an adjective ending optionally followed (inward) by a
// participle ending. Both must lie in RV; a group-1 participle first requires
// a preceding а/я (which stays in the stem), group 2 deletes outright. If no
// participle follows, the adjective removal is undone and the word stays.
func adjectival(rw []rune, pV int) ([]rune, bool) {
	adjectiveEnds := []string{
		"ими", "ыми", "его", "ого", "ему", "ому",
		"их", "ых", "ую", "юю", "ая", "яя", "ою", "ею",
		"ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом",
	}
	for _, a := range adjectiveEnds {
		rest, ok := cutEnd(rw, a)
		if !ok || len(rest) < pV {
			continue
		}
		if pres, ok := participle(rest, pV); ok {
			return pres, true
		}
		return rest, true
	}
	return rw, false
}

func participle(rw []rune, pV int) ([]rune, bool) {
	for _, e := range []ruEnding{
		{"ивш", 2}, {"ывш", 2}, {"ующ", 2},
		{"ем", 1}, {"нн", 1}, {"вш", 1}, {"ющ", 1}, {"щ", 1},
	} {
		rest, ok := cutEnd(rw, e.suf)
		if !ok || len(rest) < pV {
			continue
		}
		if e.g == 1 {
			if len(rest) == 0 || !isAOrYa(rest[len(rest)-1]) || len(rest)-1 < pV {
				continue
			}
		}
		return rest, true
	}
	return rw, false
}

// russianVerb removes a verbal ending. Group 1 (ла, на, ете, ...) must follow
// а/я; group 2 inflected forms delete outright.
func russianVerb(rw []rune, pV int) ([]rune, bool) {
	ends := []ruEnding{
		{"уйте", 2}, {"ейте", 2}, {"ила", 2}, {"ыла", 2}, {"ена", 2},
		{"ите", 2}, {"или", 2}, {"ыли", 2}, {"ило", 2}, {"ыло", 2}, {"ено", 2},
		{"ует", 2}, {"уют", 2}, {"ить", 2}, {"ыть", 2}, {"ишь", 2}, {"ены", 2},
		{"ете", 1}, {"йте", 1}, {"ешь", 1}, {"нно", 1},
		{"ла", 1}, {"на", 1}, {"ли", 1}, {"ем", 1}, {"ло", 1}, {"но", 1},
		{"ет", 1}, {"ют", 1}, {"ны", 1}, {"ть", 1},
		{"ей", 2}, {"уй", 2}, {"ил", 2}, {"ыл", 2}, {"им", 2}, {"ым", 2},
		{"ен", 2}, {"ят", 2}, {"ит", 2}, {"ыт", 2}, {"ую", 2},
		{"й", 1}, {"л", 1}, {"н", 1},
		{"ю", 2},
	}
	return ruCutEndings(rw, pV, ends)
}

// russianNoun removes a nominal ending, longest match first.
func russianNoun(rw []rune, pV int) ([]rune, bool) {
	ends := []ruEnding{
		{"иями", 2},
		{"ией", 2}, {"иям", 2}, {"иях", 2}, {"ием", 2}, {"ями", 2}, {"ами", 2},
		{"ев", 2}, {"ов", 2}, {"ие", 2}, {"ье", 2}, {"еи", 2}, {"ии", 2},
		{"ей", 2}, {"ой", 2}, {"ий", 2}, {"ям", 2}, {"ем", 2}, {"ам", 2},
		{"ом", 2}, {"ах", 2}, {"ях", 2}, {"ию", 2}, {"ью", 2}, {"ия", 2}, {"ья", 2},
		{"а", 2}, {"е", 2}, {"и", 2}, {"о", 2}, {"у", 2}, {"ы", 2},
		{"ь", 2}, {"й", 2}, {"ю", 2}, {"я", 2},
	}
	return ruCutEndings(rw, pV, ends)
}

// ruCutEndings removes the longest ending in the list that fits inside RV.
func ruCutEndings(rw []rune, pV int, ends []ruEnding) ([]rune, bool) {
	for _, e := range ends {
		rest, ok := cutEnd(rw, e.suf)
		if !ok {
			continue
		}
		if e.g == 1 {
			if len(rest) == 0 || !isAOrYa(rest[len(rest)-1]) || len(rest)-1 < pV {
				continue
			}
		} else if len(rest) < pV {
			continue
		}
		return rest, true
	}
	return rw, false
}

// tidyUp applies step 4: strip a superlative ending (and then collapse a
// doubled н the way undouble lowers -нн-), or collapse -нн-, or drop a soft
// sign.
func tidyUp(rw []rune, pV int) []rune {
	if len(rw) == 0 {
		return rw
	}
	for _, e := range []string{"ейше", "ейш"} {
		if rest, ok := cutEnd(rw, e); ok && len(rest) >= pV {
			rw = rest
			return undoubleN(rw)
		}
	}
	if rw[len(rw)-1] == 'н' && len(rw) >= 2 && rw[len(rw)-2] == 'н' {
		return rw[:len(rw)-1]
	}
	if rw[len(rw)-1] == 'ь' && len(rw)-1 >= pV {
		return rw[:len(rw)-1]
	}
	return rw
}

func undoubleN(rw []rune) []rune {
	if len(rw) >= 2 && rw[len(rw)-1] == 'н' && rw[len(rw)-2] == 'н' {
		return rw[:len(rw)-1]
	}
	return rw
}
