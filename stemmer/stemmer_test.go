package stemmer

import (
	"strings"
	"testing"
)

func TestStemTokenResolvesByWritingSystem(t *testing.T) {
	cases := []struct {
		token string
		want  string
	}{
		{"running", "run"},  // English Porter applies.
		{"машины", "машин"}, // Russian Snowball applies.
		{"и", ""},           // Russian stopword drops the token.
		{"run2", "run2"},    // Not a word of either script: passthrough.
		{"café", "café"},    // Non-ASCII, non-Cyrillic: passthrough.
		{"жёлтый", "желт"},  // ё normalized on the Russian path.
	}
	for _, c := range cases {
		t.Run(c.token, func(t *testing.T) {
			if got := StemToken(c.token); got != c.want {
				t.Errorf("StemToken(%q) = %q, want %q", c.token, got, c.want)
			}
		})
	}
}

func TestEmptyRegistryPassesTokensThrough(t *testing.T) {
	// A registry with no matching language must be inert: removing a language
	// (its file/init gone) silently disables stemming for that script.
	withRegistry(t)
	if got := StemToken("машины"); got != "машины" {
		t.Errorf("StemToken with empty registry = %q, want %q", got, "машины")
	}
	if got := StemToken("running"); got != "running" {
		t.Errorf("StemToken with empty registry = %q, want %q", got, "running")
	}
}

func TestFirstRegisteredStemmerClaimsWin(t *testing.T) {
	// Регистрация до русской перехватывает кириллицу: порядок в реестре —
	// это и есть приоритет при пересечении Claims.
	withRegistry(t, namedStemmer{name: "shout", s: upperCyrillic{}})
	if got := StemToken("машины"); got != "МАШИНЫ" {
		t.Errorf("StemToken = %q, want %q", got, "МАШИНЫ")
	}
}

// upperCyrillic claims any Cyrillic token and uppercases it — a stand-in for a
// hypothetical third stemmer that would otherwise collide with Russian.
type upperCyrillic struct{}

func (upperCyrillic) Claims(token string) bool { return HasCyrillic(token) }
func (upperCyrillic) IsStopword(string) bool   { return false }
func (upperCyrillic) Stem(token string) string { return strings.ToUpper(token) }

// withRegistry подменяет реестр на время теста и восстанавливает прежний
// набор (включая регистрации из init) по завершении.
func withRegistry(t *testing.T, extra ...namedStemmer) {
	t.Helper()
	old := append([]namedStemmer(nil), registered...)
	registered = nil
	for _, e := range extra {
		Register(e.name, e.s)
	}
	t.Cleanup(func() { registered = old })
}

func TestNamesListsRegisteredLanguages(t *testing.T) {
	names := Names()
	has := func(want string) bool {
		for _, n := range names {
			if n == want {
				return true
			}
		}
		return false
	}
	if !has("en") || !has("ru") {
		t.Errorf("Names() = %v, want the built-in en and ru", names)
	}
}

func TestSelectEmptyMeansAllLanguages(t *testing.T) {
	f, err := Select(nil)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]string{"running": "run", "машины": "машин", "и": ""}
	for tok, want := range cases {
		if got := f(tok); got != want {
			t.Errorf("Select(nil)(%q) = %q, want %q", tok, got, want)
		}
	}
}

func TestSelectRestrictsToNamedLanguages(t *testing.T) {
	f, err := Select([]string{"en"})
	if err != nil {
		t.Fatal(err)
	}
	if got := f("running"); got != "run" {
		t.Errorf("english token not stemmed: %q", got)
	}
	// Russian is not selected: its tokens (and stopwords) pass through untouched.
	for _, tok := range []string{"машины", "и"} {
		if got := f(tok); got != tok {
			t.Errorf("non-selected language token %q changed to %q", tok, got)
		}
	}
	// Case and whitespace in the config are not allowed to break selection.
	f2, err := Select([]string{" EN", "ru "})
	if err != nil {
		t.Fatal(err)
	}
	if got := f2("running"); got != "run" {
		t.Errorf("upper-cased language name not honored: %q", got)
	}
	if got := f2("машины"); got != "машин" {
		t.Errorf("ru not selected: %q", got)
	}
}

func TestSelectRejectsUnknownLanguage(t *testing.T) {
	if _, err := Select([]string{"en", "zz"}); err == nil {
		t.Fatal("expected an error for an unknown language")
	}
}
