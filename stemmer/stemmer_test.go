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
	withRegistry(t, upperCyrillic{})
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
func withRegistry(t *testing.T, extra ...Stemmer) {
	t.Helper()
	old := append([]Stemmer(nil), registered...)
	registered = nil
	for _, s := range extra {
		Register(s)
	}
	t.Cleanup(func() { registered = old })
}
