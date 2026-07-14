package lexical

import "testing"

// TestTokenizeFuncMatchesTokenize: the streaming tokenizer must emit exactly the tokens
// the slice one emits. A tokenizer that differs even slightly changes every posting list
// and therefore every ranking, and nothing would say so.
func TestTokenizeFuncMatchesTokenize(t *testing.T) {
	cases := []string{
		"",
		"hello",
		"The quick brown fox, jumps over 42 lazy dogs!",
		"  leading and trailing   ",
		"UPPER Case MiXeD",
		"hyphen-separated words and under_scores",
		"unicode: naïve café résumé 東京 tokyo",
		"a an the of and to is it",
		"trailing token at the very end xyz",
		"symbols!!! ???  ---  ...",
		"digits 123 4567 mixed42tokens",
	}
	for _, c := range cases {
		want := tokenize(c)
		var got []string
		tokenizeFunc(c, func(tok string) { got = append(got, tok) })
		if len(want) != len(got) {
			t.Fatalf("%q: got %d tokens %v, want %d %v", c, len(got), got, len(want), want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%q: token %d is %q, want %q", c, i, got[i], want[i])
			}
		}
	}
}
