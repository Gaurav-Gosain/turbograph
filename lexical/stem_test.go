package lexical

import (
	"reflect"
	"testing"
)

func TestTokenizeStemming(t *testing.T) {
	cases := []struct {
		text string
		stem bool
		want []string
	}{
		// Exact/code tokens are kept as-is: digits and all-caps acronyms are
		// identifiers, not inflected words.
		{"RUNNING runs running run2", true, []string{"running", "run", "run", "run2"}},
		// Russian is stemmed and its stopwords are dropped.
		{"этот в машины", true, []string{"машин"}},
		// Russian stopwords stay when stemming is off: default unchanged.
		{"этот в машины", false, []string{"этот", "в", "машины"}},
		// A mixed-script token with a digit is exact, so it never stems.
		{"Кафка2024 машины", true, []string{"кафка2024", "машин"}},
	}
	for _, c := range cases {
		got := tokenize(c.text, c.stem)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("tokenize(%q, stem=%v) = %v, want %v", c.text, c.stem, got, c.want)
		}
	}
}
