package lexical

import (
	"reflect"
	"testing"

	"github.com/Gaurav-Gosain/turbograph/stemmer"
)

func TestTokenizeStemming(t *testing.T) {
	enAll, err := stemmer.Select(nil)
	if err != nil {
		t.Fatal(err)
	}
	enOnly, err := stemmer.Select([]string{"en"})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		stem func(string) string
		want []string
	}{
		// Exact/code tokens are kept as-is: digits and all-caps acronyms are
		// identifiers, not inflected words.
		{"RUNNING runs running run2", enAll, []string{"running", "run", "run", "run2"}},
		// Russian is stemmed and its stopwords are dropped.
		{"этот в машины", enAll, []string{"машин"}},
		// Russian stopwords stay when stemming is off: default unchanged.
		{"этот в машины", nil, []string{"этот", "в", "машины"}},
		// A mixed-script token with a digit is exact, so it never stems.
		{"Кафка2024 машины", enAll, []string{"кафка2024", "машин"}},
		// English-only selection stems English but leaves Russian untouched.
		{"running машины", enOnly, []string{"run", "машины"}},
	}
	for _, c := range cases {
		got := tokenize(c.text, c.stem)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("tokenize(%q, stem=%v) = %v, want %v", c.text, c.stem != nil, got, c.want)
		}
	}
}
