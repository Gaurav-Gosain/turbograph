package stemmer

import "testing"

func TestStemRussian(t *testing.T) {
	cases := []struct{ word, want string }{
		{"машина", "машин"},
		{"машины", "машин"},
		{"машине", "машин"},
		{"машин", "машин"},
		{"работала", "работа"},
		{"работать", "работа"},
		{"красота", "красот"},
		{"прочитал", "прочита"},
		{"читает", "чита"},
		{"бегают", "бега"},
		{"важнейшие", "важн"},
	}
	for _, c := range cases {
		if got := stemRU(c.word); got != c.want {
			t.Errorf("stemRU(%q) = %q, want %q", c.word, got, c.want)
		}
	}
}

func TestStemNormalizesYo(t *testing.T) {
	// The reference stemmer maps ё to е, so no ё may survive; the adjective
	// жёлтый stems to желт (adjective endings delete outright, no а/я guard).
	if got := stemRU("жёлтый"); got != "желт" {
		t.Errorf("stemRU(жёлтый) = %q, want %q", got, "желт")
	}
}

func TestRussianStopwordNormalizesYo(t *testing.T) {
	// Ещё is on the stoplist in its е spelling; the ё form must match too.
	ru := snowballRU{}
	if !ru.IsStopword("еще") {
		t.Error("еще must be a stopword")
	}
	if !ru.IsStopword("ещё") {
		t.Error("ещё must be a stopword (ё normalized to е)")
	}
}
