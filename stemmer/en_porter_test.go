package stemmer

import "testing"

func TestPorterEnglish(t *testing.T) {
	cases := []struct{ word, want string }{
		{"run", "run"},
		{"runs", "run"},
		{"running", "run"},
		{"jumped", "jump"},
		{"runner", "runner"},
		{"happy", "happi"},
		{"happiness", "happi"},
	}
	for _, c := range cases {
		if got := stemEN(c.word); got != c.want {
			t.Errorf("stemEN(%q) = %q, want %q", c.word, got, c.want)
		}
	}
}

func TestPorterEnglishPassThrough(t *testing.T) {
	// Non-word tokens are filtered out by Claims before Stem is ever called;
	// the guard inside stemEN is a second line of defence that must behave the
	// same for whatever survives.
	cases := []struct{ word, want string }{
		{"café", "café"},
		{"run2", "run2"},
		{"a", "a"},
	}
	for _, c := range cases {
		if got := stemEN(c.word); got != c.want {
			t.Errorf("stemEN(%q) = %q, want %q", c.word, got, c.want)
		}
	}
}
