package lexical

import "testing"

// TestSearchMatchesInflections verifies that with Stem enabled a query in one
// grammatical form retrieves documents written in another, and that the default
// (Stem off) keeps the classic exact-token behavior.
func TestSearchMatchesInflections(t *testing.T) {
	docs := []struct{ id, text string }{
		{"en", "the daily runs train endurance"},
		{"ru", "машины и автомобили едут по дороге"},
	}

	on := New(Config{Stem: true})
	for _, d := range docs {
		on.Add(d.id, d.text)
	}
	on.Finalize()

	if res := on.Search("running", 10); len(res) == 0 || res[0].ID != "en" {
		t.Errorf("stemmed EN search missed the inflected form: %+v", res)
	}
	if res := on.Search("машина", 10); len(res) == 0 || res[0].ID != "ru" {
		t.Errorf("stemmed RU search missed the inflected form: %+v", res)
	}

	off := New(Config{})
	for _, d := range docs {
		off.Add(d.id, d.text)
	}
	off.Finalize()

	if res := off.Search("running", 10); len(res) != 0 {
		t.Errorf("default (no stem) must not match the inflected EN form, got %+v", res)
	}
	if res := off.Search("машина", 10); len(res) != 0 {
		t.Errorf("default (no stem) must not match the inflected RU form, got %+v", res)
	}
}

// TestStemmedAddBatchMatchesAdd keeps the streaming batch path and the
// sequential path tokenizing identically when stemming is on.
func TestStemmedAddBatchMatchesAdd(t *testing.T) {
	ids := []string{"a", "b"}
	texts := []string{"running and runners", "машины едут"}
	seq := New(Config{Stem: true})
	for i := range ids {
		seq.Add(ids[i], texts[i])
	}
	par := New(Config{Stem: true})
	par.AddBatch(ids, texts)

	for _, q := range []string{"running", "run", "машина"} {
		s, p := seq.Search(q, 10), par.Search(q, 10)
		if len(s) != len(p) {
			t.Fatalf("query %q: seq %d hits, batch %d hits", q, len(s), len(p))
		}
		for i := range s {
			if s[i] != p[i] {
				t.Fatalf("query %q: seq %v vs batch %v", q, s, p)
			}
		}
	}
}
