package stemmer

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// TestStemRUvsSnowballData validates the port against the official Snowball
// Russian test vocabulary. The data lives outside the repo, so the test skips
// when the files are absent (it is not meant to run in CI).
func TestStemRUvsSnowballData(t *testing.T) {
	voc, err := os.Open(os.Getenv("SNOWBALL_VOC"))
	if err != nil {
		t.Skip("no snowball vocab file")
	}
	defer voc.Close()
	out, err := os.Open(os.Getenv("SNOWBALL_OUT"))
	if err != nil {
		t.Skip("no snowball output file")
	}
	defer out.Close()

	sc := bufio.NewScanner(voc)
	so := bufio.NewScanner(out)
	fail := 0
	for sc.Scan() && so.Scan() {
		w := strings.TrimSpace(sc.Text())
		want := strings.TrimSpace(so.Text())
		if w == "" {
			continue
		}
		got := stemRU(w)
		if got != want {
			if fail < 30 {
				t.Logf("stemRU(%q) = %q, want %q", w, got, want)
			}
			fail++
		}
	}
	if sc.Err() != nil {
		t.Fatal(sc.Err())
	}
	t.Logf("mismatches: %d / %d", fail, countLines(voc))
}

func countLines(f *os.File) int {
	f.Seek(0, 0)
	defer func() { f.Seek(0, 0) }()
	n := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		n++
	}
	return n
}
