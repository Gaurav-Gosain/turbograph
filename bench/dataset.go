package bench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// Dataset is a loaded benchmark: the corpus to ingest and the labeled queries to
// score against, with relevance keyed by document id.
type Dataset struct {
	Name  string
	Docs  []rag.Document
	Cases []eval.Case
}

// LoadBEIR loads a dataset in the BEIR layout: corpus.jsonl ({"_id","title",
// "text"}), queries.jsonl ({"_id","text"}), and a qrels TSV
// (query-id<TAB>corpus-id<TAB>score, score >= 1 meaning relevant, with an
// optional header row). Relevance is at the document level, the BEIR convention.
func LoadBEIR(corpusPath, queriesPath, qrelsPath string) (*Dataset, error) {
	docs, err := loadCorpus(corpusPath)
	if err != nil {
		return nil, err
	}
	queries, err := loadQueries(queriesPath)
	if err != nil {
		return nil, err
	}
	rel, err := loadQrels(qrelsPath)
	if err != nil {
		return nil, err
	}
	cases := make([]eval.Case, 0, len(rel))
	for qid, docIDs := range rel {
		q, ok := queries[qid]
		if !ok || len(docIDs) == 0 {
			continue
		}
		cases = append(cases, eval.Case{Query: q, Relevant: docIDs})
	}
	return &Dataset{Name: "beir", Docs: docs, Cases: cases}, nil
}

func loadCorpus(path string) ([]rag.Document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var docs []rag.Document
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			ID    string `json:"_id"`
			Title string `json:"title"`
			Text  string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("bench: corpus parse: %w", err)
		}
		text := rec.Text
		if rec.Title != "" {
			text = rec.Title + "\n" + rec.Text
		}
		docs = append(docs, rag.Document{ID: rec.ID, Text: text})
	}
	return docs, sc.Err()
}

func loadQueries(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			ID   string `json:"_id"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("bench: queries parse: %w", err)
		}
		out[rec.ID] = rec.Text
	}
	return out, sc.Err()
}

func loadQrels(path string) (map[string][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string][]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	first := true
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		cols := strings.Fields(line)
		if len(cols) < 3 {
			continue
		}
		// Skip a header row such as "query-id corpus-id score".
		if first {
			first = false
			if cols[2] == "score" || strings.Contains(cols[0], "query") {
				continue
			}
		}
		if cols[2] == "0" {
			continue
		}
		out[cols[0]] = append(out[cols[0]], cols[1])
	}
	return out, sc.Err()
}

// LoadSuiteFile loads a turbograph eval suite (JSONL of {query, relevant}) and the
// corpus separately, for chunk-level or custom datasets.
func LoadSuiteFile(path string) ([]eval.Case, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return eval.LoadSuite(f)
}
