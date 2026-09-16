package rag

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// CompileMetaFilter builds a document-metadata predicate from "key=value"
// pairs: a document matches when every pair's key is present in its metadata
// and the scalar value is textually equal to the pair's value (strings compare
// as-is, numbers and booleans by their JSON text; arrays and objects never
// match). nil or empty input yields a nil predicate (no filtering). Missing
// metadata never matches, so a filter never leaks chunks whose document failed
// to attach the requested fields.
func CompileMetaFilter(pairs []string) (func(json.RawMessage) bool, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	preds := make([]struct{ k, v string }, 0, len(pairs))
	for _, p := range pairs {
		k, v, ok := strings.Cut(p, "=")
		if !ok || strings.TrimSpace(k) == "" {
			return nil, fmt.Errorf("meta filter %q: want key=value", p)
		}
		preds = append(preds, struct{ k, v string }{strings.TrimSpace(k), v})
	}
	// A document's metadata is shared by all its chunks, so memoize the verdict
	// by the raw bytes: one unmarshal per distinct document per query, not per
	// candidate chunk.
	cache := make(map[string]bool)
	return func(meta json.RawMessage) bool {
		if len(meta) == 0 {
			return false
		}
		key := string(meta)
		if v, ok := cache[key]; ok {
			return v
		}
		ok := metaMatches(meta, preds)
		cache[key] = ok
		return ok
	}, nil
}

// metaMatches reports whether the raw JSON metadata satisfies every predicate.
func metaMatches(meta json.RawMessage, preds []struct{ k, v string }) bool {
	dec := json.NewDecoder(bytes.NewReader(meta))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return false
	}
	for _, p := range preds {
		text, ok := scalarText(m[p.k])
		if !ok || text != p.v {
			return false
		}
	}
	return true
}

// scalarText renders a JSON scalar as its canonical text; non-scalars report
// false so they never match a string value.
func scalarText(v any) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case json.Number:
		return t.String(), true
	case bool:
		if t {
			return "true", true
		}
		return "false", true
	}
	return "", false
}
