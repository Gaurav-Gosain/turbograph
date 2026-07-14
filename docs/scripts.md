# Transform scripts

turbograph can run your own programs over each document at ingest, before it is
chunked. A script is any executable file: a compiled Go binary, a Python file, a
shell script, anything the host can run. It reads one JSON document on stdin and
writes one back on stdout, a contract every language implements in a few lines.

Use them to strip navigation and boilerplate out of scraped pages, redact secrets,
normalize whitespace, lift front-matter into metadata, classify or tag documents,
convert a format turbograph does not know, or drop junk before it ever reaches the
index.

## The trust boundary (read this first)

**Scripts are registered by the operator, not by the caller.** You point turbograph
at a directory when you start it:

```
turbograph serve --data ./data --scripts ./scripts
```

Every executable in that directory becomes available *by name*. An ingest request
may then ask for those names, and only those names:

```json
{ "documents": [ { "id": "page.md", "text": "..." } ],
  "transform": ["strip-nav.py", "drop-empty.sh"] }
```

A caller can never supply a path, an argument vector, or a shell string. An unknown
name is rejected with 400 before anything is executed.

This is deliberate. turbograph is routinely run as a server, and an endpoint that
executed a caller-supplied command would hand remote code execution to anyone who
could reach it. So:

> **Enabling `--scripts` makes every program in that directory runnable by anyone
> who can reach the API.** Put only code you trust there, and protect the API
> (`--api-key`) as you would a shell. With `--scripts` unset the feature is
> entirely off and there is nothing to attack.

Programs are executed directly, never through a shell, so there is no command
injection through arguments or filenames. Each run is bounded by
`--script-timeout` (30s by default) and is killed if it overruns.

## The contract

turbograph writes one JSON object to stdin:

```json
{ "id": "page.md", "text": "the document text", "meta": { "author": "ada" } }
```

Your script writes one JSON object to stdout:

| field  | meaning                                                                      |
| ------ | ---------------------------------------------------------------------------- |
| `text` | the document's new text. Required unless you drop it                         |
| `meta` | replaces the document's metadata. Omit it to leave the existing metadata alone |
| `drop` | `true` to skip this document entirely. Nothing else is needed                |

Rules:

- **Exit non-zero to fail.** Whatever you print on stderr is surfaced with the
  error, so write there when something goes wrong.
- **A failure is isolated to one document.** The rest of the ingest continues, and
  the failed document is reported in the response's `failed` list (and marked in
  the web UI), exactly like a PDF that will not parse.
- **Returning empty text is an error**, because it is almost always a bug. If you
  mean to discard the document, say `{"drop": true}`.
- **Scripts compose.** `"transform": ["a", "b"]` pipes each document through `a`,
  then feeds `a`'s output into `b`.

## Examples

**Python** — strip navigation and footer lines, and tag what you cleaned:

```python
#!/usr/bin/env python3
import sys, json
d = json.load(sys.stdin)
lines = [l for l in d["text"].splitlines()
         if not l.strip().lower().startswith(("nav:", "footer:"))]
print(json.dumps({"text": "\n".join(lines), "meta": {"cleaned": True}}))
```

**Go** — drop anything too short to be worth indexing:

```go
package main

import (
	"encoding/json"
	"os"
	"strings"
)

type doc struct {
	ID   string          `json:"id"`
	Text string          `json:"text"`
	Meta json.RawMessage `json:"meta,omitempty"`
	Drop bool            `json:"drop,omitempty"`
}

func main() {
	var d doc
	json.NewDecoder(os.Stdin).Decode(&d)
	if len(strings.Fields(d.Text)) < 20 {
		json.NewEncoder(os.Stdout).Encode(doc{Drop: true})
		return
	}
	json.NewEncoder(os.Stdout).Encode(d)
}
```

Build it into the scripts directory (`go build -o scripts/drop-short ./cmd/dropshort`)
and it is available as `drop-short`.

**Shell + jq** — redact anything that looks like a key:

```sh
#!/bin/sh
jq -c '{text: (.text | gsub("sk-[A-Za-z0-9]{16,}"; "[redacted]"))}'
```

## Using them

From the web UI, tick the scripts you want under **transform scripts** in the
documents panel before you drop files in. From the API, name them in `transform`
on `POST /api/ingest` or `POST /api/ingest/files`; extracted PDF text runs through
the same stage, so a script that cleans up PDF output is written once and works for
both paths.

The response reports what happened:

```json
{ "chunks": 41,
  "dropped": ["stub.md"],
  "failed":  [ { "id": "broken.md", "error": "script tag.py failed: exit status 1: KeyError: 'text'" } ] }
```

## Where this runs in the pipeline

```
walk -> extract (PDF/OCR) -> TRANSFORM (your scripts) -> chunk -> embed -> index -> graph
```

Transforms run at ingest only, so they add no latency to queries. They see the
document after text extraction and before chunking, which is the point at which
changing the text still changes everything downstream: the chunks, the embeddings,
the lexical index, and the graph.
