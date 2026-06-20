# turbograph Python client

A clean, dependency-free Python client for [turbograph](https://github.com/Gaurav-Gosain/turbograph),
a fast graph-RAG server. It uses only the Python standard library (urllib, json,
base64, dataclasses, typing) and runs on Python 3.9+, matching turbograph's
zero-dependency ethos.

## Install

The client is a single small package with no runtime dependencies. Pick whichever
fits your workflow:

- Copy the `turbograph/` directory into your project and import it.
- Install it as an editable package from this directory:

  ```sh
  pip install -e .
  ```

There is nothing to compile and no third-party package to fetch.

## Quick start

```python
from turbograph import Client

tg = Client("http://localhost:8080", bucket="default")

# Ingest plain text (single-document convenience form).
tg.ingest_text("doc1", "The quick brown fox jumps over the lazy dog.")

# Or ingest several documents at once, with optional metadata.
tg.ingest_text([
    {"id": "a", "text": "...", "meta": {"title": "A"}},
    {"id": "b", "text": "..."},
])

# Retrieve.
for hit in tg.query("what does the fox do?", top_k=5):
    print(hit.score, hit.doc_id, hit.text)
```

## Buckets

A bucket is an isolated corpus. The client appends a `bucket` query parameter to
every bucket-scoped call. Set the default at construction, and override per call
with `bucket=...`:

```python
tg = Client("http://localhost:8080", bucket="legal")
tg.query("indemnification clause")                # uses "legal"
tg.query("dosage guidance", bucket="medical")     # uses "medical" for this call

tg.create_bucket("research")
tg.buckets()
tg.delete_bucket("research")
```

## Streaming chat

`chat` is a generator over the server's server-sent-events stream. It yields typed
`SSEEvent` objects: `sources`, then `token` (repeated), then `abstain`, `error`,
or `done`.

```python
for event in tg.chat("summarize the fox document"):
    if event.event == "sources":
        print("sources:", [s["doc_id"] for s in event.data["sources"]])
    elif event.event == "token":
        print(event.data["text"], end="", flush=True)
    elif event.event == "abstain":
        print("abstained:", event.data["message"])
    elif event.event == "error":
        print("error:", event.data["error"])
```

For the common case, `chat_text` consumes the whole stream and returns the answer
string plus the parsed sources:

```python
answer, sources = tg.chat_text("what does the fox do?", top_k=3)
print(answer)
print([s.doc_id for s in sources])
```

Chat parameters: `top_k`, `graph_mix`, `mmr_lambda`, `entity_mix`, `min_sim`,
`rerank`, `global_` (answer corpus-wide from community summaries), `meta_keys`,
`history` (prior turns for query rewriting), and `model`.

## Files and images

```python
# Ingest a file from disk; the extension decides text vs. server-side extraction.
tg.ingest_path("notes.md")
tg.ingest_path("report.pdf", meta={"team": "research"})

# Ingest pre-encoded binary files directly.
tg.ingest_files([{"id": "r1", "b64": "<base64>", "meta": {"k": "v"}}])

# Ingest an image: it is captioned by a vision model and indexed as text.
with open("chart.png", "rb") as fh:
    tg.ingest_image("chart1", fh.read(), ext="png", model="llava")

# Fetch a stored image referenced by a retrieved image chunk.
for hit in tg.query("the revenue chart"):
    if hit.kind == "image":
        print(tg.asset_url(hit.image_ref))
        data = tg.get_asset(hit.image_ref)
```

## Documents and versions

```python
tg.documents()                  # list documents in the bucket
view = tg.document("doc1")      # full text, metadata, chunk spans
tg.delete_document("doc1")

tg.versions("doc1")             # version history, oldest first
tg.version_text("doc1", 1)      # text of version 1
tg.restore("doc1", 1)           # restore version 1 (git-revert semantics)
```

## Graph builds (entities and communities)

These are expensive, on-demand indexing passes that stream progress. Use the
streaming form to show progress, or the blocking form to wait for completion.

```python
# Streaming.
for event in tg.build_entities(model="llama3", batch=4):
    if event.event == "progress":
        print(event.data)

# Blocking; returns the final "done" payload.
tg.build_communities_blocking(model="llama3")
tg.communities()               # the generated community summaries
```

After building communities, `chat(..., global_=True)` answers broad, corpus-wide
questions from the community summaries.

## Server, models, and persistence

```python
tg.models()    # available models, default, capabilities
tg.status()    # version, backend readiness, bucket stats
tg.save()      # persist the bucket to disk
```

## Error handling

Any non-2xx response raises `TurbographError`, carrying the server's
`{"error": "..."}` message, the HTTP `status`, and the raw `body`:

```python
from turbograph import TurbographError

try:
    tg.query("")
except TurbographError as e:
    print(e.message, e.status)
```

## Example

`example.py` is a runnable tour. Start a turbograph server, then:

```sh
python example.py
```

## Tests

The test suite needs no live server (it unit-tests the SSE parser and mocks
`urlopen` for URL and body assertions):

```sh
python -m pytest -q
# or, without pytest installed:
python -m unittest discover -s tests
```
