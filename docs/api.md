# HTTP API reference

turbograph serves a JSON HTTP API, a server-sent-events (SSE) stream for chat and
long-running builds, an OpenAI-compatible endpoint, and the embedded web UI. This
document is the complete surface. Official clients wrap it for
[Python](../clients/python/) and [TypeScript](../clients/typescript/).

A machine-readable OpenAPI 3 description is served at `GET /openapi.json` (also
`/api/openapi.json`) and shipped at
[server/static/openapi.json](../server/static/openapi.json), so Swagger UI, code
generators, and Postman can consume the API directly. For example, point Swagger
UI at `http://localhost:8080/openapi.json`, or generate a client with
`openapi-generator`.

## Conventions

- Base URL defaults to `http://localhost:8080` (`turbograph serve --addr`).
- Every store-scoped endpoint accepts a `bucket` query parameter and defaults to
  `default`. A bucket is one isolated corpus (one `.tg` file).
- Request and response bodies are JSON unless noted. Errors return a non-2xx
  status with `{"error": "message"}`.
- Endpoints are under `/api`, and `/v1` is the OpenAI-compatible surface. Three
  core endpoints also keep short aliases for existing clients: `POST /api/ingest`
  (alias `/ingest`), `POST /api/query` (alias `/query`), and `GET /api/stats`
  (alias `/stats`). The `/api/*` form is canonical.
- Streaming endpoints emit SSE: lines of `event: <name>` and `data: <json>`,
  separated by a blank line. Chat streams `sources`, then `token` repeatedly,
  then `done`; it may emit `abstain` or `error` instead. Build endpoints stream
  `progress`, then `done` or `error`.

## Ingestion

### `POST /api/ingest`  (alias `POST /ingest`)
Add or replace text documents.
```json
{ "documents": [ { "id": "notes.md", "text": "...", "meta": { "author": "ada" } } ],
  "replace": false }
```
`replace: true` rebuilds the bucket from scratch; otherwise documents are added
incrementally (changed content updates in place). Returns `{ "chunks": <int> }`.

### `POST /api/ingest/files`
Ingest binary or text files; the server extracts text (PDF, OCR, plain text).
```json
{ "files": [ { "id": "report.pdf", "b64": "<base64>", "meta": { "page": 12 } } ] }
```
Returns `{ "chunks", "indexed", "failed", "saved" }`.

### `POST /api/ingest/image`
Caption an image with a vision model and index the caption (describe then embed).
Requires a vision-capable backend and a configured asset directory.
```json
{ "id": "fig3.png", "b64": "<base64>", "ext": "png",
  "model": "qwen2.5-vl", "prompt": "Describe this figure.", "meta": { } }
```
Returns `{ "id", "image_ref", "caption" }`.

## Retrieval and chat

### `POST /api/query`  (alias `POST /query`)
Retrieve chunks without generating an answer (no model required).
```json
{ "query": "how does HNSW search work", "top_k": 6,
  "graph_mix": 0, "mmr_lambda": 0, "entity_mix": 0 }
```
Returns `{ "results": [ QueryResult ] }`, where each `QueryResult` is:
```json
{ "id": "doc#3", "doc_id": "doc", "score": 1.25, "similarity": 0.82,
  "text": "...", "start": 120, "end": 540,
  "meta": { }, "kind": "image", "image_ref": "ab12cd34.png" }
```
`start`/`end` are rune offsets of the chunk in its document; `kind`/`image_ref`
are present only for image chunks.

### `POST /api/chat`  (SSE)
Retrieve and stream a grounded answer.
```json
{ "query": "...", "model": "qwen3.5:2b", "top_k": 6, "graph_mix": 0,
  "mmr_lambda": 0, "entity_mix": 0, "min_sim": 0, "rerank": false,
  "history": [ { "role": "user", "content": "..." } ],
  "meta_keys": ["author"], "global": false }
```
- `global: true` answers a corpus-wide question from the community summaries
  rather than individual passages (build them first; see below).
- `meta_keys` injects those document-metadata fields into each passage given to
  the model.
- `min_sim` sets the abstention threshold; `rerank` enables pointwise LLM
  reranking.

Events: `sources` (a `{ "sources": [QueryResult] }` payload), then `token`
(`{ "text": "..." }`) repeatedly, then `done`. May emit `abstain`
(`{ "message": "..." }`) or `error` (`{ "error": "..." }`).

## Documents

### `GET /api/documents`
List documents in the bucket: `{ "documents": [ { "id", "chunks", "bytes" } ] }`.

### `GET /api/document?doc=<id>`
The full document for preview and highlighting:
```json
{ "id": "doc", "text": "full text...", "meta": { },
  "spans": [ { "id": "doc#0", "pos": 0, "start": 0, "end": 540 } ] }
```
Render `text` and highlight each span to show where retrieved chunks sit.

### `DELETE /api/document?doc=<id>`
Remove a document, its chunks, metadata, and history. Returns
`{ "deleted", "chunks" }`.

## Version history

- `GET /api/versions?doc=<id>` lists `{ "versions": [ { "n", "hash", "time",
  "bytes", "chunks", "current" } ] }`, oldest first.
- `GET /api/version?doc=<id>&n=<k>` returns `{ "text": "..." }` for one version.
- `POST /api/restore?doc=<id>&n=<k>` makes version `k` current by re-ingesting it.

## Graphs and communities

- `GET /api/graph` and `GET /api/entity-graph` export the chunk-similarity and
  entity graphs for visualization (`{ "nodes", "edges" }`).
- `POST /api/build-entities?model=<m>&batch=<n>` (SSE) extracts the entity graph,
  streaming `progress` then `done`.
- `POST /api/build-communities?model=<m>` (SSE) generates a thematic summary per
  community for global queries, streaming `progress` then `done`.
- `GET /api/communities` lists `{ "communities": [ { "label", "size", "summary",
  "chunks", "doc_ids" } ] }`.

## Buckets, config, and assets

- `GET /api/buckets`, `POST /api/buckets`, `DELETE /api/buckets` manage corpora.
- `GET /api/config` and `POST /api/config` read and update runtime configuration
  (backends, chunking, S3); secrets are write-only.
- `GET /api/models` lists available models and embedding readiness;
  `POST /api/pull?model=<m>` (SSE) pulls an Ollama model.
- `POST /api/save` persists the bucket to disk.
- `GET /api/asset/<id>` serves a stored image by its content-addressed id.

## Operations and interop

- `GET /healthz`, `GET /readyz` for liveness and readiness.
- `GET /stats` and `GET /api/status` report corpus and backend state.
- `GET /debug/pprof/...` exposes Go profiling when enabled.
- `POST /v1/chat/completions` is OpenAI-compatible, so existing chat clients work
  unchanged; retrieval knobs are accepted as extension fields and ignored by
  stock clients.

The corpus is also served over MCP (`turbograph mcp`) for agent tools, and the
on-disk `.tg` format plus its JSON export are specified in
[format.md](format.md).
