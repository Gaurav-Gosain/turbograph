# The `.tg` store format

A `.tg` file is a complete turbograph corpus: its chunks, their embeddings, the
content history, and an optional entity graph, in one portable file. Save one,
copy it anywhere, load it, and you are back exactly where you left off without
re-embedding a thing. This document is the format's specification.

## At a glance

```
turbograph-data/
  default.tg     one bucket, one file
  research.tg    another bucket
  archive.tg
```

A bucket is a `.tg` file. The server's data directory holds one file per bucket
(`<name>.tg`); the CLI reads and writes a single store path. Loading rebuilds the
search indexes from the stored embeddings, so the expensive step (embedding) is
never repeated.

## Encoding

A `.tg` file is a single [`encoding/gob`](https://pkg.go.dev/encoding/gob)
stream holding one `snapshot` value. There is no magic number, no header, and no
text framing: the file is the gob encoding of the struct below and nothing else.
gob is self-describing (it carries field names and types), so a snapshot written
by an older build loads in a newer one, and fields added later default to their
zero value when absent. That is the whole compatibility story; there is no
explicit version integer to bump.

The struct that gets encoded (`rag/persist.go`):

```go
type snapshot struct {
    Cfg       Config                   // how this corpus was built
    Dim       int                      // embedding dimension
    Chunks    []Chunk                  // the text, in ingestion order
    Embeds    [][]float32              // raw embeddings, one per chunk
    Hashes    map[string][32]byte      // doc id -> content hash (dedup)
    Entities  []entity.Entity          // entity graph nodes (optional)
    Relations []entity.Relation        // entity graph edges (optional)
    Versions  map[string][]docVersion  // per-document content history
}
```

## What is stored, and why

The format keeps only the inputs that are expensive to recompute and
deterministically rebuilds everything else on load.

| Field       | Holds                                                | Why it is (or is not) stored |
|-------------|------------------------------------------------------|------------------------------|
| `Cfg`       | chunking, quantization, HNSW, graph, and fusion knobs | so a reload reproduces the same build; a bring-your-own `Chunker` is **not** stored (it is code, not data) and must be reattached after loading |
| `Dim`       | the embedding width                                  | to size the indexes before adding vectors |
| `Chunks`    | each chunk's id, doc id, ordinal, and text           | the source of truth for the lexical index and previews |
| `Embeds`    | one raw `float32` vector per chunk                   | the source of truth for the vector index and MMR; storing these is what makes a reload skip embedding |
| `Hashes`    | a content hash per document id                       | so content-level dedup survives a reload |
| `Entities`  | entity-graph nodes (name, type, description, chunks) | the entity graph is extracted with an LLM, far too costly to rebuild on load |
| `Relations` | typed, weighted edges between entities               | same reason; together with `Entities` it restores the GraphRAG graph |
| `Versions`  | each document's content snapshots (hash, time, size, chunk count, full text) | powers the version history, diffs, and restore |

Everything derived is left out and rebuilt on load: the quantizer, the HNSW
vector index, the BM25 lexical index, the chunk-similarity graph, and the
detected communities. This keeps the file small, immune to index-internal layout
changes, and forward-compatible.

## Load and save

```go
// Save the current store.
f, _ := os.Create("research.tg")
store.Save(f)
f.Close()

// Load it back, attaching an embedder for future queries and ingestion.
f, _ = os.Open("research.tg")
store, _ = rag.Load(embedder, f)
f.Close()
```

`Save` refuses to write an empty store. `Load` reconstructs the indexes from the
stored embeddings, restores the entity graph if present, and returns a store
ready to query. The embedder you pass to `Load` is only used for new queries and
new documents; nothing already in the file is re-embedded.

## Properties to rely on

- **Self-contained.** One file is one corpus. No sidecar files, no database.
- **Portable.** Copy it between machines; load it with any build that can decode
  the struct.
- **Embedding-free reload.** Loading costs indexing time minus the embedding
  step, which is the part that talks to a model.
- **Forward-compatible.** New optional fields read as zero in older files; older
  files load in newer builds.

## Caveats

- A custom `Chunker` is not persisted. After loading, set `Cfg.Chunker` again if
  you ingest more documents with a bring-your-own splitter; the built-in
  `Strategy` string is stored and needs nothing.
- The file is gob, not JSON. Read it with a turbograph build (or the Go `gob`
  package against the struct above), not a text editor.
- There is no encryption or checksum. Treat a `.tg` file like any other data
  file: the embeddings are derived from your source text, so handle it with the
  same care as the documents that produced it.
