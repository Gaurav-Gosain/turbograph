# @turbograph/client

A clean, dependency-free TypeScript client for [turbograph](../../), a fast
graph-RAG server. It works in modern browsers and Node 18+ using the global
`fetch` and `ReadableStream`. There are no runtime dependencies; TypeScript is
the only devDependency.

## Features

- Strongly typed methods and interfaces for the full turbograph HTTP+JSON API.
- Streaming chat over server-sent events, exposed as an async iterable you can
  drive with `for await`.
- Bucket-scoped: every call appends the `bucket` query parameter.
- A typed `TurbographError` carrying the server's `{error}` message on any non-2xx.
- A small, standalone SSE parser that reassembles events across byte-chunk
  boundaries and works the same in Node and the browser.

## Install

This package lives in the turbograph repository. Build it from source:

```sh
cd clients/typescript
npm install
npm run build      # compiles src/ to dist/ (ESM + .d.ts)
```

Then import it:

```ts
import { Turbograph } from "@turbograph/client";
```

## Construct a client

```ts
const tg = new Turbograph({
  baseUrl: "http://localhost:8080", // default
  bucket: "default",                // default; appended as ?bucket=...
});

// A bucket-scoped client sharing the same settings:
const work = tg.withBucket("work");
```

## Ingest

```ts
// Plain text.
await tg.ingestText([
  { id: "doc1", text: "Rust has no garbage collector.", meta: { topic: "lang" } },
]);

// Rebuild the corpus from scratch instead of appending.
await tg.ingestText(docs, { replace: true });

// Binary files (PDF, etc.) for server-side text extraction.
await tg.ingestFiles([{ id: "paper.pdf", b64: base64String, meta: { year: 2024 } }]);

// An image: stored, captioned by a vision model, then indexed by its caption.
await tg.ingestImage({
  id: "fig1",
  image: bytes,        // Uint8Array or a base64 string
  ext: "png",
  model: "llava",
  prompt: "Describe this figure for retrieval.",
});
```

## Query

```ts
const hits = await tg.query("which language has no garbage collector?", {
  top_k: 5,
  graph_mix: 0,   // personalized-PageRank graph signal; 0 disables it
  mmr_lambda: 0,  // MMR diversification; 0 disables it
  entity_mix: 0,  // entity-graph signal in [0,1]
});

for (const h of hits) {
  console.log(h.doc_id, h.score, h.similarity, h.text);
}
```

## Streaming chat

`chat()` returns an async iterable of typed events. The server emits a
`sources` event first, then `token` events, then `done`; or `abstain` when the
evidence gate fires, or `error` on a server error.

```ts
for await (const ev of tg.chat("Compare Rust and Go memory management.", {
  top_k: 4,
  rerank: false,
  metaKeys: ["topic"],
  history: [{ role: "user", content: "Tell me about Rust." }],
})) {
  switch (ev.type) {
    case "sources":
      console.log(`using ${ev.sources.length} sources`);
      break;
    case "token":
      process.stdout.write(ev.text);
      break;
    case "abstain":
      console.log("abstained:", ev.message);
      break;
    case "error":
      console.error("error:", ev.error);
      break;
    case "done":
      console.log("\n[done]");
      break;
  }
}
```

### Buffered chat

`chatText()` consumes the stream and resolves to the full answer and sources:

```ts
const { answer, sources, abstained } = await tg.chatText("What are goroutines?", {
  top_k: 3,
});
```

### Global (corpus-wide) chat

Pass `global: true` to answer thematic questions from community summaries
(build them first; see below):

```ts
const result = await tg.chatText("What themes does this corpus cover?", {
  global: true,
});
```

### Cancellation

Pass an `AbortSignal` to cancel a stream:

```ts
const ac = new AbortController();
setTimeout(() => ac.abort(), 5000);
for await (const ev of tg.chat("...", { signal: ac.signal })) { /* ... */ }
```

## Documents and versions

```ts
const docs = await tg.documents();                 // DocInfo[]
const view = await tg.document("doc1");            // DocView with chunk spans
await tg.deleteDocument("doc1");

const versions = await tg.versions("doc1");        // DocVersion[], oldest first
const text = await tg.versionText("doc1", 1);
await tg.restore("doc1", 1);
```

## Entity graph and communities (LLM indexing passes)

Both stream SSE progress. Use the iterable form to show progress, or the
`*Sync` convenience to block until done.

```ts
// Iterate progress:
for await (const ev of tg.buildEntities("llama3", { batch: 4 })) {
  if (ev.type === "progress") console.log(`${ev.done}/${ev.total}`);
}

// Or block until done:
const entities = await tg.buildEntitiesSync("llama3");
const communities = await tg.buildCommunitiesSync("llama3");

const summaries = await tg.communities();          // CommunitySummary[]
```

## Models, status, save, buckets

```ts
const models = await tg.models();   // { models, default, pdf, embed_model, embed_ready }
const status = await tg.status();   // version, storage, generation, embedding, stats
await tg.save();                    // persist the current bucket to disk

const buckets = await tg.buckets();
await tg.createBucket("work");
await tg.deleteBucket("work");      // the "default" bucket cannot be deleted
```

## Assets (ingested images)

```ts
const url = tg.assetUrl(hit.image_ref);            // direct URL for an <img src>
const data = await tg.getAsset(hit.image_ref);     // Blob in browsers, Uint8Array in Node
```

## Error handling

Every non-2xx response throws a `TurbographError` carrying the server's
`{error}` message, the HTTP status, the requested URL, and the parsed body.

```ts
import { TurbographError } from "@turbograph/client";

try {
  await tg.document("missing");
} catch (err) {
  if (err instanceof TurbographError) {
    console.error(err.status, err.message);
  }
}
```

## Testing

The SSE parser and event mapping have unit tests that need no live server:

```sh
npm test     # builds, then runs node --test over ./test/
```

## License

MIT
