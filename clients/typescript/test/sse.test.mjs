// SSE parser unit tests. They require no live server: a tiny in-memory
// ReadableStream is fed to parseSSE and the parsed events are asserted.
//
// Run after building:   npm run build && node --test ./test/
// (the test imports the compiled output from ../dist/index.js)

import test from "node:test";
import assert from "node:assert/strict";

import {
  parseSSE,
  parseChatEvent,
  toBase64,
  fromBase64,
} from "../dist/index.js";

/** Build a ReadableStream that emits the given string in the given byte chunks. */
function streamOf(text, chunkSize = text.length) {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (offset >= bytes.length) {
        controller.close();
        return;
      }
      const end = Math.min(offset + chunkSize, bytes.length);
      controller.enqueue(bytes.subarray(offset, end));
      offset = end;
    },
  });
}

async function collect(stream) {
  const out = [];
  for await (const msg of parseSSE(stream)) out.push(msg);
  return out;
}

test("parses a single event", async () => {
  const msgs = await collect(
    streamOf('event: token\ndata: {"text":"hi"}\n\n'),
  );
  assert.equal(msgs.length, 1);
  assert.equal(msgs[0].event, "token");
  assert.equal(msgs[0].data, '{"text":"hi"}');
});

test("parses multiple events in one payload", async () => {
  const wire =
    'event: sources\ndata: {"sources":[]}\n\n' +
    'event: token\ndata: {"text":"a"}\n\n' +
    'event: token\ndata: {"text":"b"}\n\n' +
    'event: done\ndata: {"done":true}\n\n';
  const msgs = await collect(streamOf(wire));
  assert.deepEqual(
    msgs.map((m) => m.event),
    ["sources", "token", "token", "done"],
  );
});

test("reassembles events split across byte chunks", async () => {
  const wire =
    'event: token\ndata: {"text":"hello"}\n\nevent: done\ndata: {"done":true}\n\n';
  // One byte at a time stresses the buffering logic.
  const msgs = await collect(streamOf(wire, 1));
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].event, "token");
  assert.equal(msgs[1].event, "done");
});

test("joins multiple data lines with newlines", async () => {
  const msgs = await collect(streamOf("event: x\ndata: a\ndata: b\n\n"));
  assert.equal(msgs[0].data, "a\nb");
});

test("ignores comment lines and strips one leading space", async () => {
  const msgs = await collect(
    streamOf(": keep-alive\nevent: token\ndata: spaced\n\n"),
  );
  assert.equal(msgs.length, 1);
  // Exactly one leading space after the colon is stripped per the SSE spec.
  assert.equal(msgs[0].data, "spaced");
});

test("handles CRLF line endings", async () => {
  const msgs = await collect(
    streamOf("event: token\r\ndata: {\"text\":\"x\"}\r\n\r\n"),
  );
  assert.equal(msgs.length, 1);
  assert.equal(msgs[0].event, "token");
});

test("flushes a trailing event without a final blank line", async () => {
  const msgs = await collect(streamOf("event: done\ndata: {}\n"));
  assert.equal(msgs.length, 1);
  assert.equal(msgs[0].event, "done");
});

test("parseChatEvent maps SSE messages to typed events", () => {
  assert.deepEqual(
    parseChatEvent({ event: "token", data: '{"text":"hi"}' }),
    { type: "token", text: "hi" },
  );
  assert.deepEqual(
    parseChatEvent({ event: "abstain", data: '{"message":"none"}' }),
    { type: "abstain", message: "none" },
  );
  assert.deepEqual(parseChatEvent({ event: "done", data: '{"done":true}' }), {
    type: "done",
    done: true,
  });
  const src = parseChatEvent({
    event: "sources",
    data: '{"sources":[{"id":"c1","doc_id":"d1","score":0.5,"similarity":0.4,"text":"t","start":0,"end":1}]}',
  });
  assert.equal(src.type, "sources");
  assert.equal(src.sources.length, 1);
  assert.equal(src.sources[0].id, "c1");
  assert.equal(parseChatEvent({ event: "unknown", data: "{}" }), null);
});

test("base64 round trip works", () => {
  const bytes = new Uint8Array([0, 1, 2, 250, 255, 128, 64]);
  const b64 = toBase64(bytes);
  assert.deepEqual(Array.from(fromBase64(b64)), Array.from(bytes));
});
