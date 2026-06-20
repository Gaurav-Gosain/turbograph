// End-to-end example against a running turbograph server.
//
//   1. Build the client:  npm run build
//   2. Start turbograph on http://localhost:8080
//   3. Run:               node examples/basic.mjs
//
// This script ingests a few documents, runs a query, streams a chat answer,
// and (if a model is configured) builds communities for a global question.

import { Turbograph } from "../dist/index.js";

const tg = new Turbograph({
  baseUrl: process.env.TURBOGRAPH_URL ?? "http://localhost:8080",
  bucket: process.env.TURBOGRAPH_BUCKET ?? "default",
});

async function main() {
  // --- Status ------------------------------------------------------------
  const status = await tg.status();
  console.log(
    `turbograph ${status.version}; generation model: ${status.generation.model} (reachable: ${status.generation.reachable})`,
  );

  // --- Ingest plain text -------------------------------------------------
  const ingest = await tg.ingestText([
    {
      id: "rust",
      text: "Rust is a systems programming language focused on safety and performance. It has no garbage collector and enforces memory safety via ownership.",
      meta: { topic: "languages" },
    },
    {
      id: "go",
      text: "Go is a statically typed compiled language designed at Google. It has a garbage collector and lightweight goroutines for concurrency.",
      meta: { topic: "languages" },
    },
  ]);
  console.log(`ingested; corpus now has ${ingest.chunks} chunks`);

  // --- Query -------------------------------------------------------------
  const hits = await tg.query("which language has no garbage collector?", {
    top_k: 3,
  });
  console.log("\nTop query hits:");
  for (const h of hits) {
    console.log(
      `  [${h.doc_id}] score=${h.score.toFixed(3)} sim=${h.similarity.toFixed(3)}  ${h.text.slice(0, 60)}...`,
    );
  }

  // --- Streaming chat ----------------------------------------------------
  console.log("\nStreaming chat answer:");
  for await (const ev of tg.chat("Compare memory management in Rust and Go.", {
    top_k: 4,
  })) {
    if (ev.type === "sources") {
      console.log(`  (using ${ev.sources.length} sources)`);
    } else if (ev.type === "token") {
      process.stdout.write(ev.text);
    } else if (ev.type === "abstain") {
      console.log(`  abstained: ${ev.message}`);
    } else if (ev.type === "error") {
      console.error(`  error: ${ev.error}`);
    } else if (ev.type === "done") {
      console.log("\n  [done]");
    }
  }

  // --- Buffered chat -----------------------------------------------------
  const result = await tg.chatText("What are goroutines?", { top_k: 3 });
  console.log(`\nBuffered answer (${result.sources.length} sources):`);
  console.log(result.abstained ? `abstained: ${result.abstained}` : result.answer);

  // --- Global chat over community summaries ------------------------------
  if (status.generation.reachable) {
    console.log("\nBuilding community summaries...");
    const done = await tg.buildCommunitiesSync();
    console.log(`built ${done.communities} community summaries`);
    const global = await tg.chatText("What themes does this corpus cover?", {
      global: true,
    });
    console.log("Global answer:");
    console.log(global.abstained ? `abstained: ${global.abstained}` : global.answer);
  }

  // --- Documents and versions -------------------------------------------
  const docs = await tg.documents();
  console.log(`\nDocuments: ${docs.map((d) => d.id).join(", ")}`);
  const versions = await tg.versions("rust");
  console.log(`'rust' has ${versions.length} version(s)`);
}

main().catch((err) => {
  console.error("example failed:", err);
  process.exitCode = 1;
});
