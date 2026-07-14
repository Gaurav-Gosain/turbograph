# Research sweep, July 2026

A multi-agent survey of 66 open-source graph-RAG, agent-memory, knowledge-base and
local-search projects. 300 candidate ideas were extracted and adversarially assessed
against turbograph's constraints (Go stdlib only, single static binary, no external
database); 229 survived, collapsing to about 25 distinct mechanisms once deduplicated.

**Read the summary below before the recommendations.** The most valuable output of the
sweep was not a feature list. It was a bug list: reading the survivors back against the
source turned up several defects that silently corrupt the artifact the whole sharing
story rests on. Three of them are fixed (see `git log`): the extraction cache leaked
forgotten documents into the shipped `.tg`, the update path silently re-pointed entity
mentions at replaced content, and merge compared vector dimensions but not embedding
models, so two different 768-dimensional models merged into one incoherent space.

The second finding is that **the A/B harness is at ceiling** and therefore cannot
currently measure any retrieval change (verified: baseline recall@5 = 1.000, MRR =
1.000 on all 12 cases). Everything in Group B is blocked on fixing that first. Porting
retrieval features into an eval that cannot see them would be shipping on faith.

---

# turbograph roadmap — synthesized from the survivor set

## 1. What the landscape actually offers

Stripped of vocabulary, the ~90 surviving ideas collapse into about 25 distinct mechanisms. Most of what looked new is either already shipped under another name (contextual prefixes, RRF, PPR, MMR, content-keyed extraction cache, chunk-level embedding reuse, doc-level version history, abstention) or is cloud/server machinery that dissolves on contact with a single static binary (OCI registries, prolly trees, Merkle anti-entropy, ACLs, page-level WAL replication, ColBERT/MUVERA multi-vector — Ollama pools).

What is genuinely missing falls into four buckets:

1. **Real defects.** The most valuable output of this exercise is not a feature list, it's a bug list. Reading the survivors against the source turned up ~10 concrete correctness/cost defects, several of which silently corrupt the artifact the whole strategy depends on: merge accepts stores from *different embedding models*, the entity graph is never invalidated on a document *update*, `tg ingest` never re-ingests a *changed* file, the extraction cache leaks deleted documents' LLM-derived entity names into a shared `.tg`, and two concurrent `tg add` processes silently clobber each other. Fix these before adding anything.

2. **The write path.** The MCP surface is read-only (`search/answer/get/multi_get`). An agent connected over MCP literally cannot grow the knowledge base — the flagship story does not close. Everything else in the "agents accrete a KB over time" bet is downstream of this.

3. **Incrementality of the derived layers.** `s.commSummary = nil` on *every* `AddDocuments`; `RelevantCommunities` re-embeds every summary on *every* query; a delete triggers a full HNSW + BM25 + edge-rediscovery rebuild; `Load` rebuilds all indexes from scratch (measured ~99s at 50k chunks). For a store an agent appends to daily, the derived layers are economically unusable.

4. **One well-measured retrieval finding.** HippoRAG 2's ablation table indicts `entitySeeds` by name: embedding the whole query and matching it against *entity name vectors* is their **worst** configuration (59.6 avg recall@5 vs 87.1 for the full system). Link the query to **facts**, not entity names, and damp by entity promiscuity. That is the one retrieval idea in the set with a clean, mechanism-explained, 12–27-point delta against a line of code that exists in this repo today.

Everything else in the retrieval column is small, plausible, and unproven — which is fine, because that is what the A/B harness is for. **But the harness is currently at ceiling** (recall@5 = 1.000, MRR = 1.000 on the existing fixture set), so it cannot currently distinguish any of them. Fixing the eval is therefore a hard prerequisite, not a nice-to-have.

---

## 2. Tier 0 — Fix these first. They are bugs, not features.

None of these are ports. They are things this review found. Total: a few days.

| # | Defect | Where | Fix | Effort |
|---|--------|-------|-----|--------|
| B1 | **Deleted documents leak into the shipped `.tg`.** `pruneExtractCacheLocked` is called from `BuildEntityGraph` and `Merge` but **not** from `DeleteDocument`. After `tg forget`, LLM-authored entity names/descriptions derived from that document remain in the gob and ship to whoever you hand the file to. | `rag/update.go` `DeleteDocument` | Call `s.pruneExtractCacheLocked()` after `eg.DropChunks`. Regression test: build 2 docs → `BuildEntityGraph` → delete one → `Save` → assert the deleted doc's entity name is absent from the gob bytes. | S (1 line + test) |
| B2 | **Entity graph is not invalidated on document UPDATE.** `applyPreparedLocked` → `removeDocLocked` never calls `eg.DropChunks` (only `DeleteDocument` does). Chunk IDs are `doc#pos`, so after an edit, `doc#5` still exists but holds *different text* — entity mentions now cite the wrong passage. Silent mis-attribution of evidence. | `rag/update.go` | Call `eg.DropChunks(gone)` + `rebuildEntityLocked` on the update path. Then make chunk identity content-derived (`docID#sha256(text)[:12]`, keep `Pos` as a separate ordering field) so this class of bug is structurally impossible and stored citations survive re-ingest. Keep a load-time compat path for `doc#pos` stores. | S → M (with content IDs) |
| B3 | **Relation descriptions are silently discarded.** `entity/entity.go`: `if d != "" && r.Description == "" { r.Description = d }` — a relation seen in ten chunks keeps only the *first* description. Those descriptions are exactly what `RelationContext` renders into the answer prompt. Nine-tenths of the evidence on every repeated edge is thrown away at ingest. | `entity/entity.go` | Accumulate a deduped `Descriptions []string` (mirror the existing `descSeen` pattern used for entities); keep a computed `Description` for gob back-compat. | S |
| B4 | **`rag.Merge` can silently corrupt the vector space.** The only compatibility gate is `dst.dim != srcDim`. Two stores built with *different* 768-d embedders (nomic vs bge vs a Matryoshka truncation) merge without complaint into one incoherent space. Retrieval just quietly gets worse, forever, with no error. | `rag/merge.go`, `rag/persist.go` | Record an encoder fingerprint in the snapshot: `{EmbedModel, EmbedDim, QueryPrefix, DocPrefix, Normalized, Quant{Bits,ResidualDims,Seed}}`. Merge refuses on mismatch with an actionable diff; absent fingerprint (old files) = warn, allow. Optionally strengthen with a canary: embed one fixed string at build, store `sha256(rounded vec)` — proves two stores share a geometry rather than a model *name*. | S |
| B5 | **No concurrency control on the `.tg`.** `Store.mu` is in-process only; `Save` rewrites the whole gob; there is no lock file. Two `tg add` processes → the second `Save` annihilates the first's ingest. This lands squarely on "agents building a KB over time via the CLI". | new `rag/lock.go`, `cmd/turbograph` save path | `golang.org/x/sys/unix.Flock` (in the dependency budget) on `<store>.lock`; persist a `gen uint64` in the snapshot; save = lock → re-stat → refuse if `gen` moved → `gen++` → temp+rename → unlock. Readers take a shared lock. Document that this is a *local-filesystem* guarantee (flock is unreliable on NFS/SMB). | S–M |
| B6 | **Directory ingest is not idempotent.** `cmd/turbograph/main.go` short-circuits with `if j.Done(id) \|\| store.HasDoc(id) { return nil }` — a file whose *content changed* is never re-ingested, and a file *deleted from disk* stays in the store forever. | `cmd/turbograph` | See recommendation **A4 (`tg sync`)**. | — |
| B7 | **The knowledge-graph facts fed to the generator are ranked by extraction count.** `RelationContext` sorts by `(both_endpoints_grounded, rel.Weight)`, and `Weight` is documented as "number of times the relationship was seen" — monotonically higher for generic hubs and recurring boilerplate. It also never sees the query. | `rag/entitygraph.go` | See **R2 (flow-scored relations)**. | — |
| B8 | **`RetrieveParams.Filter` is dead code and structurally broken.** No caller anywhere sets it, `Chunk` has no metadata field so it cannot see `docMeta`, and it is applied as a **post-filter over the ~3×TopK seeds** — so any selective filter (a scope, a date range, a source) returns near-zero results, silently. | `rag/store.go` | See **A5/A-filter**. `index/hnsw.go:SearchFiltered` already exists (accept-predicate integrated into traversal) and is called by *nothing outside its own test*. | — |
| B9 | **Community summaries are destroyed on every add.** `s.commSummary = nil` in `AddDocuments` and `DeleteDocument`. Forced, because `graph/community.go` compacts labels to `[0,n)` on every run, so `map[int]string` cannot survive a rebuild. | `rag/store.go`, `graph/community.go` | See **A2**. | — |
| B10 | **Every global query re-embeds the entire community summary set.** `RelevantCommunities` calls `s.embedder.Embed` over all summaries per query. On a store with hundreds of communities that is hundreds of Ollama calls per question. | `rag/community.go` | Cache `CommVec [][]float32` in the snapshot beside `CommSummary`. ~10 lines, pure win. | S |

**Measurement for Tier 0:** these are correctness fixes, so the gate is a regression test each, plus a "no nDCG change" run on the existing suite (they must not move retrieval quality — if B2's chunk-ID change *does* move it, that itself is the finding).

---

## 3. Group A — high value for the agent-knowledge-base focus

Ranked. The first four are the ones that make the strategic bet actually work.

### A1. Fact provenance: `Relation.Chunks` + verbatim evidence quote
**From:** Graphiti (episodic subgraph) / KAG (mutual indexing) / GraphRAG (claim `source_quotes`) — three papers, one mechanism.

**Why it wins.** `entity.Relation` is `{Source, Target, Description, Weight}` — it has **no chunk provenance at all**. You can go entity→chunks but never fact→chunks. Consequences today: (a) a fact cannot be cited; (b) `server/verify.go` cannot audit a fact against its source; (c) `DropChunks` only removes a relation when an *endpoint entity* loses all its chunks, so `tg forget`-ing the only document that ever asserted "Alice left Acme" leaves the relation alive with **no supporting text anywhere in the corpus** — an uncitable, unverifiable, permanent fabrication. This is the prerequisite for A13 (supersession), R1 (fact linking), and honest citations.

**Sketch.** `entity/entity.go`: add `Chunks []string` to `ExtractedRelation` and `Relation`; `Graph.AddExtraction` already receives the chunk id (it populates `Entity.Chunks`) — append it to the relation too. **No prompt change, therefore no `extractPromptVersion` bump, therefore no cache invalidation** — this is what makes it cheap. `DropChunks`: filter `relation.Chunks` and delete any relation left with zero provenance (gate on `len(r.Chunks) > 0` so pre-existing stores aren't swept). `rag/persist.go`: `snapshot.Relations` already round-trips `entity.Relation`, so the field persists for free. `RelationContext`: emit `[chunk-id]` alongside each fact.

Separately, and worth one prompt-version bump: extend the *existing* line format to carry an optional verbatim quote per relation (`relation|src|dst|how they relate|quote`). Then `server/verify.go`'s evidence and the KB's evidence become the same objects, and a quote that does not occur (normalized substring) in its source chunk is a free, deterministic hallucination filter.

**Effort:** S (chunk back-pointer) + S (quote, with a prompt bump).
**Measure:** unit test that `tg forget` removes now-unsupported relations; count of relations whose quote fails substring verification (that number *is* your extractor's hallucination rate, and you have never seen it).

---

### A2. Stable community identity + incremental summaries
**From:** Graphiti §2.3 (single-step label-propagation extension) / txtai (incremental topics) — same mechanism.

**Why it wins.** This is the highest value-per-line item for the stated focus. Today, appending **one note** to a 5,000-chunk store discards **every** community summary and forces a full-corpus LLM re-summarization. And it's *forced*, not lazy: `graph/community.go` compacts labels to `[0,n)` on every run, so `commSummary map[int]string` would silently point at different communities if kept. Nulling is the only *correct* option given the key type. So the real port is **give communities a stable identity**; the incremental step follows for free. Without this, the community layer is economically unusable for an accreting store — which means GraphRAG-style global answers, DRIFT, hierarchical trees, and the corpus digest are all dark.

**Sketch.** `graph/community.go`: give each community a stable anchor (the lowest-ordinal member at creation) and return anchor→label; a community that still contains its anchor keeps its identity across rebuilds. `rag/store.go`: `commSummary` becomes `map[string]commRec{Summary string; Members map[string]struct{}}` keyed by the anchor **chunk ID** (stable across reindex, unlike an int label). Add `Communities.Attach(node)`: one plurality vote over the new chunk's similarity-graph neighbours (the CSR is already built) — no full sweep. `AddDocuments` marks only communities whose membership drifted past a threshold (Jaccard < ~0.85 vs the summarized member set) as stale, instead of nulling everything. Persist a `driftCount`; force a full LP + resummarize past ~20% node turnover. Surface drift in `tg stats`. **Improvement over the source:** Graphiti re-summarizes the touched community on every add; don't — a 200-chunk community that gains 3 chunks has a summary that is still correct.

Bundle B10 (cache the summary embeddings) and, from Youtu, a zero-LLM **keyword label** per community (contrastive `localDF/size × globalIDF`, not the source's ascending-local-IDF, which just surfaces stopwords) so the store can name its own topics for free.

**Effort:** M.
**Measure:** LLM calls per `tg add` on a 5k-chunk store (should go from O(communities) to ~0–2); and a drift check — after N incremental adds, compare the incremental partition against a full LP run and report the divergence. *Stale summaries are worse than absent ones*, so the drift counter is the safety property, not the perf number.

---

### A3. Writable agent surface (MCP + CLI)
**From:** ByteRover (typed curate ops) / Cognee (session distillation → `remember`) / Reor (write tools).

**Why it wins.** `cmd/turbograph/mcp.go` registers `search`, `answer`, `get`, `multi_get`. That's it. An agent talking to turbograph over MCP **cannot write**. The strategic focus is "agents building up a knowledge base over time" and the loop does not close. Everything else in Group A is an optimization of a loop that cannot currently run.

The valuable design details, all cheap:
- **Typed ops with a mandatory `reason`**, returning **per-operation status** (the affordance you get for free from being in-process, which an external vector DB cannot give).
- **Entry-level MERGE.** `rag.Merge` merges whole *stores*; there is no way to fold a refinement into an existing document, so an agent that learns a better version of a fact can only append a near-duplicate. The store grows monotonically with restatements forever.
- **Deterministic gates, LLM proposals.** Do **not** put a "is this lesson durable?" judge inside turbograph — that is exactly the discrimination a 4b local model is worst at, and turbograph would become the thing that poisoned the KB. The caller is usually a frontier model that just lived the session. Expose only the deterministic parts: novelty gate (cosine against existing lessons, return `already_known` with the colliding doc id), **glossary anchoring** (return the top-N nearest canonical entity names so the agent can phrase its note using names the graph already knows, instead of minting near-duplicates — this is the best single trick in the Cognee port), templated write, delete-by-id, `--supersede <id>`.

**Sketch.** `rag/curate.go`: `type Op struct{ Kind, ID, Text string; Meta map[string]any; Into, Reason string }`, `Store.Curate(ctx, []Op) []OpResult`. ADD/UPDATE/UPSERT route through the existing `prepareDoc` (which already diffs chunks by content hash and re-embeds only what changed); DELETE through `DeleteDocument`; MERGE is the only new logic (one LLM call to adjudicate, ingest result, delete sources, record provenance to both parents — as a *new* document with its own version history, never silently substituted, because merged text is synthesized and breaks the `Chunk.Text` = verbatim invariant). Every op appends an `Assertion{Op, Agent, Time, Reason, DocIDs}` to a persisted append-only provenance log (which is also the natural carrier for `valid_from/valid_to` later — A13 becomes a *field*, not a redesign). `cmd/turbograph/mcp.go`: register `curate`, `remember`, `glossary` — **gated behind an explicit `tg mcp --writable`, default OFF**, with MERGE/DELETE gated separately from ADD.

**Effort:** M.
**Measure:** not a retrieval metric. The acceptance test is a scripted agent loop: research → `remember` → new session → `search` retrieves the remembered lesson → `--supersede` corrects it → the old lesson stops surfacing. Plus: does `tg log`/`tg revert <assertion>` actually undo one agent's contribution?

---

### A4. `tg sync <dir>` — convergent, idempotent directory reconcile
**From:** memU (manifest diff + cascade delete) / Reor (slim-doc reconcile) / Onyx.

**Why it wins.** Fixes B6. A KB that only ever grows is a KB that slowly fills with lies. And the expensive parts already exist: `Store.idHash` (doc → content sha256) is already persisted, `DeleteDocument` already cascades correctly into the entity graph, and `prepareDoc` already reuses embeddings for unchanged chunks. `tg sync` is mostly wiring. **Do not** port memU's `.memu_manifest.json` sidecar — the store *is* the manifest, and a sidecar would break the single-portable-file property.

**Sketch.** `cmd/turbograph/sync.go`. Walk the dir through the existing extract registry + transform scripts (the pipeline must be byte-identical or digests diverge spuriously); hash the post-extract text; diff against `idHash`. NEW/CHANGED → `Ingest`. ORPHAN → `DeleteDocument`. RENAME (same content hash, new path) → `RenameDocument` (rewrite `Chunk.ID`/`DocID`, re-key the maps, remap entity provenance, then rebuild indexes **from stored embeddings** — zero embedding calls, which is the whole point). Batch all deletions before a single reindex, or a prune that removes 50 files pays 50 full rebuilds. Emit `--json {added, changed, removed, renamed, unchanged}`.

**Safety, non-negotiable:** `tg sync` pointed at a typo'd or not-yet-mounted directory would cheerfully delete an agent's entire accumulated KB. **Dry-run by default; deletions behind `--prune`; hard refusal if the diff would remove more than some fraction of the corpus.**

**Effort:** M.
**Measure:** re-running sync on an unchanged tree = 0 embed calls, 0 LLM calls. Editing one file re-embeds only its changed chunks. A rename costs 0 embed calls. Deleting a file removes it from search *and* from the entity graph.

---

### A5. Chunk timestamps + a real filter path
**From:** Reor (time-filtered ANN, time as an LLM tool parameter) / Khoj (content-date extraction) / sqlite-vec (partition keys) / Onyx (scope tokens) / txtai (`similar()` as a predicate).

**Why it wins.** Two things, one mechanism. First: **`rag` has zero timestamps anywhere.** `Chunk` has no time and no sequence. An agent accreting a KB for six months cannot ask "what did I learn about X last month", and there is no substrate for decay, episodic scoping, or the bi-temporal work. Second: B8 — the filter hook is dead and structurally broken. Fix them together, because a time filter is useless without a working filter path.

**Sketch.**
- `rag/chunk.go`: `CreatedAt int64` (ingest time — the clock turbograph *controls* and can trust; mtime lies on git checkouts). Per-doc `docTime{Created, Modified, Ingested}`. Additive gob fields; old stores load with zeros.
- `rag/store.go`: replace `Filter func(Chunk) bool` with an **ordinal predicate** `Allow func(ord int) bool`, and actually *use* it: pass it into the already-written `index/hnsw.go:SearchFiltered`, and add a predicate to `lexical/bm25.go:Search` so BM25 filters at scoring time, not after top-k.
- **The selectivity fork is the load-bearing detail.** Compute the matching ordinal set first; if it's a small fraction of the corpus (< ~5%), **skip HNSW entirely and brute-force `dotf` over the matching ordinals** — exact, trivially parallel, and turbograph keeps every float32 vector in RAM anyway. Use `SearchFiltered` with inflated `ef` only for broad filters. Filtered-ANN degrades badly at low selectivity; the brute-force fork is what makes a narrow scope *correct* rather than *empty*.
- **PPR must be masked too**, not just seeded — a random walk will happily route mass *through* filtered-out chunks. Filtering the vector and lexical paths is the easy part and will *feel* done; the graph path is where the leak lives.
- Metadata: a small predicate grammar over interned scalars (`--where "source='rfc' AND added_at > '2026-01-01'"`), plus `Scope []string` as the interned fast path. **Freeze the grammar at boolean predicates. Say no to joins and aggregation in the README** — this is the slippery slope that ends with a SQL dialect inside a RAG engine.
- Surface: `--since/--as-of/--where/--scope` on `tg search|ask`, and as MCP tool arguments (the Reor insight worth stealing: make the time window an **LLM-visible tool parameter** and substitute `{TODAY}` into the system prompt, so the agent can scope its own recall).

**Effort:** M.
**Measure:** recall@k under a filter selecting 2% of the corpus, before vs after the brute-force fork (before: ~0; that's the bug). Latency of the brute-force path at 100k chunks. Confirm the graph signal does not leak across a scope boundary.

---

### A6. Bounded entity/relation descriptions + entity-embedding cache
**From:** LightRAG (`_merge_nodes_then_upsert`, fragment lists) / nano-graphrag (compaction, majority-vote type).

**Why it wins.** Three defects, all confirmed, all of which get *worse as the store grows* — i.e. they attack the strategic focus directly.
1. `Entity.Description += " " + d` forever, deduped only by exact string. `embedEntities` embeds `name + ": " + description` as the **PPR seed vector**, so a hub entity mentioned across 500 chunks gets a multi-KB description smear whose embedding is a centroid of everything and matches nothing — and eventually exceeds the embedder's context (nomic truncates ~2k tokens), at which point the entity's vector is determined entirely by whichever descriptions happened to arrive *first*.
2. First-type-wins (`if e.Type == "" { e.Type = typ }`) — an entity typed `concept` once is stuck as `concept` forever.
3. B3 (relation descriptions discarded).

**Sketch — three separable commits, cheapest first, and note that two of the three need no LLM:**
- (a) `entity/entity.go`: count every observed type and resolve by **majority vote** at `Entities()` time. Free, deterministic.
- (b) Accumulate deduped description sets for entities *and* relations; bound both by a **rune budget with deterministic selection** (dedupe, keep descriptions from the highest-mention chunks until the budget is hit). This alone caps `.tg` growth and un-poisons the seed embedding with **zero LLM calls and zero non-determinism**.
- (c) *Only then, optionally:* an LLM compaction pass over entities still over budget, run once after `Canonicalize`/`Prune`, **cached by `sha256(sorted fragments) + model`** (same shape as `extractKey`, persisted alongside `ExtractCache`) so it doesn't break the "rebuild is near-free" property. Keep the raw fragments in the store so compaction is always recomputable — LLM compaction is a *lossy rewrite of grounded extraction text*, and unlike extraction it is not re-derivable from the chunk.
- (d) **Entity-embedding cache** (the biggest cheap win): key each entity vector by `sha256(embedModel + text)` in a persisted map; embed only misses. Today, adding one document to a 10k-entity store **re-embeds all 10k**. This also fixes the latent bug that `rebuildEntityLocked` sets `entVec = nil`, so any `DeleteDocument` currently degrades entity seeding to lexical-only.

**Reject** LightRAG's "contradiction splitting" (LLM decides two fragments describe distinct entities sharing a name) — it pulls directly against `Canonicalize`, which merges surface variants by edit distance. Without a precedence rule you get merge/split thrash across rebuilds.

**Effort:** M.
**Measure:** `.tg` size and max entity description length over a simulated 200-document accretion. Embed calls per incremental `tg add` (should drop to ~0 for untouched entities). Entity-seeded retrieval recall before/after (the seed vectors are changing, so this *must* be A/B'd).

---

### A7. Extraction window + gleanings
**From:** GraphReader (2k-token extraction window) / GraphRAG + nano-graphrag (gleanings loop).

**Why it wins.** `BuildEntityGraph` extracts **per retrieval chunk**, and `DefaultChunkConfig` is `TargetWords: 120` (~160 tokens). Every entity/relation extraction call sees a fragment an order of magnitude smaller than any measured optimum, and is asked to distill self-contained facts from it. The extractor is *structurally starved*. Decoupling the extraction window from the retrieval chunk (concatenate consecutive same-document chunks up to a window budget, extract, attribute results back to constituent chunks) also cuts ingest LLM calls by ~10x — and ingest cost is the per-`add` tax the agent pays *every single time it grows the KB*.

This **supersedes `BatchSize`**, and for a principled reason the codebase already documented: batching *disjoint* chunks makes a small model "lose track of which passage it is reading" (batch of 4 → 6 entities where batch of 2 → 17). A **contiguous window does not have that failure mode** — it is one coherent passage. Same call reduction, without the quality cliff.

Gleanings then rides on top: after the first extraction, re-prompt with the prior output inlined ("You previously extracted: … Emit only entities matching the types already used, in the same format; do not repeat anything above"). Turbograph's `Generator` is single-turn (`Generate(ctx, system, prompt)`), so **simulate the conversation single-turn** rather than widening the interface. Replace the source's `Y/N` continuation gate — a small model asked "did you miss some?" says yes reflexively — with a **structural gate**: continue only if the previous glean actually produced new entity names. Free, deterministic, and it measures the thing the question was a proxy for.

**Sketch.** `rag/entitygraph.go`: group consecutive chunks by DocID/Pos into windows (`EntityBuildOptions.WindowWords`, **a knob, not a hardcoded 2k** — the 2k optimum is GPT-4-class); extract over `strings.Join` of chunk **bodies** (`c.Text`, not `IndexText` — the contextual prefix is an index artifact and must not leak into extraction); attribute entities back to the constituent chunks whose text contains the name/alias, falling back to all chunks in the window. `entity/llm.go`: `MaxGleanings` (default 0 until A/B'd). **The gleaning count MUST enter `extractKey`** (or bump `extractPromptVersion`), or a store built with `--gleanings 0` serves its shallow cached answers forever to a later `--gleanings 2` run and the flag silently does nothing.

**Effort:** M. Both changes force a one-time full re-extraction of existing stores — state that up front.
**Measure:** **precision, not entity count.** The gleaning prompt is a presupposition ("MANY entities were missed") and a 4b model told it missed things will *invent* things; the count goes up either way. Hand-label a sample. Also measure entity-seeded retrieval recall@k (this is a fragile signal — recall@3 already drops 0.958 → 0.83 when entity seeding is enabled, so extraction changes must be checked against it, not just against extraction stats). Sweep `WindowWords` with `TG_AB=1`.

---

### A8. The authored context layer: pinned blocks, path-scoped context, retrieval hints
**From:** Letta (memory blocks) / qmd (path-scoped contexts) / PageIndex ("expert knowledge") / AGENTS.md (override semantics). One mechanism, four faces.

**Why it wins.** *Everything turbograph knows is retrieval-gated.* There is no place to put a fact that MUST be seen. An agent that learns "the canonicalizer wrongly merged Acme Corp and ACME Ltd" or "the benchmark numbers live under Evaluation > Tables, never in the Abstract" has nowhere to write it such that the next `tg ask` cannot miss it — HNSW and BM25 can both whiff, and then the single most important thing in the KB is invisible. That is a structural hole, and it's felt hardest by exactly the workload in focus. It is also the only feature that makes a shared `.tg` carry *retrieval policy* and not just documents.

**Sketch.** One persisted table, three injection points:
- `rag/blocks.go`: `Block{Label, Value, Description string; Limit int; ReadOnly bool; Version uint64}` — persisted in the snapshot (absent in old snapshots → nil → byte-identical default). Rendered into the **answer system prompt** in a `<label><description><metadata chars_current/chars_limit><value>` envelope (the metadata is what lets the model self-regulate).
- `contexts map[string]string` (path-prefix → text). DocIDs are already repo-relative paths (`filepath.Rel(root, path)`), so resolution is prefix-sort + `HasPrefix`, most-specific-first. Injected into the **reranker and answer prompts** and returned as a `context` field on search results.
- `hints []Hint{Scope, Text}` — prose steering injected into the **reranker/decompose** prompts.
- Edit tools worth copying verbatim from Letta: `memory_replace(label, old_str, new_str)` **errors if `old_str` occurs zero or >1 times** (forces a unique anchor — kills the classic "model rewrites the block and silently drops half of it"), and an over-limit edit returns a **tool error** rather than truncating (overflow is the agent's problem, and that is what drives consolidation).
- Expose all of it over MCP so an agent can write policy mid-task.

**Hard constraints, ship them *with* the feature, not after:**
- **`rag.Merge` must NOT import blocks by default.** Blocks land in the *system prompt unconditionally* — a `.tg` received from another agent whose blocks are auto-adopted is a prompt-injection vector with system-level reach, strictly worse than a poisoned chunk (a poisoned chunk still has to win retrieval). `--with-blocks` imports them ReadOnly and label-prefixed.
- Blocks are **excluded from `server/verify.go`'s evidence set** — they are instructions, not citable evidence; a faithfulness audit that treats them as ground truth would launder them.
- A hard total-char ceiling, enforced in `Store`, so the prompt cannot be starved.
- Do **not** put contexts into the *embedding* — that is what makes them cheap to edit; the moment they're embedded, a one-line edit means re-embedding every chunk under that prefix.

**Effort:** S–M.
**Measure:** honestly, this one is hard to A/B and should be justified as closing a structural hole, not as an accuracy win (Letta publishes no benchmark either). What you *can* measure: does a hint reliably fix a known systematic retrieval miss? Does `tg blocks` / `tg context list` make the agent's self-written policy visible? (A wrong block cannot be out-ranked the way a wrong chunk can — it steers every answer forever, silently. Visibility *is* the mitigation.)

---

### A9. Store card: a self-describing `.tg` + dynamic MCP instructions
**From:** Agent Skills (tier-1 name+description = trigger condition) / qmd (MCP instructions built from live index state) / Croissant (`conformsTo` + unknown-property pass-through).

**Why it wins.** There is currently **no way to find out what a `.tg` contains without fully loading it** — and `Load` gob-decodes every embedding *and rebuilds every index*. Answering "which of my 20 stores knows about auth?" today means paying full index reconstruction across all 20. And `mcp/mcp.go`'s `initializeResult` doesn't even populate the protocol's `instructions` field, so turbograph leaves the single best cold-start channel entirely unused: a fresh agent connecting to a store has no idea whether it's worth querying.

The Agent Skills insight is the right one and is subtler than it looks: the tier-1 text is a **trigger condition**, not a summary. "Helps with PDFs" is useless; "use when the user mentions PDFs, forms, or document extraction" is routable. Seed it with the store's own top-PageRank entities and top-IDF terms so it contains *literal corpus vocabulary* a router can match, rather than LLM paraphrase.

**Sketch.** `Card{Name, Description, Docs, Chunks int; EmbedModel, Metric; TopEntities []string; ...}` written as a **separate leading value** in the file (magic prefix + card gob + snapshot gob; `Load` peeks the magic and falls through to the bare-gob path for old files — a field on `snapshot` doesn't work, gob would still have to stream every embedding to reach it). `ReadCard(r)` reads only the header. `tg cards ~/kb/*.tg`. `mcp/mcp.go`: add `Instructions` to `initializeResult`; build the search/answer tool descriptions from live state instead of the current hardcoded literals. Hard-cap at ~300 tokens — tool descriptions are re-sent on every conversation turn in most hosts, so an uncapped manifest is a permanent per-turn token tax.

This is also where **B4's encoder fingerprint** lives, and where the pass-through discipline matters: gob **drops unknown struct fields on decode**, so a newer binary's field is *destroyed* by an older one on the next save. A card with an explicit `Ext map[string]json.RawMessage` bag round-trips.

**Effort:** S–M.
**Measure:** `tg cards` on a directory of stores reads a few hundred bytes each, not gigabytes. Stamp the card with the chunk count/digest it was generated from and mark it **stale** when the store has moved on — a confidently wrong trigger routes an agent to the wrong store and it will never find out.

---

### A10. Secret redaction at ingest
**From:** agentmemory (`privacy.ts`, 14 regexes + `<private>` spans).

**Why it wins.** This is the highest-value item in the whole set relative to its size, and it's buried in the middle of a rejected idea. The strategic focus is **agents growing a `.tg` and sharing it as a portable file**. An agent-grown store is fed from tool outputs — shell history, env dumps, config files, HTTP responses — and it *will* contain API keys. turbograph has **no redaction whatsoever**, and the `.tg` is designed to be handed to a teammate. That is a live secret-exfiltration path in the flagship workflow.

**Sketch.** `rag/redact.go`: compiled `[]*regexp.Regexp` (generic key/token/password, Bearer, `sk-proj-`/`sk-ant-`, GitHub PAT, Slack, AWS, GCP, GitLab, JWT-shaped triple-base64, npm token) plus a `<private>…</private>` span strip. Called in `rag/ingest.go` **before `chunkDoc` and before embedding** — once a key is embedded, quantized into HNSW and written to the gob, it cannot be retracted without a full rebuild. `Config{Redact bool}` **defaulting ON**, with `--no-redact`. **Ship it built in, not as an operator-registered transform script — a security control that is opt-in is not a security control.** Critically: `rag/versions.go` stores full document text history, so **it must store the redacted text**, or the whole thing is defeated.

Same stage, free hygiene: strip inline base64 image blobs (`data:image/`, `iVBORw0KGgo`, `/9j/` prefixes) — turbograph has a real `Kind: "image"` path, so inline blobs are pure index poison. Surface a count in `tg stats` so an operator can see it fired.

**Effort:** S.
**Measure:** a fixture corpus with planted keys; assert none appear in the saved gob bytes, in `versions`, or in `ExportJSON`. **Do not sell it as a guarantee** — regexes false-positive (mangling a doc that legitimately *discusses* a token format) and false-negative (a novel key format). It's a seatbelt; say so.

---

### A11. Gaps ledger — the store records what it does not know
**From:** CRAG (corrective retrieval, but *inverted*: the "external source" is the agent).

**Why it wins.** CRAG's insight is that a failed corpus lookup should trigger a *different* retrieval formulation. But turbograph's external channel **is the agent**, so a failed lookup should return a **work item**, not a Google call. A store that tells you what it does not know, remembers it across sessions, and *keeps* remembering after it is shared, is a feature no other local RAG file format has — and every piece it needs already exists (the entity extractor over the question, `entity/canonical.go`, `s.entIndex` to diff against what the store has ever heard of, `Similarity` as the confidence).

**Sketch.** `rag/gaps.go`: `Gap{Query, KeywordQuery string; MissingEntities []string; BestSim float32; Count int; First, Last time.Time}`, keyed by the canonicalized keyword query, persisted as a new snapshot field. On an uncovered/partial verdict, run the entity extractor over the **question**, canonicalize, diff against `entIndex` — the unmatched surface forms are literally what the corpus has never heard of. `tg gaps [--json]` + an MCP `gaps` tool → a frequency-ranked to-ingest queue. `rag.Merge` sums counts. **Auto-retire:** on ingest, re-run the top-N gap queries and drop any that now clear the threshold — that is what makes the ledger trustworthy rather than a growing pile of stale TODOs. The keyword rewrite needs **no LLM**: stopword-strip + the canonicalized entity surface forms is deterministic and dedupes cleanly.

**Privacy, and it is not a footnote:** a shared `.tg` would now contain **every question anyone failed to answer against it**. Opt-in, with `--no-gaps` and `tg gaps --clear`, decided *before* anyone ships a store with one.

**Effort:** M.
**Measure:** gap precision — what fraction of recorded gaps are real coverage holes vs. junk from a malformed query? (Gate emission on the question containing ≥1 entity the store has never seen; that's the same check that makes the record useful.)

---

### A12. Authored edges: `[[links]]`, `supersedes`, `tg link`
**From:** Basic Memory (deterministic `mem:`/`[[wikilink]]` parsing + forward references) / SiYuan (typed refs) / AGENTS.md (`.override` semantics) / PageIndex (cross-references).

**Why it wins.** turbograph's chunk graph is 100% *derived* — cosine similarity plus LLM-extracted relations. There is no channel through which an agent can **assert** a relationship. For a KB an agent grows over months, asserted edges are the one artifact that compounds with agent-hours and cannot be recomputed from the text; everything else in the store is a pure function of the corpus.

Two sub-features, both zero-LLM:
- **Link edges.** Harvest markdown links, `[[wikilinks]]`, and heading anchors at ingest. These are explicit, already-parsed, 100%-precision pointers — *not* the prose-regex ("see Appendix G") that legal-document papers propose, which fires rarely and with false positives in an agent's markdown notes. A resolved cross-reference is a **non-hallucinatable edge that similarity provably cannot recover** ("see the auth design" has near-zero cosine to the auth design's contents), and it feeds the existing PPR with a new `RefWeight` config knob and *zero* changes to `graph/`.
- **`supersedes`.** `tg add --supersedes <doc-id>` writes into `docMeta` (no schema change — it's already `json.RawMessage`); build a shadow map at load; drop shadowed chunks after scoring, before the TopK cut. This gives **retraction with an audit trail** — the cheap 90% of the bi-temporal gap, with no valid-time schema, and it gives `rag.Merge` a precedence rule where today it has none (Merge is a pure union: merging a teammate's store with a stale contradicting fact under a different id **silently doubles the contradiction**).
- **Forward references.** A link to a doc that does not exist yet is not an error — persist it unresolved, and run a strict exact-match resolution sweep at end-of-ingest **and inside `Merge`**. That last one is what makes two agents' stores actually *fuse* rather than sit side by side. `tg doctor` reports unresolved links (dangling refs accumulate silently forever otherwise).

**Effort:** M.
**Measure:** `RefWeight` must go through `TG_AB=1` — an authored link is high-precision but not necessarily relevant *to this query*, and set too high, PPR will drag in every wiki-linked neighbour and flood the context. Be prepared for the honest outcome that hand-asserted edges are too sparse to move nDCG on any real corpus; ship the `supersedes`/curation value first, treat the retrieval boost as an unproven hypothesis. **Silent suppression is the footgun** — a shadowed doc must surface as `suppressed_by: X` in `--json`, never vanish.

---

### A13. Bi-temporal supersession — but gated on a 100-line go/no-go test first
**From:** MemStrata (keyed deterministic ledger) / Graphiti (entity-pair-scoped invalidation) / Mem0 v3 (append-only reversal) / TOKI (two-clock separation) / "Don't Ask the LLM to Track Freshness".

**Why it wins.** This is the acknowledged key gap, and the survivor set converges on a design that is materially better than the obvious Graphiti port. Three findings, from three independent sources, all pointing the same way:

1. **Never call an LLM to decide "does this supersede that."** Zep/Graphiti scores **7%** and Mem0 **18%** on the versioned-fact task — *below plain BM25 at 48%* — and the diagnosis is LLM judgment entangled at write time, where the error is destructive and permanent. Mem0 subsequently **deleted its own ADD/UPDATE/DELETE reconciliation pass** and reported LoCoMo 71.4 → 92.5. A vendor admitting its published design was wrong, in that direction, is unusually credible.
2. **Cosine cannot do this.** MemStrata's 98-pair calibration: CONTRADICT pairs have *higher* mean cosine (0.812) than duplicate paraphrases (0.800); AUROC for separating duplicates from non-duplicates is **0.59** — chance. No similarity threshold can ever drive invalidation. It has to be a **key lookup**.
3. **Two clocks, not one.** `max(ingest serial)` hands the crown to any stale document an agent backfills tomorrow — which for a CLI-driven KB built over months is the *normal* case, not an edge case. Prefer an opportunistically-extracted world time, fall back to ingest time, and make the fallback **visible** rather than silent.

**Design:** an append-only `Assertion{ID, SubjectKey, RelationKey, Object, SourceChunkID, SentenceSpan, ValidFrom, ValidTo, CreatedAt, ExpiredAt, SupersededBy, Reinforcements}` ledger, with supersession as a **hash-map lookup on `(subject, relation)`** — no cosine, no LLM, no threshold. Retrieval drops hits whose backing assertion has `ValidTo != nil`. Retired rows are never deleted, so as-of queries and "why did the store stop believing X?" fall out for free. **The triple is an index; the *source sentence* is the payload** — pack the original text, not the reconstructed triple (measured: packing triples instead of sentences drops code-mutation accuracy 1.00 → 0.80). Store the sentence **span** (`ChunkID + rune offsets`, which `Chunk.Start/End` already gives you), not a verbatim copy, so the ledger stays an index over the corpus rather than a second source of truth that can drift.

**Prerequisite, and it is the whole gate:** the mechanism has exactly one point of failure — if the extractor folds the value into the subject ("the handler named parseV1" instead of subject="the request handler", object="parseV1"), the key changes when the value changes, the map lookup misses, and **supersession silently degrades into plain append while looking fine**.

> **Build the test before the feature.** A single-slot extraction prompt (`{is_triple, subject, relation, object}`, with fields defined by their role in *change detection*, and the invariant "two statements differing only in the value must produce identical subject and relation") plus a table test over `{stateA, stateB}` fixture pairs asserting `norm(subjA)==norm(subjB) && objA != objB`. ~100 lines. Run it against the extraction cache; wire it into the regression suite so it fires on model swap. **If a local 7B does not hold key stability, A13 will not fire, and it is far better to learn that from a 100-line test than from a large ledger implementation that quietly does nothing.**

Also ship, independently and immediately, the **free deterministic case**: on a same-document update (`applyPreparedLocked`), any relation whose provenance chunks all belonged to the superseded version gets `ValidTo = now` instead of surviving silently. That's the dominant case in practice (an agent rewriting its own note), it costs **zero model calls**, and it needs only A1.

**Effort:** S (the go/no-go test) → L (the ledger).
**Measure:** an *evolving* fixture slice in the A/B suite — re-assertions of existing facts with changed values, ingested in a controlled order, **marker-free** (a build-failing grep for `old/new/current/deprecated/legacy/v1/v2` in the fixtures: removing one leaked marker changed accuracy by up to 14 points in the source, meaning the marker was doing the work). Metrics: **stale-fact-error rate** (answered with the superseded value) under a **forced-answer regime** (abstention *masks* staleness — a system handed both values often says "the sources conflict" and scores as not-wrong). Keep half the suite static: the point is to catch a temporal mechanism that wins on evolving facts by destroying static recall. Add at least one **backfill** case where ingest order and truth order disagree.

---

### A14. Container format: sections, persisted indexes, tombstoned deletes
**From:** Lance (manifest + immutable fragments) / Automerge (chunked container) / sqlite-vec (fixed slabs) / Memvid.

**Why it wins.** Measured, on this repo:

- Full-store gob encode: **0.67s / 262 MB at 50k chunks; 2.6s / 1.05 GB at 200k.** Every `tg add` pays it.
- `Load` rebuild (HNSW + BM25 + edges + communities, none of which are persisted): **11.4s at 10k chunks, 98.9s at 50k.** Every `tg add` pays this too.
- Single-document delete: **117ms at 195 chunks, 401ms at 3195** — and worse than it looks: `removeDocLocked` sets `needsRebuild`, and `reindexLocked` then also resets `s.edges` and `indexedUpTo`, forcing a **full kNN edge rediscovery** over the entire corpus.

So an agent doing fifty `tg add`s against a 200k-chunk KB burns ~50 × (99s reload + 2.6s save) to add fifty documents. **The load/rebuild dominates the save by ~150×** — which means several of the survivor ideas (append-only formats, WALs, Merkle trees) optimize the wrong 1%.

**Do them in this order:**
1. **Persist the derived indexes** (HNSW graph + BM25 postings + edges) as their own sections. This is the 99s. Nothing in the survivor set proposed it, and it is the single biggest cost in the loop.
2. **Sectioned container:** magic + version + length-delimited, checksummed, self-describing sections `[type][len][sha256[:4]][body]`. Unknown section types are **retained verbatim and re-emitted on save** (gob today *drops* unknown fields and destroys them). This gives forward-compat, corruption detection, and merge-by-concatenation. Old bare-gob files sniff and fall through — **do not orphan anyone's store.**
3. **Shard chunks/embeds/codes into fixed slabs (~1024 chunks each)**, content-hashed independently. This is the load-bearing detail: with one monolithic embeds section, appending one document still rewrites 318 MB. With slabs, an append rewrites one ~3 MB tail slab. Sealed slabs are immutable, which also makes cross-version dedup fall out for free.
4. **Tombstoned deletes.** A `dead []uint64` bitset instead of array compaction. `index/hnsw.go:224` **already implements** `searchLayerFiltered` (expand through rejected nodes for connectivity, exclude from results) — the tombstone-safe traversal is already written; it just needs a bitset. Not compacting the arrays keeps **chunk ordinals stable**, which means `s.edges`, the CSR, PPR and communities all survive a delete with **zero rebuild**. Delete becomes a bit flip. `tg compact` does the one true rebuild on the agent's schedule.

**Explicitly reject** sqlite-vec's slot *reuse*: an ordinal in turbograph is a node id referenced by HNSW neighbor lists, the CSR, PPR, and entity evidence lists. Recycling slot 4711 makes every one of those point at a vector that is no longer what they were built against — no crash, no error, just quietly wrong retrieval.

**Effort:** L. Highest blast radius on the list; sequence it after the Tier 0 fixes.
**Measure:** open + one add + save, at 10k/50k/200k chunks, before/after. Delete latency vs corpus size (should go flat). And — because dead nodes leak PPR mass and skew BM25 `avgdl`/`df` between compactions — **retrieval quality** at dead-fraction 0/10/20/30%, to pick the compaction threshold empirically rather than guessing 20%.

---

### A15. Agent-surface tools (a grab-bag of small, high-leverage additions)
**From:** Khoj (grep/list/view) / Serena (degradation ladder) / Hindsight (token budget) / Smart Connections + Reor (`related`) / Serena again (`tg neighbors`, `tg facts`).

Five small things, none of which need new indexes:

- **`grep` / `list_files` / `view_file(line range)` over the corpus.** Semantic search is a *sampling* operator and can never answer "find ALL mentions of X". An agent accumulating a KB needs exhaustive recall as much as ranked recall, and it already knows the grep→open-at-line loop with zero prompt engineering. **The storage precondition is already met:** `rag/versions.go` retains full document text, and `Chunk.Start/End` are exact rune offsets, so a line number is `1 + strings.Count(text[:Start], "\n")` — *exact*, not the fragile `find()`-cursor heuristic the source uses. Emit literal grep format (`path:LINE: content`), hard-cap at 1000 lines with an explicit "narrow your regex" tail, cap `view_file` at 50 lines with a truncation notice (these caps are load-bearing — without them a model will `view_file(1, 10000)` and torch its own context).
- **Degradation ladder.** `multi_get` currently divides a byte budget *evenly* across items and hard-truncates each share — fetch 10 docs and all ten get shredded to 2 KB. Replace with a ladder: full → outline-with-line-offsets → id+size list. **Every rung preserves the addresses needed for a narrower follow-up** — degradation returns a navigation aid, never a truncated blob.
- **`budget_tokens` on `search`/`answer`.** The MCP tools and the CLI are precisely the callers who know their remaining context window and cannot express it. `multi_get` already concedes the point with `max_bytes`. Apply as a greedy pack *after* MMR/rerank. It is an *interface* improvement, not an accuracy win — sell it as such. Return the estimated token count so the agent can check (there's no tokenizer in stdlib; the estimate must be conservative and honest).
- **`tg related <doc|chunk>` — query-free retrieval.** The document you just wrote *is* the query. Reuse the chunk's **already-stored** embedding (`s.embeds[ord]`) — zero embed calls, works with Ollama offline — with hard self-exclusion (`c.DocID != target.DocID`) inside the traversal, then collapse to best-chunk-per-doc and MMR. Closes a loop nothing else closes: *before I write this note, what do I already know that touches it?* — and gives a real near-duplicate guard (today only *exact* duplicates are caught, by sha256). **Do not** port the "exclude already-linked" filter against the *similarity* graph — in turbograph "already connected" is definitionally "is a nearest neighbour", so it would exclude exactly the results you asked for. Only the *entity-KG co-mention* variant is meaningful (`--novel`).
- **`tg neighbors` / `tg facts`.** An agent can search the KB but cannot *walk* the knowledge graph it paid to build. `neighbors(entity)` is a map lookup and a sort over `entity.Graph` — no LLM, no embedding, sub-millisecond. `entity_facts(entity)` returns facts grouped by chunk id, so the agent can scan cheap and only pay for `get` when a group looks promising. That coarse-to-fine gating is the whole content of GraphReader's "action grammar", and exposing it as tools (letting the *calling* agent be the explorer, rather than building an in-process agent loop) is the right architecture for a tool whose differentiator is that the agent is already outside.

**Effort:** S each.
**Measure:** an agent-loop cost benchmark (see **R0**): tool calls and bytes-into-context to reach a correct answer, with vs. without.

---

## 4. Group B — retrieval quality

### R0. Fix the eval first. Nothing below is measurable until you do.
**From:** s3 (Gain-Beyond-RAG + the hard-query filter + GenAcc) / Youtu (open-vs-reject mode) / MemStrata (marker-free + forced-answer) / claude-context (agent-loop token benchmark).

**Why it wins.** The project's own recorded A/B finding is: *"the baseline dense+BM25 is already at ceiling: recall@5 = 1.000, MRR = 1.000, cover = 1.000… there is no headroom, so the advanced features are near-neutral to slightly NEGATIVE."* That is the ceiling problem, verbatim, and it means **the harness currently cannot distinguish any of R1–R16.** Every retrieval recommendation below is conditional on this.

Four cheap upgrades, in order:
1. **Grow the fixture set** and apply the **hard-query filter**: precompute the naive-RAG baseline accuracy per question once (cache it, keyed by `sha256(chatModel+embedModel+corpusHash)`), then **drop every question the baseline already answers**. In the source this cut their pool from 170k to 70k — ~60% of questions were contributing nothing but ceiling. Note this *creates a hard requirement to grow the suite*: filtering 12 fictional docs down to 3 cases and reporting a mean delta over 3 cases is noise, not evidence.
2. **`GenAcc`**: cascade `span_check` (token-boundary contiguous containment over the *existing* `normalizeAnswer`, ~20 lines, free) → LLM judge only on the residual. `ExactMatch` agrees with human judgment on **15.8%** of samples vs **96.4%** for GenAcc — which is the mechanical consequence of scoring a cited-prose sentence against a two-word gold. turbograph's answers *are* cited prose, so EM is essentially always 0 and the harness is near-blind to generation-side changes. Keep the judge **off in CI** (self-preference bias) and report GenAcc *alongside* EM/F1, never instead.
3. **Signed per-query gain**, not arm means. Averages already burned this project once (the "facts hurt" confound) and hid a real regression (decomposition dropping recall@3 from 0.958 to 0.83). A signed per-query delta surfaces both.
4. **Forced-answer + marker-free discipline** for anything temporal (see A13's measurement note).

Separately: an **agent-loop cost benchmark** (`eval/agent_harness.go`) — a minimal ReAct loop over the existing Ollama client, arm A = `read`+`grep`, arm B = same plus turbograph's MCP tools, measuring **tokens and tool calls to task completion** over turbograph's own repo with known ground-truth file sets. That axis is invisible to nDCG and it is the one the MCP is actually sold on. **Guard it hard:** it only means anything *at equal task success*, and with a local model driving the loop, success rates will not be equal — report success first and refuse to report token deltas unless success is within tolerance.

**Effort:** M (mostly fixture authoring, which is the real cost).
**Measure:** meta. The acceptance test is that a known-good change (e.g. turning off BM25 entirely) now *shows up*.

---

### R1. Query→fact linking + degree-damped PPR seeds
**From:** HippoRAG 2 (recognition memory).

**Why it wins.** Best-evidenced retrieval idea in the set, and the only one whose evidence directly indicts a line of code in this repo. `entitySeeds` embeds the whole query and matches it against **entity name+description vectors**, keeping the top 24 above cosine 0.30 plus a lexical bonus. That is precisely HippoRAG 2's `w/ Query to Node` ablation arm — their **worst** configuration (59.6 avg recall@5 vs 87.1 full). Even their `NER to Node` arm (74.6) beats it. **The linking unit is the finding:** link the query to **triples**, not entity names, and the gap is 12–27 points. The mechanism is obviously right on inspection: cosine 0.30 against 24 name vectors is a very loose gate, so one off-topic-but-similar entity name drags the whole walk.

Three parts, cheap and pure Go:
- **Fact index.** Render each relation to a string (`display(src) + " " + Description + " " + display(tgt)`), embed it (copy `embedEntities` verbatim — same batch shape), persist as `FactVec` beside the existing `EntVec`. `entitySeeds` already does a flat linear cosine scan; relation counts are graph-scale (thousands), so a **parallel flat slice is the right shape — no new HNSW, no quantization.**
- **Degree damping.** `fact_score / max(1, len(entity.Chunks))`, averaged over the facts naming that entity, capped at the top 5 phrase nodes, everything else zeroed. An IDF-like penalty that stops promiscuous hub entities from swallowing PPR mass — turbograph has **no such penalty today**, so an entity in 200 chunks seeds just as hard as a rare one.
- **The LLM filter is optional and should be shipped second.** It's worth only ~0.7–1.7 points in their own ablation while the linking-unit change is worth 12–27. If latency bites, ship query→fact linking *without* the filter and take most of the win for free. If you do build it: use a real similarity floor on the fuzzy snap-back, not the reference's `cutoff=0.0`, which always snaps to the nearest candidate and can silently *resurrect* a fact the model meant to drop.

**Bonus for the agent surface:** return the surviving facts alongside the chunks. "*These facts are why these chunks ranked*" is a free explanation trail, and it's exactly what an agent building a KB wants to see.

**Fallback is a real safety property:** if no facts survive, return `nil` seeds and let `store.go`'s existing `escore == nil` fast path hand back the pure hybrid ranking. The graph can never make a query *worse* than vanilla vector search — that matches turbograph's existing philosophy exactly, and the structure is already there.

**Prerequisite:** A1 (`Relation.Chunks`) + multi-predicate retention.
**Effort:** M.
**Measure:** `TG_AB=1`, recall@k and nDCG, against the current `entitySeeds`, on a **sparse** corpus (a young `.tg` with few relations is the common case, and the fallback path dominates there — measure on realistic sparsity, not a dense one).

---

### R2. Flow-scored, query-conditioned relation ranking
**From:** PathRAG (degree-normalized flow), reduced to ~30 lines.

**Why it wins.** Fixes B7. `RelationContext` ranks the facts it feeds the generator by `rel.Weight` = **extraction count** — monotonically higher for generic hub entities and for boilerplate that recurs across chunks. turbograph is actively promoting its least informative edges into the answer prompt. Worse, `RelationContext` **does not even take the query**: it ranks by a query-independent popularity prior.

And the fix is already sitting in the process. `entityChunkScores` computes a query-seeded Personalized PageRank whose propagation is **already degree-normalized** (`share := d*ri/g.outSum[i]`), uses it to score chunks, and then **throws the vector away**. turbograph does not need PathRAG's flow algorithm — it needs to stop discarding the flow it already computes. That is plumbing, not an algorithm.

**Sketch.** Have `entityChunkScores` also return the `ppr []float32`. Change `RelationContext(chunkIDs, maxRels)` → `RelationContext(query, chunkIDs, maxRels)`. Keep `both_endpoints_grounded` as the **primary** sort key (it's a query-*conditioned* relevance signal and it's doing real work); replace the `rel.Weight` tiebreak with `ppr[src] * ppr[tgt] / pow(deg(src)*deg(tgt), beta)`, `beta` a config knob (0 = pure PPR product, 1 = full hub suppression). Also replace `maxRels` (a flat *count*) with a **token budget**, so the relation block and the chunk block compete on a common currency — today a handful of verbose relation descriptions can silently crowd out the chunk block, and nothing tells you it happened.

**Cheap independent baseline worth A/B-ing in the same experiment** (5 lines): rank by `cosine(qv, embed(rel.Description))`, using the fact vectors R1 already builds. **If that beats the flow score, take it and skip the flow entirely.**

**Effort:** S.
**Measure:** `TG_AB=1` on AnswerF1 *and* the faithfulness audit, across three rankers (Weight / flow / description-cosine). Note PPR mass itself concentrates on hubs, so a naive `ppr[src]*ppr[tgt]` could *reinforce* hub bias — the degree divisor and the `beta` knob are not optional.

---

### R3. Unified graph: passage nodes as first-class PPR citizens
**From:** HippoRAG 2 (deep passage integration).

**Why it wins.** Best-measured *structural* idea in the set, and it replaces a hack visible in the source. Today: PPR runs on the entity graph → **projects** entity scores onto chunks by summing `p` over each entity's chunk list → separately runs a chunk-similarity PPR → blends additively via `GraphMix` → convex-blends the entity term via `EntityMix`. That is three ranking signals stitched with two hand-tuned scalars, and the projection step is exactly what HippoRAG 2 ablates away — their `w/o Passage Node` arm is the **single largest loss in their table** (−6.1 avg recall@5, −11 on MuSiQue). The mechanism explains why: a chunk sitting on a hot entity neighbourhood but with poor query-embedding similarity cannot be lifted by a projection that only sums the entities it happens to contain; in a unified walk it accumulates mass *through* the graph.

`graph.Builder` is already a generic weighted-undirected-CSR builder over integer node ids, and `PersonalizedPageRank` already takes `seeds map[int]float32` — the exact shape needed. Node ids `0..len(entList)-1` = entities, `len(entList)+ord` = chunks. Fact edges = `eg.Relations()` with `Weight` (which is *already* a co-occurrence count, matching their `weight += 1.0` semantics exactly). Context edges = `Entity.Chunks`. Synonym edges = R12.

**Two guards that MUST ship with it or the whole thing regresses.** turbograph has **already measured** that letting the graph dominate ranking collapses precision — `GraphMix` defaults to 0 and the source comment says so. HippoRAG 2 reads the chunk ranking *straight off* the PPR output, which is structurally the regime turbograph measured as bad. It only works because of:
1. **Damping 0.5**, not 0.85 (short walks keep mass near the seeds — load-bearing, not incidental).
2. **Dense reset mass on EVERY passage node** (`0.05 × minmax(dense)` over the whole corpus, not just top-k). Every passage node gets nonzero reset mass while at most 5 phrase nodes do. This is the **dense floor**: PPR can only *re-rank* the vector ranking, never destroy it. Their sweep shows a shallow optimum (79.9 / 80.5 / 79.8 / 78.4 / 77.9 across weight 0.01→0.5), which is what a safe dial looks like.

Port it without those two and you will reproduce the old precision collapse and wrongly conclude the idea is bad.

**Do NOT** believe the "you can now delete RRF" claim. turbograph ran that experiment and lost. Keep the existing fusion path as the control arm; do not conflate two changes.

**Effort:** L (also a per-query cost increase: PPR now iterates over `n_entities + n_chunks` — measure it, given a 13x slowdown already killed one graph idea here).
**Measure:** `TG_AB=1`, both arms, both guards on. Multi-hop *and* single-hop, precision *and* recall.

---

### R4. Global path: relevance-gated map-reduce + a working abstention
**From:** GraphRAG / nano-graphrag (scored key points, hard score-0 drop) + Youtu (community keyword tier) + nano-graphrag (`occurrence`).

**Why it wins.** `buildGlobalPrompt` takes the top-k community summaries by cosine and stuffs **all of them** into one prompt with **no relevance floor** — the 8th-best community is always in context whether or not it's relevant. And the "abstain" branch fires only when `len(comms)==0`, which top-k cosine ranking essentially never produces, so **turbograph today has no working abstention on global queries at all.** The best part of the port is the free, corpus-voted "I don't know": if every batch scores every point 0, don't call the LLM — the corpus has voted that it does not contain the answer.

**Sketch.** Split `chatGlobal` into map + reduce. MAP: partition summaries into token-bounded batches, run one `Generate` per batch through the worker pool that **already exists verbatim** in `BuildCommunitySummaries`. **Do NOT copy GraphRAG's nested JSON schema** — use turbograph's house line format (`score|point|[refs]`), parsed by a lenient splitter modeled on `entity.Parse`. A 4b local model emits that far more reliably than nested JSON, and `entity/llm.go` already chose line-delimited output for exactly this reason. A batch that returns nothing parseable contributes zero points rather than failing the query. REDUCE: flatten, hard-drop score==0, sort desc, greedily pack with the score **left visible** ("Importance: 87" — costs nothing, gives the reducer a free weighting signal). Empty survivor list → the existing `abstain` SSE event.

Two free companions, both fixing measured waste:
- **`occurrence`** (distinct chunks covered / max, zero LLM, computable in the pass that already walks members) as a cheap prefilter *before* the LLM ever runs.
- **Community keyword documents in a second BM25 index**, fused with the dense summary rank via the existing `lexical.RRF` — so a rare proper noun reaches the right community even when a dense summary embedding blurs it. Zero LLM (contrastive `localDF/size × globalIDF` over member chunks, *not* the source's ascending-local-IDF, which just surfaces stopwords). Plus B10 (cache the summary vectors).
- Feed surviving high-score points into `server/verify.go` so the `maxVerifyClaims=8` budget is spent on the claims that actually matter, instead of the first 8 sentences regardless.

**Effort:** M.
**Measure:** **how often does score==0 actually fire on a real corpus?** The realistic failure is that a small model scores everything 60–80 and the gate never fires, making the whole thing an expensive no-op; the opposite failure (everything 0 on a hard-but-answerable question) turns a good answer into a **false abstention**, which is worse than a mediocre answer. Mitigate with a *relative* gate (drop the bottom quantile) as well as absolute. Also: this multiplies global-query LLM calls from 1 to `1+ceil(k/batch)` — it's a latency regression on small corpora where the old single prompt was fine.

---

### R5. Ascending context order (10 lines)
**From:** PathRAG (`reversed_relationship`), backed by the independently-established lost-in-the-middle effect.

`buildChatPrompt` writes `Context:` then passages 1..k **best-first**, then the KG facts, then `Question:`. So the passage the retriever is most confident in sits maximally far from the question, in the middle of the context where attention is known to sag. **Select and truncate descending; emit ascending.** The trap: naively sorting ascending *then* truncating keeps the worst passages.

**The citation contract is the one thing that must not break.** Keep `res` in descending order for selection/truncation *and for the `[n]` numbering* — the numbers are what the UI, the faithfulness audit, and the context-window expansion all key off. Emit the loop backwards while still printing `[i+1]`, so the prompt reads `[5] [4] [3] [2] [1]`, facts, question. **Never renumber, only reorder.**

**Effort:** S (~10 lines behind a flag).
**Measure:** `TG_AB=1` on AnswerF1. **Expect a null result** — it's a 56% LLM-judge win in the source and turbograph's TopK may be small enough that there's no "middle" to be lost in. Also watch citation *behaviour*: a small model has a strong learned prior that a numbered list is ordered by importance, and may get confused even if answer quality improves. Cost of trying is ~10 lines; be willing to accept the null and delete it.

---

### R6. Adaptive relevance-cliff cutoff (variable top-k)
**From:** Obsidian Copilot (`AdaptiveCutoff`).

Today `scored = scored[:p.TopK]` pads the LLM's context with whatever ranked 5th–8th no matter how bad it was. Instead: `threshold = topScore × 0.3`; always include at least `floor` (3); never exceed `ceiling` (30); between them, **stop at the score cliff**. Fused with a **two-phase note-diversity fill** — every distinct *document* gets its best chunk before any document gets a second one (O(k) on ids, orthogonal to MMR's O(k²) semantic diversity, and it fixes the common failure where one long document floods the top-k).

The source's warning that a relative threshold is meaningless on raw BM25 does not apply here: turbograph *already* normalizes dense and lexical by their per-query max, so the top score is ~1.0 by construction and the relative threshold is scale-free with no extra pass.

**Ordering matters:** run it **before MMR** (which reorders, breaking monotonicity) and **before Rerank** (which re-blends the scale). Report the cutoff score in the payload — `server/web.go` already has a score-breakdown panel this drops straight into.

**Effort:** S.
**Measure:** tokens-into-context per query, and AnswerF1. Be honest that because relevance is normalized to the per-query max, this measures the *shape of the tail* and **can never say "nothing here is relevant"** — it is not an abstention mechanism and must not be sold as one (`Similarity` + `ShouldAbstain` is that).

---

### R7. Contextual-prefix hygiene (two one-liners, probably the cheapest real win here)
**From:** Late Chunking's negative results + the CDC/caching cluster.

Two independent problems with a shipped feature:
1. **`Chunk.IndexText()` prepends the LLM-generated `Context` to the text used for BM25 as well as for the embedding.** Model-written prose is being stuffed into the lexical postings — diluting IDF and letting a query match the model's *paraphrase* instead of the source's exact terms. That is a far more direct harm channel for precise-fact lookup than any vector effect, and BM25 is precisely the channel needle-in-a-haystack retrieval depends on. **Split `IndexText()` into `EmbedText()` (context + body) and `LexText()` (body only).** One line, and it preserves BM25 as the clean exact-match channel while keeping the dense benefit.
2. **`rag/contextual.go` has no cache at all** — it re-asks the model **per chunk on every single ingest**. Worse, `prepareDoc` explicitly *refuses* to reuse an embedding when `c.Context != ""`, so **with contextual retrieval enabled, the entire embedding-reuse path is dead**: a one-word edit to a document re-embeds *and* re-LLMs the whole thing. Add a context cache keyed on `contentHash(chunk body)` (persisted alongside `ExtractCache`), then the context is stable for a stable body and the reuse guard can be relaxed.

**Effort:** S + S.
**Measure:** (1) `TG_AB=1` on nDCG — this changes the indexed string, so it must be measured. (2) LLM calls + embed calls on re-ingesting an edited document with `--contextual` on (should drop from O(document) to O(edit)).

*Related, larger:* **content-defined chunk boundaries** (gearhash over the *fragment* stream `splitRecursive` already produces, with the inverse-CDF trick to keep sizes tight) would make edits *local*, so an insert at the top of a document stops reflowing every downstream chunk hash and busting both caches. M effort, opt-in strategy, must be A/B'd for quality — and the gear table must be a frozen literal, or every existing `.tg` silently rechunks.

---

### R8. Deterministic sentence-level citation → triage for the faithfulness auditor
**From:** RAGFlow (`insert_citations`).

One embedding batch of the answer's sentences — **no chat call**. Every primitive exists: `server/verify.go:splitAnswerSentences`, `rag/community.go:cosine`, `lexical/bm25.go` term weights, and the chunk vectors are already in `hnsw.Vector(ord)`. Per sentence, `sim = 0.1×tokenCoverage + 0.9×cosine`; cite every chunk within `0.99×max`, capped at 4; splice `[n]` markers reusing the existing convention.

**Critical deviation:** do **not** copy `thr=0.63`. That is calibrated to a specific embedding model; turbograph runs whatever Ollama model the user picked. Use a **relative** criterion (the `0.99×max` tie window *is* portable).

The real payoff is composition: `auditFaithfulness` currently spends its `maxVerifyClaims=8` budget on the **first 8 sentences regardless**. Route **only the uncited sentences** to the expensive audit and the same budget goes to the sentences that actually need it.

**Effort:** S.
**Measure:** claims audited per answer that turn out UNSUPPORTED, before vs after (the triage should raise the hit rate). **Do not oversell "uncited = hallucination"** — embedding similarity is not entailment; a confident fabrication that paraphrases a retrieved chunk's vocabulary will be cheerfully cited. Label it "most similar source", not "source", and never present it in the UI as verification — that would make the faithfulness story *weaker* while appearing to strengthen it.

---

### R9. Structure-aware chunking (three separable chunkers)
**From:** RAGFlow (heading-family auto-detection) / claude-context (AST chunking) / RAGFlow (row-as-chunk tables).

- **Auto-detected heading families.** `markdownChunker` already tracks a heading stack and emits `Piece.Headings`; `chunkDoc` already prepends `A > B > C`. But that only happens for markdown ATX headings, and only when the operator globally sets `Strategy=markdown`. A plain-text manual, a contract, an RFC, a pasted spec with `3.2.1` numbering — all fall to `recursiveChunker` with no breadcrumb. The delta is the **auto-detection vote** across heading families (count regex hits per family, pick the winner, `-1` → fall through to recursive), which makes chunk strategy **per-document** instead of one global config, degrading safely. ~150 lines of `regexp`, zero LLM. Copy the `not_title()` guard verbatim (>12 tokens, or sentence punctuation, or a long unspaced run) — it's what makes it safe.
- **Go AST chunker.** `go/parser` + `go/ast` + `go/token` are **stdlib**. All four current chunkers are size-based, so a Go corpus — including turbograph's own repo — is today being cut mid-function. Walk `file.Decls`, emit one `Piece` per top-level decl with `Headings: ["package rag", "func (s *Store) Search"]` (which gets contextualized for free by shipped machinery). **Greedily bin-pack adjacent sibling decls** up to the target (or a file of 40 one-line getters becomes 40 junk chunks that pollute HNSW, BM25 *and* the PPR graph), and **set overlap to zero** for structural chunks (overlap on a semantically complete unit is pure harm — it inflates duplicate top-K hits and fights MMR). Parse failure → fall back to `recursiveChunker` (a half-written file mid-edit must never fail an ingest). Non-Go languages: extend the **existing operator-registered transform-script** contract with an optional `chunk` mode — an operator drops in a 20-line script using their own parser. Zero deps, zero new attack surface.
- **CSV/TSV row-as-chunk.** `encoding/csv` + `strconv` type inference, one chunk per row rendered as `- col: value` lines (headers can never be split from the row — the actual point), typed columns riding the existing `docMeta` rail. **Hard-cap rows** and fail loudly: row-as-chunk is the wrong tool above a few thousand rows, and the failure is *silent* (memory balloons, and dense retrieval quietly stops discriminating between rows that differ only in a number). Skip XLSX — shared-strings tables and date-serial quirks make "stdlib-native" a trap; the existing `CommandExtractor` already routes it to an external tool.

**Effort:** S–M each.
**Measure:** `TG_AB` per chunker. **Ship each as an opt-in strategy; only flip a default on a version boundary** — changing chunk boundaries changes chunk IDs, invalidates every extraction-cache entry, and forces a full re-embed. That's a migration event for exactly the long-lived stores this project cares about.

---

### R10. Pre-generation sufficiency gate (opt-in)
**From:** EverMemOS (Reconstructive Recollection).

Occupies a pipeline position turbograph genuinely lacks. `decomposeQuery` plans sub-questions **up front and blindly**, paying on every query. `auditFaithfulness` runs **post-generation**, when the answer is already wrong and the cost is sunk. A conditional **pre-generation adequacy gate** is a third thing: it's the only one that can spend more budget exactly when the corpus is thin on this question, and its rewrite is *evidence-conditioned* (it sees what was found and asks only for the gap) rather than speculative. In the source it triggers a second round on **31%** of questions — ~69% pay only the verifier call. The strategy menu (pivot / temporal / concept-expansion / constraint-relaxation) and the forced three-query-style diversity (keyword / natural question / HyDE-statement) are cheap prompt engineering, liftable verbatim.

**Sketch.** `server/sufficiency.go`: `verifyContext(query, res) (sufficient bool, missing []string)` → if insufficient, `gapQueries(query, missing) []string` → retrieve each → merge → run the **existing** rank-blended reranker over the union. **Cap at one extra round; do not loop.** Opt-in (`--verify-context`, `require_sufficient` on the MCP `answer` schema). It should also become the preferred alternative to *unconditional* decomposition: fire decompose only when the verifier says insufficient *and* names a multi-entity gap.

**Effort:** M.
**Measure:** **the trigger rate is the acceptance test.** If it fires on 80% of queries with your default model, the mechanism is not working and it stays off. A small local model is a bad judge of "is this context sufficient" and will fail in one of two useless ways — always sufficient (you paid a call for nothing) or always insufficient (you doubled retrieval cost and added noise to the pool, which measurably hurts precision).

---

### R11–R16 (smaller, all `TG_AB` gated, all plausibly null)

- **R11. Setwise reranker.** `rag/rerank.go`'s `blendWeight()` ramp exists *solely* because a pointwise 0–10 score from a small model is "noisy and uncalibrated" — a defensive hack around miscalibration. A setwise comparator ("which of these c+1 passages is most relevant?" → one token) needs no calibration by construction. **The value is calibration, not speed** — the source's cost argument inverts here (one batched call → ~20 sequential prefills, and Ollama serializes by default). Also port the "output A when uncertain" escape hatch as a *tie* instruction in the current pointwise prompt (3 lines, free): equal scores resolve to retrieval rank automatically, because the blend's base score is monotonically decreasing in rank. **M; measure latency AND nDCG; be willing to delete one of the two rerankers.**
- **R12. Synonym edges** (entity–entity edges weighted by raw cosine, non-destructive, complementing `Canonicalize`'s destructive edit-distance merge which cannot merge JFK/John F. Kennedy and *can* wrongly merge Ada Lovelace / Ada Lovelace Institute). **Do not inherit the 0.8 threshold or the 2047 fanout** — 0.8 on short entity *names* under a local embedder is very noisy (Paris/London, plaintiff/defendant, Q3/Q4 all clear it) and each false synonym is a permanent mass leak. Cap fanout at 5–10, tune per embedding model. Keep their two guards (skip names ≤2 alphanumeric chars, skip self). **S; likely null; ship at weight 0.**
- **R13. Edge-witness co-mention.** Reward a chunk for citing *several mutually adjacent* entities (the passage that *witnesses* the edge) rather than one high-PPR entity. ~30 lines inside `entityChunkScores`, all inputs exist. Fold in as a **bounded multiplicative bonus** on top of PPR — do **not** copy the source's lexicographic sort, which throws away turbograph's continuous PPR score. **S; alpha=0 default.**
- **R14. BM25 stemming.** `lexical/tokenize.go` is 47 lines with no stemmer, so "what meetings did I attend" misses "attending a meeting". ~100 LOC Porter/Snowball, no deps. Use a **conservative** stemmer — turbograph's corpora are exactly the kind where a `TurboQuant`/`turboquantize` collision would hurt. **S.**
- **R15. Doc2query in the Context slot.** Append 3–5 generated questions to the *same* cached contextual-prefix call (index-only, never shown to the generator — `IndexText` vs `Text` is already the enforced boundary). Zero new indexes, zero new chunks, zero GC, zero faithfulness risk. **Explicitly reject** replacing the chunk's embedding with its questions — any content no generated question covers becomes unretrievable, and the index stops representing the corpus. **M; A/B against contextual retrieval, which already occupies this niche.**
- **R16. Multi-store federation.** `tg search --store a.tg --store b.tg --weight b=0.3`. `rag.Manager` already owns N independent stores; `lexical.RRF` already exists. Because RRF fuses on **ranks**, cross-store composition needs **no dimension check and no shared embedding model** — strictly more permissive than `Merge`, which hard-rejects it. Reversible (unmount), attributable (which store said this). **State plainly in the docs that PPR, communities and the entity graph do NOT compose** — mounting fuses *retrieval*, not *knowledge*, and a graph-RAG product that silently drops its graph signal is the worst possible surprise. **M.**

---

## 5. Group C — everything else (worth doing, not urgent)

- **Self-expanding schema** (Youtu). A closed `{Nodes, Relations, Attributes}` type vocabulary bounding both extraction and decomposition, grown with a support threshold (promote a type only after ≥μ distinct chunks propose it), canonicalized via the *existing* `entity/canonical.go` applied to *type strings*. Genuinely attacks the open-vocabulary bleeding that `plausibleEndpoint`/`genericNames` exist to mop up. **The sharp hazard:** the schema goes into the extraction prompt, so it must enter `extractKey` — but then every schema growth invalidates the entire cache, and early in a KB's life the schema changes on almost every chunk → O(n²) re-extraction, destroying the "rebuild is near-free" property. Resolve it deliberately (freeze per build epoch, promote at the *end*) or this is a performance regression, not an improvement. **M, medium risk.**
- **Entity cards** (Hindsight). LLM-summarize an entity's accumulated fragments instead of concatenating them; embed `Name + ": " + Card` in `embedEntities`; inject the top-matching cards into the answer prompt. "Tell me about X" is the dominant query shape for an agent's own KB and community summaries don't serve it. Mostly subsumed by A6(c). **Off the write path** — one LLM call per touched entity per ingest is 50 calls for a document naming 50 entities.
- **Op log / `tg undo`** (Turso CDC). A coarse, doc-level `Op{Kind, DocID, Hash, PrevHash}` tape — *not* byte-level before/after images, which `rag/versions.go` already provides. `tg log`, `tg undo`, `tg undo --since`. Requires fixing `DeleteDocument`'s `delete(s.versions, id)` (it destroys the very text an undo needs). **The tape must not become a transaction log** — canonicalization and pruning are whole-corpus derivations; they must be re-derived on the new base, not inverted. **M.**
- **`tg branch`** = `cp store.tg .branches/exp.tg` + a `{Name, ParentHash}` sidecar; `--adopt` verifies the parent hasn't moved (stale-rebase check) then `os.Rename`s over main. Every command already takes `--store`, so a branch needs almost no new code. Speculative ingest becomes safe and reversible. **S.** (Not the COW/fragment-sharing version — that's a cloud answer to a problem a local file doesn't have.)
- **`tg stats` / footprint report.** Derived-text tokens (community summaries + entity descriptions + contextual prefixes + extract cache) vs source text, reported **separately from vector bytes**. A derived/source *text* ratio > 1.0 is a genuine smell; a naive combined metric would report embeddings as "inflation" and make every store look like a failure. Also: `Versions` stores the **full text of every prior version forever** and `Merge` copies it — redact an API key, re-ingest, and the key still ships. Needs `tg export --no-history` / `tg vacuum`. **S.**
- **`tg export --skill-dir`** (lossy markdown interop for agents that can't run the binary) and, more valuably, **`tg ingest ./some-skill/`** — using a SKILL.md's `description` as the contextual-retrieval prefix for every chunk in the bundle, which is a free, human-written, high-quality prefix and points turbograph at the growing corpus of public skill repos. ~60-line hand parser, no YAML dep. **S–M.**
- **Reject-mode/open-mode eval** (Youtu). turbograph's *production* prompt is already reject mode ("Use only the context"); what's missing is the **open-mode counterfactual**, so the delta measures how much of your own benchmark number is the LLM's parametric memory. Cheap (one extra system prompt + scoring column). **But near-meaningless on private agent corpora** — document that the gap is only interpretable on public/memorized corpora, or it becomes a number dutifully reported and never acted on. **S.**
- **`structured_search`** (qmd): let the agent supply typed sub-queries (`lex`/`vec`/`hyde`/`entity`/`community`) with weights. The clever bit is **caller-supplied HyDE** — the caller is already an LLM, so it writes the hypothetical answer document itself and turbograph just embeds it as the query vector: HyDE with **zero LLM calls on turbograph's side**. Fuse with turbograph's *existing* score-additive blend, **not** qmd's RRF-with-magic-constants (turbograph deliberately abandoned rank fusion for final ranking; the acceptance test is that a plan of one `lex` + one `vec` at default weights degenerates to exactly today's behaviour). Ship alongside the automatic `search`, never as a replacement. **M.**
- **`intent` parameter** (qmd): a second, non-matching input channel ("I'm debugging why OAuth refresh fails in staging") threaded into the reranker/decompose prompts but **never into BM25 or the query vector** (assert that in a test). Zero cost on the hot path. **Off by default** — it's a confirmation-bias amplifier: told the searcher's hypothesis, a pointwise judge will systematically up-rank passages that *confirm* it and down-rank the passage saying the real cause is elsewhere, which is the single most valuable passage in that search. Phrase it as task *context*, not a hypothesis to confirm. **S.**
- **`links` on search results** (txtai): after MMR/rerank, an O(k²) neighbour lookup restricted to the selected set → an adjacency list an agent can *follow* (with `get`) instead of re-querying, plus a free redundancy diagnostic (a result set that's one dense clique means MMR isn't working). **Do not** serialize it into the *prompt* without an A/B — the edges are cosine ≥ 0.5, i.e. "these two chunks use similar words", which is very close to the opposite of corroboration, and inviting a model to treat graph proximity as evidential support is a plausible faithfulness regression. **S.**
- **Sentence-level max-sim evidence spans** (ColBERT-lite, query-time only): after retrieval, split the top-K into sentences, one batched embed call, `max_j cos(q, sentence_j)`, keep `argmax_j` and map it back through `locateSpan` to rune offsets → return a **one-sentence evidence span** instead of a 500-token chunk. For an agent surface that's worth more than the ranking delta, and it costs one embed call with **no ingest cost and no format change**. Only pay for stored sentence codes if the query-time version proves the lift. **M.**
- **`tg conflicts`** (TOKI, stripped of the operator algebra): once an assertion ledger exists, a conflict is just a key whose version list holds >1 distinct object. "What does this knowledge base disagree with itself about?" is a first-class question turbograph cannot answer at all, and it's ~40 lines. **Park-and-surface, never block a write** — an agent ingesting 500 documents overnight cannot stop and wait for a human. **S (given A13).**

---

## 6. Deliberately not doing

The tempting ones, and why.

**Any agentic loop that puts N LLM calls on the query path.** DRIFT (60–100+ generations/query), Self-RAG's critic-weighted beam search (up to ~96 generations × a critic call per node), CRAG-MH (sequential per-hop conflict pipelines), PageIndex's vectorless LLM tree search (a *serial* LLM call per tree level, unbatchable). Every one of these is minutes per question on a local 7b, and turbograph's positioning is *fast and local*. Where a good idea is buried inside one (DRIFT's "seed decomposition from global structure", CRAG's "route the failure back to the agent"), it's taken as a small delta above — never as the architecture.

**Anything that makes a similarity threshold responsible for deciding what's a duplicate or a contradiction.** MemStrata's 98-pair calibration is decisive: CONTRADICT pairs sit at mean cosine **0.812**, *above* duplicate paraphrases at **0.800**, and the Merge class (the pairs that most need *both* kept) has the **highest** cosine of all at 0.938. AUROC 0.59 — chance. This kills consolidation-by-cosine, decay-by-similarity, and the "cluster near-duplicates and merge them" family. It also means: **run that 98-pair calibration on turbograph's own embedder** and keep the result, so the next person who proposes a cosine dedup knob can be shown a number rather than argued with.

**LLM contradiction/invalidation judgment at ingest.** Zep/Graphiti scores 7% and Mem0 18% on the versioned-fact task — *below plain BM25 at 48%* — and Mem0 deleted its own reconciliation pass and gained 21 points. Beyond the evidence: such a verdict depends on **corpus state**, not chunk content, so it is **not content-cacheable** and it does **not survive `Merge`** — you would be trading away the single property that makes rebuilds and merges cheap.

**Quantized rescoring during graph traversal.** Already measured at 13× and rejected. Two survivors are the same shape (LM-DiskANN fat nodes; cognee's expand-then-rescore). The prior should be strongly negative.

**Executable code inside the `.tg`.** Agent Skills' `scripts/` tier proposes shipping executable bodies inside the artifact whose entire purpose is to be passed between agents. That is a malware distribution format, and it directly reverses the operator-registered-by-name rule already written into `script/script.go`'s package doc. The feature has value *only* in the sharing case — which is precisely the case where the trust gate must refuse. A consent dialog over an opaque payload is the weakest control there is, and the design's own selling point is that the source **never enters the reviewer's context**. Ship scripts beside the `.tg`, in a repo, like any other code.

**ACLs / permissions / policy engines.** Onyx's ACL tokens, MemOS's governance metadata, Croissant's ODRL licence evaluator. A scope token in a gob file the reader fully controls is not an access control — anyone holding the file can pass any token, or just read the chunks. Shipping this as "ACL" would be security theatre and a genuinely dangerous promise. Ship the *same mechanism* as **namespaces/tags** (A5), honestly named. A half-enforced licence evaluator is worse: it produces a machine-readable claim of compliance that no counsel will honour, and invites someone to ship restricted text because the binary said `true`.

**Access-reinforced decay / recency boosts.** Making `Retrieve` write access counters turns a read-only command into a read-modify-write on a multi-hundred-MB gob, makes the same query on the same `.tg` return different results depending on usage history (breaking the eval harness and the portable-file story), and creates an explicit rich-get-richer loop. Age is also not staleness: a stable API contract ingested 60 days ago is not less true than yesterday's throwaway note, and a recency multiplier systematically buries exactly the foundational knowledge a long-lived KB exists to hold. If a salience signal is wanted, the corpus-intrinsic, deterministic, already-computed one (`Entity.Mentions` / `Relation.Weight`) needs no write path — and note turbograph has *already measured* that centrality-flavoured reranking lowers precision.

**Replacing chunk embeddings with generated questions** (RAGFlow), and **`^30` field boosts on generated keywords.** Any content no generated question happens to cover becomes unretrievable; the index stops representing the corpus and starts representing what a 4b model guessed someone might ask. A 15× lexical weight on a hallucinated keyword is a precision disaster.

**Multi-vector retrieval** (ColBERT/PLAID/MUVERA/late chunking). All of it needs per-token or per-span encoder output. Ollama's `/api/embed` returns pooled vectors only. Late chunking is reachable via a `llama.cpp --pooling none` provider, but it forks the UX (unreachable from Ollama, the local-first default), requires per-model char↔token alignment calibration that WordPiece tokenizers make approximate, and **silently produces garbage** on a CLS-pooled model rather than erroring. Revisit only if the local embedding-provider landscape changes.

**Cloud-scale storage machinery.** Prolly trees, Merkle anti-entropy sync, OCI registry protocols, embedded WALs, page-level replication. These pay for random-access update, cross-tenant dedup, and network transfer. turbograph is append-dominant, single-process, and its distribution model is "copy this one file" — at which point you have already transferred every byte and the Merkle tree saved exactly nothing. The measured bottleneck is **index rebuild on load** (99s at 50k chunks), which none of them address. A14 does.

---

### One-paragraph sequencing

Tier 0 (bugs) → **R0** (fix the eval, or nothing below is measurable) and **A1** (fact provenance, which unblocks half the list) → **A3** (writable MCP, or the strategic bet doesn't close) → **A2 / A4 / A5** (make an accreting store actually work) → **R1 / R2** (the two retrieval changes with real evidence against real code) → **A6 / A7** (extraction quality) → **A10** (redaction, before anyone shares a store built from tool output) → then the long poles (**A14** container, **A13** ledger, **R3** unified graph), each gated on a measurement and each with an explicit delete condition.