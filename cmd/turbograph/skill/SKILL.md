---
name: turbograph
description: Build and query a durable knowledge base with turbograph. Use when you learn something worth keeping across sessions, when you need to recall what you or a teammate previously learned about a codebase or domain, or when you want to hand a knowledge base to someone else as a single file.
---

# turbograph

A knowledge base is **one file**: `kb.tg`. You add to it, search it, and hand it to
someone else. It holds the text, its embeddings, a similarity graph, and an entity
graph, so retrieval finds things by meaning and by association, not just by keyword.

You drive it from the shell. There is no server to start and no service to configure.

## Setup

Point at a store once per session and everything else picks it up:

```bash
export TURBOGRAPH_STORE=./kb.tg      # every command defaults to this
export TURBOGRAPH_MODEL=qwen3.5:4b   # only needed for `ask` and `entities`
```

`turbograph add` creates the store on first write. There is no init step.

## The loop

**Before you answer from memory, search.** The store is the memory; your context is not.

```bash
turbograph search --q "how does auth token refresh work" --topk 6
```

Returns JSON: `{"query": ..., "hits": [{"id", "doc_id", "score", "text"}]}`. Read the
`text`, and cite `doc_id` when you use it. If you need the whole source document
rather than the matching passage, `turbograph docs` lists the ids.

**When you learn something durable, add it.** Pipe it straight in:

```bash
turbograph add --id "auth/token-refresh" <<'EOF'
Refresh tokens rotate on every use. The old token is revoked immediately, so a
retry with a stale token fails with 401 rather than reissuing. This is why the
mobile client's offline queue must not replay a refresh.
EOF
```

Give it a real `--id`. The id is the unit of update: adding the same id again
**replaces** that document, which is how you correct something you got wrong. Without
an id you get a content hash, and you can never update it again — only add near
duplicates beside it.

Attach metadata when the provenance matters:

```bash
turbograph add --id "decisions/2026-07-queue" --meta '{"source":"design-review","date":"2026-07-14"}' < notes.md
```

**When something turns out to be wrong, remove it.** A knowledge base that only
accumulates will confidently serve you stale facts.

```bash
turbograph docs                       # see what is in there
turbograph forget --id "auth/token-refresh"
```

## What to put in it

The test is: **would a competent teammate, joining in three months, want to be told
this?** That is the bar.

Worth adding:

- A decision and the reasoning behind it, especially the option you rejected and why.
- A non-obvious constraint: why the retry is capped at 3, why this table is not indexed.
- A hard-won fact about how a system actually behaves, as opposed to how it reads.
- A summary of a long investigation, in the form you would want to find it in.

Not worth adding:

- Anything already in the code, the README, or the git history. Retrieval will not
  beat `grep` on the code, and duplicating it means it goes stale independently.
- Conversation transcripts, your own reasoning traces, or tool output. Store the
  conclusion, not the path to it.
- Anything you would not act on if you found it. Noise dilutes retrieval: every
  irrelevant chunk is a chunk that can outrank a relevant one.

Write each entry so it stands alone. It will be retrieved out of context, months
later, by someone who cannot ask you what you meant.

## Grounded answers

`ask` retrieves and then answers from what it retrieved, and tells you what it used:

```bash
turbograph ask --q "why can't the offline queue replay a refresh?" --json
```

```json
{"question": "...", "answer": "...", "sources": [{"id": "auth/token-refresh#0", "doc_id": "auth/token-refresh"}]}
```

**Check the sources.** If the answer is not supported by them, the model made it up;
say so rather than passing it on. If `sources` is empty, the store does not know, and
the right answer is "nothing in the knowledge base covers that."

`ask` needs a model. `search` does not — prefer `search` and read the passages
yourself when you can, since it is faster and there is nothing to hallucinate.

## The entity graph

Extracting entities and relationships lets retrieval follow associations: a question
about a person surfaces passages that never name them but are linked through the
graph.

```bash
turbograph entities                              # reads each chunk with the model
turbograph search --q "..." --entity 0.5         # blend the entity graph into ranking
```

This is the expensive pass: it reads every chunk with a language model, so a large
corpus takes minutes. It is cached by content, so running it again after adding a few
documents only reads the new ones and costs almost nothing. Run it after a batch of
additions rather than after each one.

## Sharing

The `.tg` file **is** the knowledge base. Copy it, commit it, send it, attach it to a
release. Someone else can search it immediately with no setup.

Merge stores to combine what two people (or two agents) learned separately:

```bash
turbograph merge --into team.tg alice.tg bob.tg
turbograph entities --store team.tg    # nearly free: the merged stores carry their extraction cache
```

Merging is idempotent and content-addressed: merging the same store twice adds
nothing, and a document both stores have is not duplicated. So you can re-merge
freely as stores evolve.

Both stores must have been built with the same embedding model. Merging stores built
with different models fails with a clear error rather than producing a corrupt index.

## Notes

- Every command takes `--json` (and `search` emits JSON by default) — parse it, do
  not scrape the human output.
- `turbograph docs --json` is the cheapest way to see whether the store knows about a
  topic at all before you spend a retrieval on it.
- Writes are atomic. An interrupted `add` leaves the previous store intact.
- The store is a single file. Back it up by copying it.
