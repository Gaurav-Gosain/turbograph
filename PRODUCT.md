# Product

## Register

product

The web UI is an app shell (graph canvas, chat panel, ingest, config, command
palette), so design serves the task. With one qualifier: this UI is also the
showcase. It is what someone looks at to decide whether turbograph is worth
running. It therefore has to carry identity, not just utility — but identity
earned through precision and information density, never through decoration.

## Users

**Engineers running a local-first RAG workbench over their own corpus.** They are
on their own machine, with their own documents, with Ollama running locally and no
data leaving the box. They are technical, they are fluent in the category's best
tools, and they are skeptical: they have seen RAG demos that look impressive and
retrieve badly.

Their job is not "chat with my documents." It is:

- get a grounded answer they can actually trust, with citations they can audit;
- understand *why* a chunk was retrieved (dense? lexical? graph? entity?), so they
  can tell an exact keyword hit from a graph-associated one;
- tune retrieval on their own corpus and see whether a change helped or hurt.

**A second class of user is an agent**, not a person. turbograph serves the corpus
over MCP (`search`, `get`, `multi_get`, `answer`) and an OpenAI-compatible endpoint.
Agents search, then pull the exact regions they need within a context budget. The
API surface is a designed surface, and it holds to the same standard as the UI.

## Product Purpose

A fast, local-first graph-RAG engine: one static Go binary, no external vector or
graph database, embeddings and generation from a local Ollama. It quantizes and
indexes embeddings, fuses dense and lexical retrieval, propagates relevance across
a chunk-similarity graph and an optional entity knowledge graph, and answers with
numbered citations.

**Success is not a fluent answer. Success is a trusted one.** The user should be
able to see the retrieval score broken into its parts, read the exact prompt the
model received, watch each claim get checked against the evidence, and A/B two
retrieval configs on their own documents. The product wins when the user stops
guessing and starts measuring.

## Brand Personality

**Precise. Honest. Fast.**

The voice is measured and technical, and it does not oversell. The README says what
is approximate and what is exact; the benchmarks doc reports the changes the data
did *not* support. That candor is the brand, and the interface has to speak the same
way: state the number, show the breakdown, admit the uncertainty. No hype words, no
"AI-powered," no exclamation marks.

The feeling to evoke is the one you get from a good instrument: it is dense, it is
legible, it responds instantly, and you trust its readings. Confidence, not delight.

## Anti-references

All four are explicitly rejected:

- **The generic AI chat app.** Purple/indigo gradients, glassmorphism, rounded chat
  bubbles, sparkle icons, "AI-powered" everywhere. turbograph is a retrieval
  instrument that happens to have a chat surface, not a ChatGPT clone.
- **The enterprise dashboard.** Heavy chrome, cards nested inside cards, hero
  metrics, toolbars of unlabelled icon buttons. Density is welcome; ceremony is not.
- **The toy dev tool.** Bouncy motion, emoji, mascots, cute empty states. Anything
  playful undermines trust in a measurement tool.
- **The sterile academic demo.** Unstyled defaults, an unfinished research artifact.
  Being technical is not a licence to be ugly.

The line through all four: **nothing decorative, nothing cute, nothing ceremonial,
nothing unfinished.**

## Design Principles

1. **Show the work.** The signature move. Every number is decomposable and every
   claim is auditable: the retrieval score breaks into dense / lexical / graph /
   entity, the exact assembled prompt is one click away, each answer sentence can be
   checked against its evidence, and two retrieval configs can be diffed side by
   side. Where a competitor shows a score, turbograph shows *why*.

2. **The instrument, not the app.** Terminal-grade precision. Monospace throughout is
   the identity, not an accident. Density is a feature; these users want information,
   not whitespace. The tool disappears into the task.

3. **Fast is a feature you can feel.** Speed is in the name, the AVX kernels, and the
   architecture. The interface must never squander it: motion is 150–250ms, it starts
   fast, and it conveys state rather than performing.

4. **Measure, don't assert.** Nothing ships on a hunch. A change to retrieval is
   benchmarked, and if the data does not support it, it is dropped and the negative
   result is written down. This applies to design too: if a UI change is claimed to
   help, show the before and after.

5. **Local-first means the user is in control.** Their data never leaves the machine,
   and the same is true of authority: every advanced feature (graph boost, entity
   mix, reranking, decomposition, verification, transform scripts) is opt-in,
   explained, and reversible. The product never quietly does something clever behind
   the user's back.

## Accessibility & Inclusion

Best effort, pragmatic; do the cheap right things, don't let them block feature work.

- **Contrast:** aim for 4.5:1 on body text. The current palette's `--muted` and
  `--faint` on the near-black surfaces are the known risk; fix violations when
  touching a surface rather than in a dedicated sweep.
- **Reduced motion:** honored globally (`prefers-reduced-motion` disables transforms
  and collapses durations). Non-negotiable, and already in place.
- **Keyboard:** the product already claims full keyboard control (command palette,
  shortcuts). Keep that true; visible focus states are part of it.
- **Not currently invested in:** deep screen-reader semantics and ARIA for the graph
  canvas. Acknowledged gap, deliberately deferred.
