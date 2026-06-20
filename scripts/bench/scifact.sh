#!/usr/bin/env bash
# Reproduce the SciFact retrieval numbers from docs/benchmarks.md.
#
# Downloads the BEIR SciFact dataset, then ingests and scores it with the shipped
# default configuration using `turbograph bench`. Requires a running model
# server: a local Ollama with the embedding model pulled by default, or pass
# --embed-api/--embed-url/--embed-key for an OpenAI-compatible endpoint.
#
# Usage:
#   scripts/bench/scifact.sh [extra turbograph bench flags...]
#
# Example with a specific model and JSON output:
#   scripts/bench/scifact.sh --embed-model embeddinggemma --json scifact.json
set -euo pipefail

DATA_DIR="${BENCH_DATA_DIR:-/tmp/turbograph-bench}"
SCIFACT="$DATA_DIR/scifact"
URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

mkdir -p "$DATA_DIR"
if [ ! -d "$SCIFACT" ]; then
  echo "downloading BEIR SciFact to $DATA_DIR" >&2
  curl -fsSL "$URL" -o "$DATA_DIR/scifact.zip"
  unzip -q "$DATA_DIR/scifact.zip" -d "$DATA_DIR"
fi

# Build the binary if it is not already on PATH.
BIN="${TURBOGRAPH_BIN:-}"
if [ -z "$BIN" ]; then
  go build -o "$DATA_DIR/turbograph" ./cmd/turbograph
  BIN="$DATA_DIR/turbograph"
fi

echo "scoring SciFact (document-level nDCG@10, Recall@10) ..." >&2
"$BIN" bench \
  --format beir \
  --corpus "$SCIFACT/corpus.jsonl" \
  --queries "$SCIFACT/queries.jsonl" \
  --qrels "$SCIFACT/qrels/test.tsv" \
  --k 10 \
  "$@"
