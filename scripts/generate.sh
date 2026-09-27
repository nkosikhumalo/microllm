#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT="${1:-hello }"
TOKENS="${2:-64}"
cd "$ROOT/go"
go run ./cmd/generate -model "$ROOT/models/exported/model.json" -vocab "$ROOT/data/tokenized/vocab.json" -prompt "$PROMPT" -tokens "$TOKENS"
