#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/go"
go run ./cmd/tokenize -input "$ROOT/data/raw/train.txt" -output "$ROOT/data/tokenized"
