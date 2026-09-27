#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/java"
mvn -q package
java -jar target/microllm-train-0.1.0-SNAPSHOT.jar --vocab "$ROOT/data/tokenized/vocab.json" --tokens "$ROOT/data/tokenized/tokens.json" --output "$ROOT/models/exported/model.json" --learning-rate 0.001 "$@"
