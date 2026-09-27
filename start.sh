#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CALLER_DIR="$PWD"
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_ACCENT=$'\033[38;5;213m'
  C_INFO=$'\033[38;5;117m'
  C_OK=$'\033[38;5;84m'
  C_DIM=$'\033[2m'
else
  C_RESET="" C_ACCENT="" C_INFO="" C_OK="" C_DIM=""
fi
ui_title() { printf "\n%s╭──────────────────────────────────────────────╮%s\n" "$C_ACCENT" "$C_RESET"; printf "%s│       MiniLLM  ·  QUANTIZATION STUDIO         │%s\n" "$C_ACCENT" "$C_RESET"; printf "%s╰──────────────────────────────────────────────╯%s\n\n" "$C_ACCENT" "$C_RESET"; }
ui_step() { printf "\n%s  %s  %s%s\n" "$C_INFO" "$1" "$2" "$C_RESET"; }
ui_welcome() {
  printf "\n%s╭────────────────────────────────────────────────────────╮%s\n" "$C_ACCENT" "$C_RESET"
  printf "%s│  YOUR TEXT IN. A SMALLER MODEL OUT.                     │%s\n" "$C_ACCENT" "$C_RESET"
  printf "%s╰────────────────────────────────────────────────────────╯%s\n" "$C_ACCENT" "$C_RESET"
  printf "\n%sMiniLLM learns patterns from your files, then makes two compressed copies.%s\n" "$C_INFO" "$C_RESET"
  printf "PTQ compresses the trained model. QAT fine-tunes for compressed weights.\n"
  printf "Then the benchmark compares size and text prediction quality.\n"
  printf "\n%sYour files stay on this computer. Choose one or more files in a moment. Use Ctrl-click to select several.%s\n\n" "$C_DIM" "$C_RESET"
  sleep 3
}
pick_dataset_path() {
  if [[ -t 0 && -t 2 ]] && command -v zenity >/dev/null 2>&1; then
    zenity --file-selection --multiple --separator=$'\n' --title="Choose training files" --file-filter="All files | *" 2>/dev/null
    return
  fi
  if [[ -t 0 && -t 2 ]] && command -v kdialog >/dev/null 2>&1; then
    kdialog --getopenfilename "$CALLER_DIR" "*" --title "Choose a local dataset file"
    return
  fi
  printf "Dataset file path: "
  IFS= read -r REPLY || return 1
  printf '%s' "$REPLY"
}
normalize_local_path() {
  local raw="$1" decoded="" char hex index=0
  raw="${raw#"${raw%%[![:space:]]*}"}"
  raw="${raw%"${raw##*[![:space:]]}"}"
  if [[ "$raw" == file://* ]]; then
    raw="${raw#file://}"
    case "$raw" in
      localhost/*) raw="/${raw#localhost/}" ;;
      /*) ;;
      *) echo "Only local file:// URLs are supported: $1" >&2; return 2 ;;
    esac
    while (( index < ${#raw} )); do
      char="${raw:index:1}"
      if [[ "$char" == "%" && "${raw:index+1:2}" =~ ^[[:xdigit:]]{2}$ ]]; then
        hex="${raw:index+1:2}"
        printf -v char '%b' "\\x$hex"
        index=$((index + 3))
      else
        index=$((index + 1))
      fi
      decoded+="$char"
    done
    raw="$decoded"
  fi
  if [[ "$raw" == "~" || "$raw" == "~/"* ]]; then
    raw="${raw/#\~/$CALLER_DIR}"
  fi
  if [[ "$raw" != /* && ! -e "$CALLER_DIR/$raw" && -e "/$raw" ]]; then
    raw="/$raw"
  fi
  printf '%s' "$raw"
}
ACTION="${1:-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$ROOT/go"
case "$ACTION" in
  run|learn)
    DATA_PATHS=()
    EVAL=""
    EPOCHS=100
    QAT_EPOCHS=2
    LEARNING_RATE=0.001
    QAT_LEARNING_RATE=0.00005
    VAL_FRACTION=0.1
    WINDOWS_PER_EPOCH=0
    OUTPUT_DIR=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --data) DATA_PATHS+=("${2:?--data requires a path}"); shift 2 ;;
        --eval) EVAL="${2:?--eval requires a text file}"; shift 2 ;;
        --epochs) EPOCHS="${2:?--epochs requires a number}"; shift 2 ;;
        --qat-epochs) QAT_EPOCHS="${2:?--qat-epochs requires a number}"; shift 2 ;;
        --learning-rate) LEARNING_RATE="${2:?--learning-rate requires a number}"; shift 2 ;;
        --qat-learning-rate) QAT_LEARNING_RATE="${2:?--qat-learning-rate requires a number}"; shift 2 ;;
        --val-fraction) VAL_FRACTION="${2:?--val-fraction requires a number}"; shift 2 ;;
        --windows-per-epoch) WINDOWS_PER_EPOCH="${2:?--windows-per-epoch requires a number}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
        -h|--help)
          echo "Usage: ./start.sh run [--data FILE_OR_DIR ...] [--eval HELDOUT.txt] [--epochs N] [--qat-epochs N] [--output-dir DIR]"
          exit 0
          ;;
        *) echo "Unknown run option: $1" >&2; exit 2 ;;
      esac
    done
    if [[ ${#DATA_PATHS[@]} -eq 0 ]]; then
      ui_title
      ui_welcome
      if ! PICKED_PATHS="$(pick_dataset_path)"; then
        echo "File selection canceled. No training was started."
        exit 0
      fi
      if [[ -z "$PICKED_PATHS" ]]; then
        echo "No files were selected." >&2
        exit 2
      fi
      mapfile -t DATA_PATHS <<< "$PICKED_PATHS"
    else
      ui_title
    fi
    for index in "${!DATA_PATHS[@]}"; do
      DATA_PATHS[$index]="$(normalize_local_path "${DATA_PATHS[$index]}")"
      [[ "${DATA_PATHS[$index]}" = /* ]] || DATA_PATHS[$index]="$CALLER_DIR/${DATA_PATHS[$index]}"
      [[ -f "${DATA_PATHS[$index]}" || -d "${DATA_PATHS[$index]}" ]] || { echo "Dataset path not found: ${DATA_PATHS[$index]}" >&2; exit 2; }
    done
    DATA="${DATA_PATHS[0]}"
    if [[ -n "$EVAL" ]]; then
      EVAL="$(normalize_local_path "$EVAL")"
      [[ "$EVAL" = /* ]] || EVAL="$CALLER_DIR/$EVAL"
      [[ -f "$EVAL" ]] || { echo "Evaluation file not found: $EVAL" >&2; exit 2; }
    fi
    if [[ -z "$OUTPUT_DIR" ]]; then
      DATA_NAME="$(basename "$DATA")"
      DATA_NAME="${DATA_NAME%.*}"
      OUTPUT_DIR="$ROOT/models/runs/$DATA_NAME"
    elif [[ "$OUTPUT_DIR" != /* ]]; then
      OUTPUT_DIR="$CALLER_DIR/$OUTPUT_DIR"
    fi
    TOKEN_DIR="$OUTPUT_DIR/tokenized"
    FP32="$OUTPUT_DIR/model.fp32.json"
    PTQ="$OUTPUT_DIR/model.ptq.int8.json"
    QAT="$OUTPUT_DIR/model.qat.int8.json"
    REPORT="$OUTPUT_DIR/benchmark.json"
    mkdir -p "$TOKEN_DIR" "$OUTPUT_DIR"
    ui_step "01 / 06" "Preparing your files"
    TOKEN_INPUT_ARGS=()
    for source_path in "${DATA_PATHS[@]}"; do TOKEN_INPUT_ARGS+=(--input "$source_path"); done
    go run ./cmd/tokenize "${TOKEN_INPUT_ARGS[@]}" --output "$TOKEN_DIR"
    if [[ -z "$EVAL" ]]; then EVAL="$TOKEN_DIR/corpus.txt"; fi
    ui_step "02 / 06" "Preparing the training engine"
    (cd "$ROOT/java" && mvn -q package)
    ui_step "03 / 06" "Learning from your data"
    java -jar "$ROOT/java/target/microllm-train-0.1.0-SNAPSHOT.jar" \
      --vocab "$TOKEN_DIR/vocab.json" --tokens "$TOKEN_DIR/tokens.json" \
      --output "$FP32" --learning-rate "$LEARNING_RATE" --epochs "$EPOCHS" \
      --val-fraction "$VAL_FRACTION" --windows-per-epoch "$WINDOWS_PER_EPOCH"
    ui_step "04 / 06" "Compressing trained weights · PTQ"
    go run ./cmd/quantize --input "$FP32" --output "$PTQ" --scheme per-channel
    ui_step "05 / 06" "Tuning for compressed weights · QAT"
    go run ./cmd/qat --input "$FP32" --tokens "$TOKEN_DIR/tokens.json" \
      --vocab "$TOKEN_DIR/vocab.json" --output "$QAT" --epochs "$QAT_EPOCHS" \
      --learning-rate "$QAT_LEARNING_RATE" --val-fraction "$VAL_FRACTION" \
      --windows-per-epoch "$WINDOWS_PER_EPOCH"
    ui_step "06 / 06" "Comparing model quality and size"
    go run ./cmd/benchmark --fp32 "$FP32" --ptq "$PTQ" --qat "$QAT" \
      --eval "$EVAL" --vocab "$TOKEN_DIR/vocab.json" --report "$REPORT"
    printf "\n%s✓ Your compressed models are ready%s\n" "$C_OK" "$C_RESET"
    printf "  PTQ  %s\n" "$PTQ"
    printf "  QAT  %s\n" "$QAT"
    printf "  Report  %s\n\n" "$REPORT"
    ;;
  quantize|ptq|quantize-ptq)
    if [[ $# -eq 0 ]]; then
      INPUT="$ROOT/models/exported/model.json"
      OUTPUT="$ROOT/models/quantized/model.int8.json"
      if [[ ! -f "$INPUT" ]]; then
        echo "No source model found at $INPUT" >&2
        echo "Pass --input PATH --output PATH, or create a compatible export first." >&2
        exit 2
      fi
      mkdir -p "$(dirname "$OUTPUT")"
      exec go run ./cmd/quantize --input "$INPUT" --output "$OUTPUT" --scheme per-channel
    fi
    exec go run ./cmd/quantize "$@"
    ;;
  qat|quantize-qat)
    (cd "$ROOT/java" && mvn -q package)
    exec go run ./cmd/qat "$@"
    ;;
  benchmark)
    exec go run ./cmd/benchmark "$@"
    ;;
  test)
    go test ./...
    (cd "$ROOT/java" && mvn test)
    ;;
  help|-h|--help)
    cat <<'HELP'
MiniLLM Quantization Engine

Usage:
  ./start.sh                              Show an overview, then open a local multi-file picker
  ./start.sh run [--data FILE_OR_DIR ...]  Pick files or pass one or more local paths
  ./start.sh ptq [quantize flags]         Run post-training quantization
  ./start.sh qat [QAT flags]              Fine-tune with fake quantization, then quantize
  ./start.sh benchmark [flags]            Compare FP32, PTQ-INT8, and QAT-INT8 checkpoints
  ./start.sh test                         Run Go and Java tests

Run defaults: 100 training epochs, 2 QAT epochs. Outputs are stored in models/runs/<dataset>/.

Default output directory: models/runs/<dataset-name>/
The PTQ and QAT compressed checkpoints are model.ptq.int8.json and model.qat.int8.json.
The picker returns local paths only; no file contents are uploaded. If zenity/kdialog is unavailable, enter a path in the terminal.
Folders can be passed with --data DIR.
HELP
    ;;
  *)
    echo "Unknown action: $ACTION (choose run, ptq, qat, benchmark, test, or help)" >&2
    exit 2
    ;;
esac
