# 🟦 Go · Quantization and Runtime

[← Project overview](../README.md)

The Go module prepares local input text, validates the shared checkpoint format, converts weights to INT8, orchestrates Java-based QAT, and evaluates model variants. Terminal panels use [Lip Gloss](https://github.com/charmbracelet/lipgloss).

## Requirements

- Go 1.22+
- Java 21+ and Maven for QAT orchestration
- Optional input extractors: Poppler (`pdftotext`), LibreOffice, and Tesseract

From this directory, run Go unit tests with:

```bash
go test ./...
```

From the repository root, `./start.sh test` runs both Go and Java suites.

## Input preparation

The tokenizer command accepts repeated `--input` arguments; each can be a file or a directory. Directories are traversed recursively and in sorted order. It writes normalized corpus text, vocabulary JSON, and token IDs to the output directory.

```bash
go run ./cmd/tokenize \
  --input ../data/raw/train.txt \
  --input ../data/raw/notes.md \
  --output ../data/tokenized
```

PDF text extraction requires `pdftotext`; Office extraction requires LibreOffice; image text recognition requires Tesseract. Scanned PDFs without a text layer are not OCRed in this version. Audio/video files require a same-name `.txt`, `.srt`, or `.vtt` transcript. This pipeline learns from extracted text, not pixels, audio, or video frames.

## PTQ · post-training quantization

PTQ validates an FP32 checkpoint and writes a signed INT8 checkpoint. It supports symmetric per-tensor and per-channel scales; symmetric quantization uses zero point 0.

```bash
go run ./cmd/quantize \
  --input ../models/exported/model.json \
  --output ../models/quantized/model.ptq.int8.json \
  --scheme per-channel
```

Optional `--calibration TEXT --vocab VOCAB.json` records activation ranges as checkpoint metadata. Calibration does not quantize activations.

## QAT · quantization-aware fine-tuning

The Go `qat` command validates the FP32 checkpoint, sends its weights through an internal bridge to the Java trainer, then converts the fine-tuned weights to a real INT8 checkpoint. Java simulates INT8-rounded weights during forward passes while keeping trainable FP32 master parameters.

```bash
cd ..
./start.sh qat \
  --input models/exported/model.json \
  --tokens data/tokenized/tokens.json \
  --vocab data/tokenized/vocab.json \
  --output models/quantized/model.qat.int8.json \
  --epochs 2 --windows-per-epoch 256
```

Use token IDs and vocabulary compatible with the checkpoint. `--windows-per-epoch 0` means all causal windows; a positive value caps work per epoch. QAT is an experiment, not a guarantee of better perplexity.

## Three-way benchmark

```bash
./start.sh benchmark \
  --fp32 models/exported/model.json \
  --ptq models/quantized/model.ptq.int8.json \
  --qat models/quantized/model.qat.int8.json \
  --eval data/eval/heldout.txt \
  --vocab data/tokenized/vocab.json \
  --report models/quantized/benchmark.json
```

The report includes file size, next-token perplexity, measured throughput, sampled Go heap, estimated stored-weight memory, and evaluation sample counts. Use held-out text for a quality comparison. The benchmark warns when the evaluation sample is small.

## Runtime and limits

INT8 embedding and matrix payloads remain packed in memory. Projection scales each value as it is used and accumulates in float64, so this is not an optimized integer matrix kernel. Lower throughput than FP32 is possible even when the INT8 checkpoint is much smaller.

Supported model input is the project’s version 1 `microllm` JSON export. Quantized exports use `microllm-int8` version 1. ONNX, GGUF, and arbitrary Hugging Face checkpoints are not supported.

See [`internal/quantization`](internal/quantization), [`cmd/benchmark`](cmd/benchmark), and the [architecture guide](../docs/ARCHITECTURE.md).
