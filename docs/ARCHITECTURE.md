# 🏗️ Architecture

[← Project home](../README.md)

## End-to-end pipeline

```text
local paths
   │
   ▼
Go input extraction + tokenizer ──► corpus.txt / vocab.json / tokens.json
                                                │
                                                ▼
                                        Java Transformer
                                                │
                                          FP32 JSON model
                                         ┌──────┴──────┐
                                         ▼             ▼
                                   Go PTQ INT8   Java QAT → Go INT8
                                         └──────┬──────┘
                                                ▼
                                    Go evaluation + JSON report
```

`start.sh` coordinates the full run. The file chooser returns local paths; data remains on the machine. The first selected input name determines the default `models/runs/<name>/` artifact directory.

## Input preparation

`go/cmd/tokenize` recursively gathers files, extracts supported text, builds a vocabulary and encodes token IDs. PDFs use Poppler `pdftotext`, Office formats use LibreOffice, and image text recognition uses Tesseract. Scanned PDFs without text layers currently return no extracted text. Audio/video need a same-name transcript file.

## Java training

The Java module builds a small causal Transformer and trains it on next-token windows. It exports a version 1 JSON checkpoint containing model configuration and row-major tensor weights. All overlapping windows are used when `--windows-per-epoch` is zero; a positive limit bounds windows per epoch.

## PTQ

Go validates the source checkpoint, then quantizes supported weight tensors to symmetric signed INT8. Per-tensor uses one scale for each tensor; per-channel uses scales along the first tensor dimension. Zero point is zero. Optional calibration writes observed activation ranges as metadata; activation quantization is not performed.

## QAT

Go validates the FP32 checkpoint and writes an internal binary bridge for Java. Java loads the initial parameters and continues training through its regular trainer, rounding weights to an INT8 grid in the forward pass and applying the straight-through estimator for gradients. Java exports updated FP32 weights; Go converts those to a `microllm-int8` checkpoint and records QAT metadata.

## Inference and benchmark

The Go runtime retains quantized embedding and matrix payloads as INT8 and scales weight values during projection. Activations and accumulators use float64. This saves stored weight memory but is not a fused integer kernel, so throughput may be lower than FP32.

The benchmark evaluates the FP32, PTQ, and QAT checkpoints on the same vocabulary and text. It writes file size, next-token perplexity, throughput, peak heap sample, stored-weight estimate, and evaluation counts. A training corpus is suitable for a pipeline check; held-out data is needed for quality claims.

## Checkpoint formats

- Source model: version 1 JSON, `format: "microllm"`.
- Quantized model: version 1 JSON, `format: "microllm-int8"`.
- QAT hand-off: temporary internal binary bridge; final checkpoints remain JSON.
