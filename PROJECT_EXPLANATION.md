# 🌱 MiniLLM Quantization Engine

[← Project home](README.md)

## The idea

This project takes a small Transformer checkpoint and asks a measurable question: how much storage can INT8 quantization save, and how close does next-token prediction stay to the original FP32 model?

The system combines a from-scratch Java training implementation with Go tooling for input preparation, quantization, inference, and benchmarking.

## The three checkpoints

- **FP32** — weights exported by the Java trainer.
- **PTQ INT8** — the FP32 checkpoint quantized after training; no fine-tuning is performed.
- **QAT INT8** — the FP32 checkpoint is fine-tuned in Java with fake-quantized weights in the forward pass, then Go writes the resulting weights as actual INT8 values.

The benchmark measures all three using the same evaluation corpus. It reports file size, next-token perplexity, throughput, heap use, and stored-weight estimate. QAT is not assumed to win; results depend on training and evaluation data.

## Flow

```text
local text / supported document
          │
          ▼
Go extraction and tokenizer ──► token IDs + vocabulary
                                      │
                                      ▼
                             Java Transformer training
                                      │
                                  FP32 model
                                  ┌───┴───┐
                                  ▼       ▼
                                 PTQ     QAT
                                  └───┬───┘
                                      ▼
                           Go three-way benchmark
```

## What this project is—and is not

It is a local engineering and learning project for experimenting with small language models and weight compression. It learns from text extracted from supported local files. Tesseract can recognize text in images, but the model does not understand visual scenes. Audio and video require text transcripts. Scanned PDFs without embedded text are not OCRed automatically in the current implementation.

It is not a general-purpose assistant, a production LLM, or a universal model converter. Checkpoints use the project’s own JSON format. The INT8 inference path stores weights compactly but uses float64 activations and accumulators, so compressed size does not imply faster inference.
