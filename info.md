# MiniLLM Quantization Engine · Project Notes

[Project README](README.md) · [Architecture](docs/ARCHITECTURE.md) · [Status](STATUS.md)

A local Go + Java project for training a small causal Transformer, compressing its trained weights to INT8 using PTQ and QAT, then measuring checkpoint size and next-token perplexity.

- Java trains the FP32 source checkpoint and runs QAT fine-tuning with fake-quantized weights.
- Go prepares input text, validates checkpoints, writes PTQ/QAT INT8 checkpoints, and benchmarks all three variants.
- INT8 weights remain packed in Go inference, while activations and accumulators are float64.
- PDF extraction reads embedded text; scanned PDFs are not OCRed. Tesseract recognizes text in images but does not understand image content.
- The project-specific version 1 JSON model format is supported; ONNX, GGUF, and arbitrary Hugging Face checkpoints are not.

Start with `./start.sh` for the local picker or see `./start.sh help` for commands.
