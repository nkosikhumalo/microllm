# ☕ Java · MiniLLM Training Engine

[← Project overview](../README.md)

This Java 21 module implements the small causal Transformer, its training loop, and the FP32 checkpoint export consumed by the Go Quantization Engine. It also provides the fake-quantized forward path used during QAT. Maven dependencies are limited to JUnit for tests; model and optimizer math are implemented in the project.

## Build and tests

```bash
mvn test
mvn package
```

The runnable jar is written to `target/microllm-train-0.1.0-SNAPSHOT.jar`.

## Train and export an FP32 checkpoint

Prepare a corpus and token IDs with Go first, then run the Java trainer:

```bash
cd ..
./scripts/tokenize.sh
./scripts/train.sh --epochs 100 --val-fraction 0.1 --patience 10
```

The scripts use `data/raw/train.txt`, `data/tokenized/`, and `models/exported/model.json` by default. To use a selected local file or directory, run the full workflow from the repository root with `./start.sh run --data PATH`.

The trainer can also be called directly after building:

```bash
cd java
java -jar target/microllm-train-0.1.0-SNAPSHOT.jar \
  --vocab ../data/tokenized/vocab.json \
  --tokens ../data/tokenized/tokens.json \
  --output ../models/exported/model.json \
  --epochs 100 --learning-rate 0.001 \
  --val-fraction 0.1 --patience 50 \
  --windows-per-epoch 256
```

`--windows-per-epoch 0` (default) trains on all available sliding windows every epoch. Larger corpora can make each epoch expensive. Use a positive cap to bound the work; training logs appear periodically. The trainer uses causal next-token prediction and splits the tail of the token stream for validation when the dataset is large enough.

## QAT training path

The Go QAT command imports a compatible FP32 checkpoint with `--init-weights`, enables fake quantization, and invokes this same Java trainer. During the forward pass, trainable weights are rounded to their INT8 grid and represented as floating point; a straight-through estimator lets gradients update the FP32 master parameters. Go then exports those learned parameters as actual INT8 values plus scale metadata.

Fake quantization currently covers weights; activation fake quantization is not implemented. QAT can improve or reduce evaluation quality depending on data and settings.

## Checkpoint compatibility

The Java exporter writes the project-specific version 1 JSON model format (`format: "microllm"`) with model configuration and row-major tensors. Go validates this format before quantization. The legacy `microllm` identifier is retained for compatibility; this is not a general ONNX, GGUF, or Hugging Face model importer.

See the [architecture guide](../docs/ARCHITECTURE.md) and [Go guide](../go/README.md).
