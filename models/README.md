# 📦 Model artifacts

[← Project home](../README.md)

The launcher writes each complete run under `models/runs/<dataset-name>/` by default. Choose a different destination with `--output-dir`.

```text
models/runs/<run-name>/
├── tokenized/
│   ├── corpus.txt
│   ├── tokens.json
│   └── vocab.json
├── model.fp32.json
├── model.ptq.int8.json
├── model.qat.int8.json
└── benchmark.json
```

- `model.fp32.json` — Java-trained source model.
- `model.ptq.int8.json` — post-training quantized model.
- `model.qat.int8.json` — fake-quantized fine-tuned model exported as actual INT8.
- `benchmark.json` — comparison metrics and evaluation sample counts.

The Go and Java JSON files use the project-specific `microllm` version 1 and `microllm-int8` version 1 formats. Generated artifacts may be large and are intended to stay outside version control unless you choose to publish a model deliberately. Use the exact `tokenized/vocab.json` with the corresponding checkpoints.
