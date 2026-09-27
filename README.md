<p align="center">
  <img src="image/title-banner.svg" alt="MiniLLM Quantization Engine" width="100%">
</p>

<p align="center">
  <img alt="Go 1.22+" src="https://img.shields.io/badge/Go-1.22%2B-00ADD8?logo=go&logoColor=white">
  <img alt="Java 21+" src="https://img.shields.io/badge/Java-21%2B-E76F00?logo=openjdk&logoColor=white">
  <img alt="Local-first" src="https://img.shields.io/badge/Run-fully%20local-22C55E">
  <img alt="Quantization" src="https://img.shields.io/badge/Weights-FP32%20%7C%20INT8-8B5CF6">
</p>

<p align="center"><strong>Train a small Transformer. Compress its weights. Measure the tradeoffs.</strong></p>

MiniLLM Quantization Engine is a local learning project that connects a small Transformer training pipeline with two INT8 compression paths. It produces real checkpoints and compares their storage size and next-token quality on the same evaluation text.

> [!NOTE]
> This is an educational, small-model system—not a production LLM or a general-purpose document understanding model. Results depend on the amount and quality of your text. QAT may not outperform PTQ; the benchmark measures what happened for that run.

## ✨ What it does

| Stage | What happens |
| --- | --- |
| **Prepare** | Reads local text and supported documents; tokenizes the extracted text. |
| **Train** | Java trains a small causal Transformer and exports an FP32 checkpoint. |
| **PTQ** | Go converts trained weights to signed INT8 without further training. |
| **QAT** | Java continues training with fake-quantized weights; Go writes the resulting weights as an INT8 checkpoint. |
| **Compare** | Go evaluates FP32, PTQ, and QAT and prints a comparison plus a JSON report. |

<p align="center"><img src="image/pipeline-strip.svg" alt="MiniLLM quantization workflow" width="100%"></p>

## 🚀 Quick start

### Requirements

- Go 1.22 or newer
- Java 21 or newer
- Maven
- For document extraction: Poppler (`pdftotext`) for PDFs and LibreOffice for Office formats
- For image OCR: Tesseract
- Optional desktop chooser: Zenity (multi-select) or KDialog (single-select)

On Ubuntu/Debian, the optional extraction tools can be installed with:

```bash
sudo apt install poppler-utils libreoffice tesseract-ocr zenity
```

Start the guided local workflow:

```bash
./start.sh
```

The launcher explains the workflow, then opens a local file chooser when Zenity or KDialog is available. Select several files with Ctrl-click in Zenity. If no chooser is installed, it asks for a path in the terminal. Only local paths are passed to the pipeline; no upload or account is involved.

Or pass files and folders directly:

```bash
./start.sh run \
  --data ./data/raw/train.txt \
  --data ./data/raw/notes.md \
  --eval ./data/eval/heldout.txt \
  --epochs 100 \
  --qat-epochs 2 \
  --windows-per-epoch 256
```

`--data` may be repeated. Directories are scanned recursively in sorted order. By default, output goes to `models/runs/<first-input-name>/`; use `--output-dir PATH` to choose a different location. The pipeline saves normalized text, vocabulary, token IDs, all three model checkpoints, and `benchmark.json` there.

### Supported inputs

- UTF-8 text and source-like files (including `.txt`, `.md`, and HTML)
- PDF files with an extractable text layer (`pdftotext`)
- Office documents supported by LibreOffice
- Images supported by Tesseract OCR (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`)
- Audio/video only when a same-name `.txt`, `.srt`, or `.vtt` transcript exists beside the media file

Scanned/image-only PDFs do not have a text layer; this version does not OCR PDF pages automatically. Image OCR extracts recognized writing only—it does not understand objects, scenes, or image meaning. Low resolution, rotation, small print, contrast, and OCR language settings affect the result. If a selected file yields no text, check the extractor dependency and inspect the generated `tokenized/corpus.txt` before trusting the run.

## 🎛️ CLI commands

```bash
./start.sh help
./start.sh test
```

Quantize a compatible FP32 checkpoint:

```bash
./start.sh ptq --input models/exported/model.json \
  --output models/quantized/model.ptq.int8.json \
  --scheme per-channel
```

Run QAT from a compatible FP32 checkpoint and the matching tokenized training data:

```bash
./start.sh qat --input models/exported/model.json \
  --tokens data/tokenized/tokens.json \
  --vocab data/tokenized/vocab.json \
  --output models/quantized/model.qat.int8.json \
  --epochs 2 --windows-per-epoch 256
```

Benchmark all three versions on held-out text:

```bash
./start.sh benchmark \
  --fp32 models/exported/model.json \
  --ptq models/quantized/model.ptq.int8.json \
  --qat models/quantized/model.qat.int8.json \
  --eval data/eval/heldout.txt \
  --vocab data/tokenized/vocab.json \
  --report models/quantized/benchmark.json
```

Run Go and Java test suites:

```bash
./start.sh test
```

## 📊 What the benchmark reports

- Checkpoint file size
- Next-token perplexity on the supplied evaluation text
- Measured token throughput
- Sampled peak Go heap
- Estimated stored-weight memory

Give `--eval` a held-out corpus whose text is represented by the model vocabulary. Without `--eval`, the end-to-end workflow evaluates on the training corpus, which checks pipeline operation but does not measure generalization. The benchmark warns on small evaluation samples; a handful of tokens cannot support a reliable quality claim.

INT8 matrix and embedding payloads remain packed during Go inference. This implementation scales weight values during projection and still uses float64 activations and accumulators; it is not a fused integer kernel and does not promise faster inference. INT8 primarily demonstrates reduced checkpoint and stored-weight size here.

## 🧠 Training time and corpus size

The trainer uses sliding causal windows. `--windows-per-epoch 0` (the default) processes all available windows each epoch. Long corpora can therefore take a while, and current progress is printed periodically rather than for every window. Set a positive limit such as `256` to cap the number of training windows per epoch. More epochs cannot replace missing examples; use varied training text and a separate held-out evaluation file.

## 🧩 How the code is split

- **Go** — input preparation, checkpoint validation, PTQ, QAT orchestration, INT8 inference, and benchmark CLI.
- **Java** — Transformer implementation, optimizer, training loop, FP32 checkpoint export, and QAT fine-tuning forward path.

More detail: [Go guide](go/README.md) · [Java guide](java/README.md) · [Architecture](docs/ARCHITECTURE.md) · [Project explanation](PROJECT_EXPLANATION.md) · [Current status](STATUS.md) · [Checkpoint artifacts](models/README.md)

## 🎨 Project visuals

<p align="center">
  <img src="image/tagline.svg" alt="Compress weights. Measure quality. Run locally." width="100%">
</p>
