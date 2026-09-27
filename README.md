<p align="center">
  <img src="image/title-banner.svg" alt="MiniLLM Quantization Engine" width="100%">
</p>

<p align="center">
  <img alt="Go 1.22 or newer" src="https://img.shields.io/badge/Go-1.22%2B-00ADD8?logo=go&logoColor=white">
  <img alt="Java 21 or newer" src="https://img.shields.io/badge/Java-21%2B-E76F00?logo=openjdk&logoColor=white">
  <img alt="Runs on your computer" src="https://img.shields.io/badge/Run-fully%20local-22C55E">
  <img alt="FP32 and INT8 model weights" src="https://img.shields.io/badge/Weights-FP32%20%7C%20INT8-8B5CF6">
</p>

<p align="center"><strong>Train a small language model, make smaller copies, and compare the results.</strong></p>

MiniLLM is a learning project. It reads text from your computer and trains a small Transformer. A Transformer is a type of model that learns patterns in text and predicts what token (a piece of a word or a whole word) may come next. The project then makes two compressed versions of the model and compares their size and prediction quality.

> [!NOTE]
> This project is for learning and small experiments. It is not a production AI system or a general-purpose document reader. Results depend on your training text. The compressed versions may not predict text as well as the original.

## What the project does

| Step | In simple terms |
| --- | --- |
| Prepare text | Reads local files, extracts text where supported, and turns text into tokens the model can use. |
| Train | Java trains the model and saves its weights as an FP32 checkpoint. A checkpoint is a file that stores a model so it can be used later. FP32 stores weights as 32-bit numbers. |
| Compress with PTQ | Go converts the trained weights to INT8 numbers without more training. PTQ means post-training quantization. |
| Tune with QAT | Java trains another copy while simulating INT8 weights. Go saves that copy in INT8 format. QAT means quantization-aware training. |
| Compare | Go checks the file size and next-token prediction quality of the original and both compressed copies. |

INT8 uses 8-bit numbers, so model weight files can be smaller. Smaller files do not always mean faster predictions. This project measures the results instead of assuming them.

<p align="center"><img src="image/pipeline-strip.svg" alt="MiniLLM workflow: prepare text, train a model, compress it, and compare results" width="100%"></p>

## Get started

### Requirements

- Go 1.22 or newer
- Java 21 or newer
- Maven
- For PDF text: Poppler (`pdftotext`)
- For Office files: LibreOffice
- For text in images: Tesseract
- Optional file chooser: Zenity or KDialog

On Ubuntu or Debian, install the optional document and image tools with:

```bash
sudo apt install poppler-utils libreoffice tesseract-ocr zenity
```

Start the guided workflow:

```bash
./start.sh
```

The launcher asks you to choose local files. If Zenity or KDialog is installed, it opens a file chooser. Otherwise, enter a file path in the terminal. Your files stay on your computer; the project does not upload them.

You can also give it file paths directly:

```bash
./start.sh run \
  --data ./data/raw/train.txt \
  --data ./data/raw/notes.md \
  --eval ./data/eval/heldout.txt \
  --epochs 100 \
  --qat-epochs 2 \
  --windows-per-epoch 256
```

Use `--data` more than once to include multiple files. A folder is read recursively. By default, results are saved under `models/runs/<first-input-name>/`. Set `--output-dir PATH` to choose another folder. The output includes prepared text, the vocabulary, token IDs, three model checkpoints, and `benchmark.json`.

### Files you can use

- Text and code files, including `.txt`, `.md`, and HTML
- PDFs that contain selectable text (requires `pdftotext`)
- Office documents supported by LibreOffice
- Images with text (requires Tesseract): `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp`
- Audio or video files with a same-name `.txt`, `.srt`, or `.vtt` transcript beside them

Scanned PDFs are not read in this version. Tesseract can recognize writing in images, but it cannot describe objects or scenes. Image quality, rotation, font size, and contrast can affect recognition. If a file seems to be skipped, check that its required tool is installed and review `tokenized/corpus.txt` in the output folder.

## Commands

See available commands and options:

```bash
./start.sh help
```

Make an INT8 copy of an existing FP32 checkpoint with PTQ:

```bash
./start.sh ptq --input models/exported/model.json \
  --output models/quantized/model.ptq.int8.json \
  --scheme per-channel
```

Run QAT using the original checkpoint and its matching tokenized training data:

```bash
./start.sh qat --input models/exported/model.json \
  --tokens data/tokenized/tokens.json \
  --vocab data/tokenized/vocab.json \
  --output models/quantized/model.qat.int8.json \
  --epochs 2 --windows-per-epoch 256
```

Compare the original and compressed checkpoints on text that was kept out of training:

```bash
./start.sh benchmark \
  --fp32 models/exported/model.json \
  --ptq models/quantized/model.ptq.int8.json \
  --qat models/quantized/model.qat.int8.json \
  --eval data/eval/heldout.txt \
  --vocab data/tokenized/vocab.json \
  --report models/quantized/benchmark.json
```

Run the Go and Java test suites with:

```bash
./start.sh test
```

## Reading the benchmark

The benchmark reports:

- Checkpoint file size
- Perplexity, a measure of how well the model predicts the next token (lower is better)
- Measured tokens processed per second
- Sampled peak Go memory use
- Estimated memory used to store model weights

Use a separate, held-out text file with `--eval` for a useful quality comparison. If you do not provide one, the workflow checks the training text instead. That confirms the pipeline ran, but does not show how well the model handles new text. Very small evaluation files make results less reliable.

The Go program keeps INT8 weights packed in memory, but converts values while making predictions. Its activations and calculation totals use 64-bit numbers. It does not use a specialized, fast INT8 calculation engine, so an INT8 model may be slower even though its weight file is smaller.

## Training time and text size

Training reads the text in overlapping sections called windows. By default, each training round uses all available windows. Large text collections can take a long time. Set `--windows-per-epoch` to a positive number, such as `256`, to limit the windows used in each round. A value of `0` means use all windows.

More training rounds cannot replace missing examples. Use varied text for training and a separate file for evaluation.

## How the code is organized

- **Go** prepares text, checks checkpoints, creates the PTQ model, coordinates QAT, runs predictions, and compares results.
- **Java** implements and trains the Transformer, saves the FP32 checkpoint, and runs the QAT training step.

More detail: [Go guide](go/README.md) · [Java guide](java/README.md) · [Architecture](docs/ARCHITECTURE.md) · [Project explanation](PROJECT_EXPLANATION.md) · [Current status](STATUS.md) · [Checkpoint files](models/README.md)

## Project artwork

<p align="center">
  <img src="image/tagline.svg" alt="Compress weights. Measure quality. Run locally." width="100%">
</p>
