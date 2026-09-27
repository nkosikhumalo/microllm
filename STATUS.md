# 📍 Project Status

[← Project home](README.md)

## Implemented

- Local file/folder selection and repeated input paths in the end-to-end launcher.
- Go text extraction for supported text formats, PDFs with a text layer, Office documents, and Tesseract-readable images.
- Java small Transformer training and FP32 JSON export.
- Go PTQ using symmetric signed INT8 with per-tensor or per-channel weight scales.
- QAT continuation from a compatible FP32 checkpoint using Java fake-quantized weights and straight-through gradients.
- Real INT8 export for both PTQ and QAT checkpoints.
- Go inference with packed INT8 embedding/matrix payloads.
- Three-way benchmark with size, next-token perplexity, throughput, heap, stored-weight estimate, and sample-count reporting.
- Go and Java test suites through `./start.sh test`.

## Current limits

- Input is text-first. OCR reads writing from images; it does not interpret image scenes. Audio/video need sidecar `.txt`, `.srt`, or `.vtt` transcripts.
- PDF extraction uses `pdftotext`; scanned PDFs without a text layer do not produce text. PDF page OCR is not currently included.
- Image OCR quality depends on Tesseract, language data, and image quality.
- Checkpoint input uses the project-specific version 1 JSON format. ONNX, GGUF, and arbitrary Hugging Face checkpoints are not supported.
- Fake quantization applies to weights; activation fake quantization is not implemented. Calibration records activation ranges only.
- Inference keeps INT8 weights packed but scales values during projection and uses float64 activations/accumulators. A fused integer kernel and inference speedup are not claimed.
- Full-window training can be expensive on large corpora. `--windows-per-epoch` caps training work; zero means use every available sliding window.
- QAT may improve or reduce quality; benchmark it on a substantial held-out corpus rather than assuming an accuracy gain.
