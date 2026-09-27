package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/nkosikhumalo/microllm/go/internal/inference"
	"github.com/nkosikhumalo/microllm/go/internal/loader"
	"github.com/nkosikhumalo/microllm/go/internal/quantization"
	"github.com/nkosikhumalo/microllm/go/internal/tokenizer"
)

var (
	benchmarkTitleStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205")).Padding(0, 1).Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63"))
	benchmarkMutedStyle = lipgloss.NewStyle().Faint(true)
	benchmarkPanelStyle = lipgloss.NewStyle().Padding(0, 1).Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("39"))
)

type Metrics struct {
	SizeBytes       int64   `json:"size_bytes"`
	Tokens          int     `json:"tokens"`
	TokensPerSecond float64 `json:"tokens_per_second"`
	WeightMemoryMB  float64 `json:"stored_weight_memory_estimate_mb"`
	PeakHeapMB      float64 `json:"peak_go_heap_mb"`
	Perplexity      float64 `json:"perplexity"`
	EvalLines       int     `json:"eval_lines"`
}

type Report struct {
	FP32                 Metrics `json:"fp32"`
	PTQ                  Metrics `json:"ptq_int8"`
	QAT                  Metrics `json:"qat_int8"`
	FP32ToPTQSizeRatio   float64 `json:"fp32_to_ptq_size_ratio"`
	FP32ToQATSizeRatio   float64 `json:"fp32_to_qat_size_ratio"`
	PTQPerplexityDeltaPC float64 `json:"ptq_perplexity_delta_percent"`
	QATPerplexityDeltaPC float64 `json:"qat_perplexity_delta_percent"`
	EvalLines            int     `json:"eval_lines"`
}

func main() {
	fpPath := flag.String("fp32", "", "original floating-point model.json")
	ptqPath := flag.String("ptq", "", "post-training INT8 checkpoint")
	flag.StringVar(ptqPath, "int8", "", "deprecated alias for --ptq")
	qatPath := flag.String("qat", "", "QAT fine-tuned INT8 checkpoint")
	evalPath := flag.String("eval", "", "held-out text file, one sequence per line")
	reportPath := flag.String("report", "", "optional JSON report path")
	vocabPath := flag.String("vocab", "../data/tokenized/vocab.json", "tokenizer vocabulary JSON")
	maxLineBytes := flag.Int("max-line-bytes", 64*1024*1024, "maximum bytes in one evaluation line")
	flag.Parse()
	fmt.Println(benchmarkTitleStyle.Render("MiniLLM Quantization Engine · Three-way benchmark"))
	if *fpPath == "" || *ptqPath == "" || *qatPath == "" || *evalPath == "" {
		fmt.Fprintln(os.Stderr, "usage: go run ./cmd/benchmark --fp32 model.json --ptq model.ptq.int8.json --qat model.qat.int8.json --eval heldout.txt")
		os.Exit(2)
	}
	if *maxLineBytes < 1 {
		fatal(fmt.Errorf("max-line-bytes must be positive"))
	}
	fmt.Println(benchmarkMutedStyle.Render("Loading evaluation vocabulary..."))
	vocab, err := tokenizer.Load(*vocabPath)
	if err != nil {
		fatal(err)
	}

	fmt.Println(benchmarkMutedStyle.Render("Evaluating FP32 checkpoint..."))
	fp32, err := loader.Load(*fpPath)
	if err != nil {
		fatal(fmt.Errorf("load FP32 checkpoint: %w", err))
	}
	modelConfig := fp32.Config
	if len(vocab.Tokens) != modelConfig.VocabSize {
		fatal(fmt.Errorf("vocabulary size does not match model"))
	}
	runtime.GC()
	fpMetrics, err := evaluateFile(fp32, *evalPath, vocab, *maxLineBytes)
	if err != nil {
		fatal(fmt.Errorf("FP32 evaluation: %w", err))
	}
	fp32 = nil
	runtime.GC()

	fmt.Println(benchmarkMutedStyle.Render("Evaluating PTQ checkpoint..."))
	ptqCheckpoint, err := quantization.Load(*ptqPath)
	if err != nil {
		fatal(fmt.Errorf("load PTQ checkpoint: %w", err))
	}
	if ptqCheckpoint.Method != "ptq" {
		fatal(fmt.Errorf("checkpoint %s is marked %q, expected PTQ", *ptqPath, ptqCheckpoint.Method))
	}
	ptq, err := ptqCheckpoint.Dequantize()
	if err != nil {
		fatal(fmt.Errorf("validate PTQ checkpoint: %w", err))
	}
	ptqWeights, err := ptqCheckpoint.InferenceWeights()
	if err != nil {
		fatal(fmt.Errorf("decode PTQ INT8 matrices: %w", err))
	}
	if ptq.Config != modelConfig {
		fatal(fmt.Errorf("FP32 and PTQ model configurations must match"))
	}
	ptqCheckpoint = nil
	runtime.GC()
	ptqMetrics, err := evaluateFile(ptq, *evalPath, vocab, *maxLineBytes, ptqWeights)
	if err != nil {
		fatal(fmt.Errorf("PTQ evaluation: %w", err))
	}
	ptq = nil
	ptqWeights = nil
	runtime.GC()

	fmt.Println(benchmarkMutedStyle.Render("Evaluating QAT checkpoint..."))
	qatCheckpoint, err := quantization.Load(*qatPath)
	if err != nil {
		fatal(fmt.Errorf("load QAT checkpoint: %w", err))
	}
	if qatCheckpoint.Method != "qat" || qatCheckpoint.QATEpochs < 1 {
		fatal(fmt.Errorf("checkpoint %s is not marked as a QAT result", *qatPath))
	}
	qat, err := qatCheckpoint.Dequantize()
	if err != nil {
		fatal(fmt.Errorf("validate QAT checkpoint: %w", err))
	}
	qatWeights, err := qatCheckpoint.InferenceWeights()
	if err != nil {
		fatal(fmt.Errorf("decode QAT INT8 matrices: %w", err))
	}
	if qat.Config != modelConfig {
		fatal(fmt.Errorf("FP32 and QAT model configurations must match"))
	}
	qatCheckpoint = nil
	runtime.GC()
	qatMetrics, err := evaluateFile(qat, *evalPath, vocab, *maxLineBytes, qatWeights)
	if err != nil {
		fatal(fmt.Errorf("QAT evaluation: %w", err))
	}
	qat = nil
	qatWeights = nil
	runtime.GC()
	fpStat, err := os.Stat(*fpPath)
	if err != nil {
		fatal(err)
	}
	ptqStat, err := os.Stat(*ptqPath)
	if err != nil {
		fatal(err)
	}
	qatStat, err := os.Stat(*qatPath)
	if err != nil {
		fatal(err)
	}
	fpMetrics.SizeBytes = fpStat.Size()
	ptqMetrics.SizeBytes = ptqStat.Size()
	qatMetrics.SizeBytes = qatStat.Size()
	report := Report{
		FP32: fpMetrics, PTQ: ptqMetrics, QAT: qatMetrics,
		FP32ToPTQSizeRatio:   float64(fpMetrics.SizeBytes) / float64(ptqMetrics.SizeBytes),
		FP32ToQATSizeRatio:   float64(fpMetrics.SizeBytes) / float64(qatMetrics.SizeBytes),
		PTQPerplexityDeltaPC: 100 * (ptqMetrics.Perplexity/fpMetrics.Perplexity - 1),
		QATPerplexityDeltaPC: 100 * (qatMetrics.Perplexity/fpMetrics.Perplexity - 1),
		EvalLines:            fpMetrics.EvalLines,
	}
	comparison := fmt.Sprintf(
		"Metric                 FP32             PTQ-INT8         QAT-INT8\n"+
			"Checkpoint size        %-16s %-16s %-16s\n"+
			"Next-token perplexity  %-16.4f %-16.4f (%+.2f%%)  %-16.4f (%+.2f%%)\n"+
			"Throughput             %-16.2f %-16.2f %-16.2f tok/s\n"+
			"Peak Go heap           %-16.2f %-16.2f %-16.2f MB\n"+
			"Stored weights (est.)  %-16.2f %-16.2f %-16.2f MB",
		size(fpMetrics.SizeBytes), size(ptqMetrics.SizeBytes), size(qatMetrics.SizeBytes),
		fpMetrics.Perplexity, ptqMetrics.Perplexity, report.PTQPerplexityDeltaPC, qatMetrics.Perplexity, report.QATPerplexityDeltaPC,
		fpMetrics.TokensPerSecond, ptqMetrics.TokensPerSecond, qatMetrics.TokensPerSecond,
		fpMetrics.PeakHeapMB, ptqMetrics.PeakHeapMB, qatMetrics.PeakHeapMB,
		fpMetrics.WeightMemoryMB, ptqMetrics.WeightMemoryMB, qatMetrics.WeightMemoryMB,
	)
	fmt.Println(benchmarkPanelStyle.Render(comparison))
	fmt.Println(benchmarkMutedStyle.Render(fmt.Sprintf("Scored %d sequences and %d next-token predictions. Quantized matrices stay INT8 in memory; activations and accumulators use float64.", report.EvalLines, fpMetrics.Tokens)))
	if fpMetrics.Tokens < 1000 || report.EvalLines < 20 {
		fmt.Println(lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("214")).Render("Small evaluation sample: treat perplexity as a pipeline check, not a quality claim. Use a larger held-out corpus."))
	}
	if *reportPath != "" {
		encoded, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(*reportPath), 0755); err != nil {
			fatal(err)
		}
		if err := os.WriteFile(*reportPath, encoded, 0644); err != nil {
			fatal(err)
		}
		fmt.Printf("Report written to %s\n", *reportPath)
	}
}

func evaluateFile(modelExport *loader.Export, path string, vocab *tokenizer.Vocab, maxLineBytes int, quantized ...map[string]inference.QuantizedMatrix) (Metrics, error) {
	weightMemoryMB := estimatedWeightMemoryMB(modelExport, nil)
	var model *inference.Model
	var err error
	if len(quantized) > 0 && quantized[0] != nil {
		weightMemoryMB = estimatedWeightMemoryMB(modelExport, quantized[0])
		model, err = inference.NewQuantized(modelExport, quantized[0])
	} else {
		model, err = inference.New(modelExport)
	}
	if err != nil {
		return Metrics{}, err
	}
	runtime.GC()
	file, err := os.Open(path)
	if err != nil {
		return Metrics{}, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), maxLineBytes)
	start := time.Now()
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	peakHeap := memory.HeapAlloc
	tokens, lines := 0, 0
	loss := 0.0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		lines++
		ids, err := vocab.Encode(line)
		if err != nil {
			return Metrics{}, fmt.Errorf("evaluation line %d: %w", lines, err)
		}
		if len(ids) < 2 {
			continue
		}
		session := model.NewSession()
		logits, err := session.Prefill(ids[:1])
		if err != nil {
			return Metrics{}, err
		}
		for position := 1; position < len(ids); position++ {
			maximum := logits[0]
			for _, value := range logits {
				maximum = math.Max(maximum, value)
			}
			exponentialSum := 0.0
			for _, value := range logits {
				exponentialSum += math.Exp(value - maximum)
			}
			loss += maximum + math.Log(exponentialSum) - logits[ids[position]]
			tokens++
			if tokens%16 == 0 {
				runtime.ReadMemStats(&memory)
				if memory.HeapAlloc > peakHeap {
					peakHeap = memory.HeapAlloc
				}
			}
			if position+1 < len(ids) {
				if session.Len() >= modelExport.Config.MaxSeqLen {
					begin := position + 1 - modelExport.Config.MaxSeqLen
					logits, err = session.Prefill(ids[begin : position+1])
				} else {
					logits, err = session.Step(ids[position])
				}
				if err != nil {
					return Metrics{}, err
				}
			}
		}
		if lines%100 == 0 {
			fmt.Printf("  scored %d sequences (%d tokens)\n", lines, tokens)
		}
	}
	if err := scanner.Err(); err != nil {
		return Metrics{}, fmt.Errorf("read evaluation file: %w", err)
	}
	if lines == 0 || tokens == 0 {
		return Metrics{}, fmt.Errorf("evaluation text contains no token pairs to score")
	}
	runtime.ReadMemStats(&memory)
	if memory.HeapAlloc > peakHeap {
		peakHeap = memory.HeapAlloc
	}
	duration := time.Since(start).Seconds()
	speed := 0.0
	if duration > 0 {
		speed = float64(tokens) / duration
	}
	return Metrics{
		Tokens: tokens, EvalLines: lines, TokensPerSecond: speed,
		WeightMemoryMB: weightMemoryMB,
		PeakHeapMB:     float64(peakHeap) / (1 << 20),
		Perplexity:     math.Exp(loss / float64(tokens)),
	}, nil
}

func estimatedWeightMemoryMB(modelExport *loader.Export, quantized map[string]inference.QuantizedMatrix) float64 {
	if len(quantized) == 0 {
		return float64(weightCount(modelExport)*8) / (1 << 20)
	}
	bytes := int64((weightCount(modelExport) - quantizedElementCount(quantized)) * 8)
	for _, matrix := range quantized {
		bytes += int64(len(matrix.Data))
		bytes += int64(len(matrix.Scales)+len(matrix.ZeroPoints)) * 8
	}
	return float64(bytes) / (1 << 20)
}

func quantizedElementCount(weights map[string]inference.QuantizedMatrix) int {
	total := 0
	for _, matrix := range weights {
		total += len(matrix.Data)
	}
	return total
}

func size(bytes int64) string {
	if bytes >= 1<<20 {
		return fmt.Sprintf("%.2f MB", float64(bytes)/(1<<20))
	}
	return fmt.Sprintf("%.2f KB", float64(bytes)/1024)
}

func fatal(err error) { fmt.Fprintln(os.Stderr, err); os.Exit(1) }

func weightCount(modelExport *loader.Export) int {
	weights := modelExport.Weights
	count := len(weights.TokenEmbedding) + len(weights.PositionEmbedding) + len(weights.FinalNorm) + len(weights.Output)
	for _, layer := range weights.Layers {
		count += len(layer.AttnNorm) + len(layer.Q) + len(layer.K) + len(layer.V) + len(layer.O) + len(layer.FFNNorm) + len(layer.FFNIn) + len(layer.FFNOut)
	}
	return count
}
