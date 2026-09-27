package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/nkosikhumalo/microllm/go/internal/tokenizer"
)

var (
	ingestTitle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205")).Padding(0, 1).Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63"))
	ingestGood  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("42"))
	ingestDim   = lipgloss.NewStyle().Faint(true)
	ingestWarn  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("214"))
)

func main() {
	var inputs []string
	flag.Func("input", "text/document/media file or directory; may be repeated", func(value string) error {
		inputs = append(inputs, value)
		return nil
	})
	output := flag.String("output", "../data/tokenized", "destination directory")
	flag.Parse()
	if len(inputs) == 0 {
		inputs = []string{"../data/raw/train.txt"}
	}
	fmt.Println(ingestTitle.Render("MiniLLM · Data preparation"))
	fmt.Println(ingestDim.Render("Extracting learnable text from the selected files"))
	var files []string
	seen := make(map[string]bool)
	for _, input := range inputs {
		found, err := collectInputs(input, *output)
		if err != nil {
			fatal("read dataset: %v", err)
		}
		for _, path := range found {
			absolute, err := filepath.Abs(path)
			if err != nil {
				fatal("resolve input %s: %v", path, err)
			}
			if !seen[absolute] {
				seen[absolute] = true
				files = append(files, path)
			}
		}
	}
	sort.Strings(files)
	var corpus strings.Builder
	used := 0
	for _, path := range files {
		text, err := extractFileText(path)
		if err != nil {
			fatal("ingest failed: %v", err)
		}
		if strings.TrimSpace(text) == "" {
			continue
		}
		if used > 0 {
			corpus.WriteString("\n\n")
		}
		corpus.WriteString("Source: ")
		corpus.WriteString(filepath.Base(path))
		corpus.WriteByte('\n')
		corpus.WriteString(text)
		used++
	}
	if used == 0 {
		fatal("no readable text was extracted from inputs %v", inputs)
	}
	vocab, err := tokenizer.Build(corpus.String())
	if err != nil {
		fatal("build vocab: %v", err)
	}
	ids, err := vocab.Encode(corpus.String())
	if err != nil {
		fatal("encode: %v", err)
	}
	if len(ids) < 2 {
		fatal("dataset must contain at least two tokens; found %d", len(ids))
	}
	if err := os.MkdirAll(*output, 0755); err != nil {
		fatal("create output directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*output, "corpus.txt"), []byte(corpus.String()), 0644); err != nil {
		fatal("save normalized corpus: %v", err)
	}
	if err := tokenizer.Save(filepath.Join(*output, "vocab.json"), vocab); err != nil {
		fatal("save vocab: %v", err)
	}
	encoded, err := json.Marshal(ids)
	if err != nil {
		fatal("marshal token IDs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*output, "tokens.json"), encoded, 0644); err != nil {
		fatal("save token IDs: %v", err)
	}
	fmt.Println(ingestGood.Render(fmt.Sprintf("Ready · %d files · %d tokens · %d vocabulary entries", used, len(ids), len(vocab.Tokens))))
	if len(ids) < 1000 {
		fmt.Println(ingestWarn.Render("Small training set. Add more varied text; extra epochs cannot replace missing examples."))
	}
	fmt.Println(ingestDim.Render("Prepared corpus: " + filepath.Join(*output, "corpus.txt")))
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
