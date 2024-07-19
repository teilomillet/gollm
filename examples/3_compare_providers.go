// File: examples/3_compare_providers.go

package main

import (
	"context"
	"log"

	"github.com/teilomillet/goal/llm"
)

func main() {
	configs, err := llm.LoadConfigs()
	if err != nil {
		log.Fatalf("Failed to load configs: %v", err)
	}

	if len(configs) < 2 {
		log.Fatalf("Please provide at least two different provider configurations in ~/.goal/configs/")
	}

	prompt := "Explain the concept of artificial intelligence in one sentence."
	results := llm.CompareProviders(context.Background(), prompt, configs["anthropic.yaml"], configs["openai.yaml"])

	llm.PrintComparisonResults(results)
}
