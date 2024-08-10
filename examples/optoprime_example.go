// File: examples/optoprime_example.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/teilomillet/gollm"
)

type OptoPrimeResponse struct {
	Reasoning  string `json:"reasoning"`
	Suggestion string `json:"suggestion"`
	Prompt     string `json:"prompt"`
}

func main() {

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetMaxTokens(1024),
		gollm.SetTemperature(0.7),
		gollm.SetAPIKey(apiKey),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	optoPrime, err := gollm.NewOptoPrime(llm)
	if err != nil {
		log.Fatalf("Failed to create OptoPrime: %v", err)
	}

	// Use filepath.Join to create a path that works on any operating system
	articlePath := filepath.Join("testdata", "test_article.txt")
	article, err := os.ReadFile(articlePath)
	if err != nil {
		log.Fatalf("Failed to read test article: %v", err)
	}

	initialPrompt := "Summarize the following news article in 3 sentences:\n\n" + string(article)
	ctx := context.Background()
	iterations := 3

	var optimizedPrompt string
	for i := 0; i < iterations; i++ {
		updatedPrompt, err := optoPrime.OptimizePrompt(ctx, initialPrompt, 1)
		if err != nil {
			log.Printf("Error in iteration %d: %v", i, err)
			continue
		}

		log.Printf("Updated prompt from OptoPrime (first 100 chars): %s", safeSubstring(updatedPrompt, 100))
		log.Printf("Updated prompt length: %d", len(updatedPrompt))

		optimizedPrompt = updatedPrompt
		log.Printf("Optimized prompt from iteration %d: %s", i, safeSubstring(optimizedPrompt, 100))
	}
	if optimizedPrompt == "" {
		log.Fatalf("Failed to optimize prompt after %d iterations", iterations)
	}

	fmt.Printf("Initial prompt:\n%s\n\n", initialPrompt)
	fmt.Printf("Optimized prompt:\n%s\n", optimizedPrompt)

	// Test the optimized prompt
	response, err := llm.Generate(ctx, gollm.NewPrompt(optimizedPrompt))
	if err != nil {
		log.Fatalf("Failed to generate summary with optimized prompt: %v", err)
	}

	fmt.Printf("\nGenerated summary:\n%s\n", response)
}

func safeSubstring(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
