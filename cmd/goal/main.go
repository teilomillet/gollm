// cmd/goal/main.go
package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"goal/llm"
)

func main() {
	provider := flag.String("provider", "openai", "LLM provider (e.g., 'openai', 'google', 'groq', or 'anthropic')")
	model := flag.String("model", "", "Model to use for the LLM provider")
	prompt := flag.String("prompt", "Hello, world!", "Prompt to send to the LLM")
	temperature := flag.Float64("temperature", 0.7, "Temperature for the LLM")
	maxTokens := flag.Int("max-tokens", 100, "Max tokens for the LLM response")
	flag.Parse()

	if *model == "" {
		log.Fatal("Model is required")
	}

	ctx := context.Background()

	llmClient, err := llm.NewLLM(*provider, *model)
	if err != nil {
		log.Fatalf("Error creating LLM client: %v", err)
	}

	llmClient.SetOption(llm.WithTemperature(*temperature))
	llmClient.SetOption(llm.WithMaxTokens(*maxTokens))

	response, err := llmClient.Generate(ctx, *prompt)
	if err != nil {
		log.Fatalf("Error generating text: %v", err)
	}

	fmt.Println("Response:", response)
}
