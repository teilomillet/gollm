// File: examples/1_basic_usage.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal/llm"
)

func main() {
	config, err := llm.LoadConfig("")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	llmClient, err := llm.NewLLMFromConfig(config)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	prompt := llm.NewPrompt("What is the capital of France?")
	response, _, err := llmClient.Generate(context.Background(), prompt.String())
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}
