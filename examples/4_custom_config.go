// File: examples/4_custom_config.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal/llm"
)

func main() {
	customConfig := &llm.Config{
		Provider:    "anthropic",
		Model:       "claude-2.0",
		Temperature: 0.9,
		MaxTokens:   50,
		LogLevel:    "debug",
	}

	llmClient, err := llm.NewLLMFromConfig(customConfig)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	prompt := llm.NewPrompt("Generate a creative name for a sci-fi spaceship.")
	response, _, err := llmClient.Generate(context.Background(), prompt.String())
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Spaceship name: %s\n", response)
}
