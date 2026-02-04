package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	// Get model name from env or use default
	model := os.Getenv("LMSTUDIO_MODEL")
	if model == "" {
		model = "local-model" // LM Studio uses whatever model is loaded
	}

	// Create LLM instance - LM Studio is now a built-in provider
	llm, err := gollm.NewLLM(
		gollm.SetProvider("lmstudio"),
		gollm.SetModel(model),
		gollm.SetMaxTokens(500),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	prompt := gollm.NewPrompt("Explain what a neural network is in 2-3 sentences.")

	fmt.Println("Sending request to LM Studio...")
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nResponse from LM Studio:")
	fmt.Println(response)
}
