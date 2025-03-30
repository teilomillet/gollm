package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: GEMINI_API_KEY environment variable not set")
		os.Exit(1)
	}

	// Choose a Gemini model
	model := "gemini-2.0-flash"

	// Create a new LLM instance for Gemini
	llm, err := gollm.NewLLM(
		gollm.SetProvider("google-openai"),
		gollm.SetAPIKey(apiKey),
		gollm.SetModel(model),
		gollm.SetMaxTokens(200),
		gollm.SetLogLevel(gollm.LogLevelDebug),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()

	// Basic text generation prompt
	prompt1 := gollm.NewPrompt("Explain the concept of 'quantum entanglement' in simple terms.")
	response1, err := llm.Generate(ctx, prompt1)
	if err != nil {
		fmt.Printf("Error generating response 1: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Gemini Response:")
	fmt.Println(response1)
	fmt.Println("---")
}
