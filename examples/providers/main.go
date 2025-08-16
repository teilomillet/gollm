package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/gollm"
)

func main() {
	// Create a new LLM instance with Ollama provider
	llm, err := gollm.NewLLM(
		gollm.SetProvider("ollama"),
		gollm.SetModel("llama3.1"),
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetOllamaEndpoint("http://localhost:11434"), // Set initial endpoint
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a prompt using NewPrompt function
	prompt := gollm.NewPrompt("Who was the first person to walk on the moon?")

	// Generate a response with the initial endpoint
	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response with initial endpoint: %s\n", response.AsText())

	// Change the Ollama endpoint
	err = llm.SetOllamaEndpoint("http://localhost:11435") // Change to a different port for demonstration
	if err != nil {
		log.Printf("Failed to set new Ollama endpoint: %v", err)
	} else {
		// Try to generate a response with the new endpoint
		response, err = llm.Generate(ctx, prompt)
		if err != nil {
			log.Printf("Failed to generate response with new endpoint: %v", err)
		} else {
			fmt.Printf("Response with new endpoint: %s\n", response.AsText())
		}
	}
}
