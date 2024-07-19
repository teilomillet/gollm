package main

import (
	"context"
	"fmt"
	"log"

	"github.com/joho/godotenv"
	"github.com/teilomillet/goal"
)

func main() {
	// Load environment variables from .env file
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: Error loading .env file")
	}

	// Example 1: Basic usage with default configuration
	llm, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	prompt := goal.NewPrompt("Tell me a short joke about programming.")
	response, _, err := llm.Generate(context.Background(), prompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Default LLM response: %s\n\n", response)

	// Example 2: Using a custom configuration file
	customConfigPath := "path/to/custom_config.yaml"
	customLLM, err := goal.NewLLM(customConfigPath)
	if err != nil {
		log.Fatalf("Failed to create LLM with custom config: %v", err)
	}

	customPrompt := goal.NewPrompt("Explain the concept of recursion briefly.")
	response, _, err = customLLM.Generate(context.Background(), customPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Custom LLM response: %s\n\n", response)

	// Example 3: Setting options programmatically
	llm.SetOption("temperature", 0.9)
	llm.SetOption("max_tokens", 50)

	haikuPrompt := goal.NewPrompt("Write a haiku about Go programming.")
	response, _, err = llm.Generate(context.Background(), haikuPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("LLM response with custom options: %s\n\n", response)

	// Example 4: Using a prompt with directives and output specification
	advancedPrompt := goal.NewPrompt("Describe the benefits of using Go for web development").
		Directive("List at least three key advantages").
		Output("Benefits of Go for web development:")

	response, _, err = llm.Generate(context.Background(), advancedPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Advanced prompt response:\n%s\n", response)
}

