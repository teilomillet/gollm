// File: examples/2_prompt_types.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/joho/godotenv"
	"github.com/teilomillet/goal"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: Error loading .env file")
	}

	llm, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Example 1: Basic Prompt
	basicPrompt := goal.NewPrompt("What is the capital of France?")
	basicResponse, _, err := llm.Generate(ctx, basicPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate basic response: %v", err)
	}
	fmt.Printf("Basic Prompt Response:\n%s\n\n", basicResponse)

	// Example 2: Prompt with Directive and Output
	directivePrompt := goal.NewPrompt("Explain the concept of recursion").
		Directive("Use a simple example to illustrate").
		Output("Explanation of recursion:")
	directiveResponse, _, err := llm.Generate(ctx, directivePrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate directive response: %v", err)
	}
	fmt.Printf("Directive Prompt Response:\n%s\n\n", directiveResponse)

	// Example 3: Prompt with Context
	contextPrompt := goal.NewPrompt("Summarize the main points").
		Context("The Internet of Things (IoT) is transforming how we live and work. It refers to the interconnected network of physical devices embedded with electronics, software, sensors, and network connectivity, which enables these objects to collect and exchange data.")
	contextResponse, _, err := llm.Generate(ctx, contextPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate context response: %v", err)
	}
	fmt.Printf("Context Prompt Response:\n%s\n\n", contextResponse)

	// Example 4: Prompt with Max Length
	maxLengthPrompt := goal.NewPrompt("Describe the benefits of exercise").
		MaxLength(50)
	maxLengthResponse, _, err := llm.Generate(ctx, maxLengthPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate max length response: %v", err)
	}
	fmt.Printf("Max Length Prompt Response:\n%s\n\n", maxLengthResponse)

	// Example 5: Prompt with Examples (Note: This requires a file with examples)
	examplesPrompt := goal.NewPrompt("Generate a creative name for a tech startup").
		Examples("path/to/examples.txt", 3, "random")
	examplesResponse, _, err := llm.Generate(ctx, examplesPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate examples response: %v", err)
	}
	fmt.Printf("Examples Prompt Response:\n%s\n", examplesResponse)
}
