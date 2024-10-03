package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
)

func main() {
	fmt.Println("Starting the LLM basic usage example...")

	// Load API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance with custom configuration
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(200),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second*2),
		gollm.SetDebugLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}
	fmt.Println("LLM created successfully with retry mechanism and custom configuration.")

	ctx := context.Background()

	// Example 1: Basic Prompt
	fmt.Println("\nExample 1: Basic Prompt")
	basicPrompt := gollm.NewPrompt("Explain the concept of 'recursion' in programming.")
	fmt.Printf("Basic prompt created: %+v\n", basicPrompt)

	response, err := llm.Generate(ctx, basicPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Basic prompt response:\n%s\n", response)

	// Example 2: Advanced Prompt with Directives and Output
	fmt.Println("\nExample 2: Advanced Prompt with Directives and Output")
	advancedPrompt := gollm.NewPrompt("Compare and contrast functional and object-oriented programming paradigms",
		gollm.WithDirectives(
			"Provide at least three key differences",
			"Include examples for each paradigm",
			"Discuss strengths and weaknesses",
		),
		gollm.WithOutput("Comparison of Programming Paradigms:"),
		gollm.WithMaxLength(300),
	)
	fmt.Printf("Advanced prompt created: %+v\n", advancedPrompt)

	response, err = llm.Generate(ctx, advancedPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Advanced prompt response:\n%s\n", response)

	// Example 3: Prompt with Context
	fmt.Println("\nExample 3: Prompt with Context")
	contextPrompt := gollm.NewPrompt("Summarize the main points",
		gollm.WithContext("The Internet of Things (IoT) refers to the interconnected network of physical devices embedded with electronics, software, sensors, and network connectivity, which enables these objects to collect and exchange data."),
		gollm.WithMaxLength(100),
	)
	fmt.Printf("Context prompt created: %+v\n", contextPrompt)

	response, err = llm.Generate(ctx, contextPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Context prompt response:\n%s\n", response)

	// Example 4: JSON Schema Generation and Validation
	fmt.Println("\nExample 4: JSON Schema Generation and Validation")
	schemaBytes, err := llm.GetPromptJSONSchema()
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}
	fmt.Printf("JSON Schema for Advanced Prompt:\n%s\n", string(schemaBytes))

	// Create a valid prompt with a non-empty input
	validPrompt := gollm.NewPrompt("Provide an overview of Go language.")
	fmt.Printf("Valid prompt created: %+v\n", validPrompt)

	// Validate the prompt
	err = validPrompt.Validate()
	if err != nil {
		fmt.Printf("Validation error: %v\n", err)
	} else {
		fmt.Println("Prompt validation succeeded.")
	}

	// Create an invalid prompt to test validation
	invalidPrompt := gollm.NewPrompt("") // Invalid because Input is required
	fmt.Printf("Invalid prompt created: %+v\n", invalidPrompt)
	err = invalidPrompt.Validate()
	if err != nil {
		fmt.Printf("Validation error (expected): %v\n", err)
	}

	// Example 5: Using Chain of Thought
	fmt.Println("\nExample 5: Using Chain of Thought")
	cotPrompt := "Explain the process of photosynthesis step by step."
	cotResponse, err := llm.Generate(ctx, gollm.NewPrompt(cotPrompt))
	if err != nil {
		log.Fatalf("Failed to generate Chain of Thought response: %v", err)
	}
	fmt.Printf("Chain of Thought response:\n%s\n", cotResponse)

	fmt.Println("\nBasic usage example completed.")
}
