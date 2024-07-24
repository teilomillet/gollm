package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/teilomillet/goal"
)

func main() {
	fmt.Println("Starting the LLM basic usage example...")

	// Load API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance with custom configuration
	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-3.5-turbo"),
		goal.SetAPIKey(apiKey),
		goal.SetMaxTokens(200),
		goal.SetMaxRetries(3),
		goal.SetRetryDelay(time.Second*2),
		goal.SetDebugLevel(goal.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}
	fmt.Println("LLM created successfully with retry mechanism and custom configuration.")

	ctx := context.Background()

	// Example 1: Basic Prompt
	fmt.Println("\nExample 1: Basic Prompt")
	basicPrompt := goal.NewPrompt("Explain the concept of 'recursion' in programming.")
	fmt.Printf("Basic prompt created: %+v\n", basicPrompt)

	response, err := llm.Generate(ctx, basicPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Basic prompt response:\n%s\n", response)

	// Example 2: Advanced Prompt with Directives and Output
	fmt.Println("\nExample 2: Advanced Prompt with Directives and Output")
	advancedPrompt := goal.NewPrompt("Compare and contrast functional and object-oriented programming paradigms",
		goal.WithDirectives(
			"Provide at least three key differences",
			"Include examples for each paradigm",
			"Discuss strengths and weaknesses",
		),
		goal.WithOutput("Comparison of Programming Paradigms:"),
		goal.WithMaxLength(300),
	)
	fmt.Printf("Advanced prompt created: %+v\n", advancedPrompt)

	response, err = llm.Generate(ctx, advancedPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Advanced prompt response:\n%s\n", response)

	// Example 3: Prompt with Context
	fmt.Println("\nExample 3: Prompt with Context")
	contextPrompt := goal.NewPrompt("Summarize the main points",
		goal.WithContext("The Internet of Things (IoT) refers to the interconnected network of physical devices embedded with electronics, software, sensors, and network connectivity, which enables these objects to collect and exchange data."),
		goal.WithMaxLength(100),
	)
	fmt.Printf("Context prompt created: %+v\n", contextPrompt)

	response, err = llm.Generate(ctx, contextPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Context prompt response:\n%s\n", response)

	// Example 4: JSON Schema Generation and Validation
	fmt.Println("\nExample 4: JSON Schema Generation and Validation")
	schemaBytes, err := advancedPrompt.GenerateJSONSchema()
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}
	fmt.Printf("JSON Schema for Advanced Prompt:\n%s\n", string(schemaBytes))

	invalidPrompt := goal.NewPrompt("") // Invalid because Input is required
	fmt.Printf("Invalid prompt created: %+v\n", invalidPrompt)
	err = invalidPrompt.Validate()
	if err != nil {
		fmt.Printf("Validation error (expected): %v\n", err)
	}

	// Example 5: Using Chain of Thought
	fmt.Println("\nExample 5: Using Chain of Thought")
	cotPrompt := "Explain the process of photosynthesis step by step."
	cotResponse, err := goal.ChainOfThought(ctx, llm, cotPrompt)
	if err != nil {
		log.Fatalf("Failed to generate Chain of Thought response: %v", err)
	}
	fmt.Printf("Chain of Thought response:\n%s\n", cotResponse)

	fmt.Println("\nBasic usage example completed.")
}
