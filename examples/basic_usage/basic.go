package basic_usage

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
)

// RunBasicUsage demonstrates basic usage of the gollm package.
func RunBasicUsage() {
	log.Println("Starting the LLM basic usage example...")

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
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}
	log.Println("LLM created successfully with retry mechanism and custom configuration.")

	ctx := context.Background()

	// Example 1: Basic Prompt
	log.Println("\nExample 1: Basic Prompt")
	basicPrompt := gollm.NewPrompt("Explain the concept of 'recursion' in programming.")
	log.Printf("Basic prompt created: %+v\n", basicPrompt)

	response, err := llm.Generate(ctx, basicPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	log.Printf("Basic prompt response:\n%s\n", response.AsText())

	// Example 2: Advanced Prompt with Directives and Output
	log.Println("\nExample 2: Advanced Prompt with Directives and Output")
	advancedPrompt := gollm.NewPrompt("Compare and contrast functional and object-oriented programming paradigms",
		gollm.WithDirectives(
			"Provide at least three key differences",
			"Include examples for each paradigm",
			"Discuss strengths and weaknesses",
		),
		gollm.WithOutput("Comparison of Programming Paradigms:"),
		gollm.WithMaxLength(300),
	)
	log.Printf("Advanced prompt created: %+v\n", advancedPrompt)

	response, err = llm.Generate(ctx, advancedPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	log.Printf("Advanced prompt response:\n%s\n", response.AsText())

	// Example 3: Prompt with Context
	log.Println("\nExample 3: Prompt with Context")
	contextPrompt := gollm.NewPrompt(
		"Summarize the main points",
		gollm.WithContext(
			"The Internet of Things (IoT) refers to the interconnected network of physical devices embedded with electronics, software, sensors, and network connectivity, which enables these objects to collect and exchange data.",
		),
		gollm.WithMaxLength(100),
	)
	log.Printf("Context prompt created: %+v\n", contextPrompt)

	response, err = llm.Generate(ctx, contextPrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	log.Printf("Context prompt response:\n%s\n", response.AsText())

	// Example 4: JSON Schema Generation and Validation
	log.Println("\nExample 4: JSON Schema Generation and Validation")
	log.Println("This example demonstrates how gollm validates prompts using JSON Schema.")
	log.Println("A valid prompt must have input text and can optionally have other properties.")

	// First, let's look at the JSON Schema that defines a valid prompt
	schemaBytes, err := llm.GetPromptJSONSchema()
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}
	log.Println("\nPrompt JSON Schema (defines what makes a valid prompt):")
	log.Printf("%s\n", string(schemaBytes))

	// Now let's test a valid prompt
	log.Println("\n1. Testing a Valid Prompt:")
	log.Println("Creating a prompt with input text: 'Provide an overview of Go language'")
	validPrompt := gollm.NewPrompt("Provide an overview of Go language.")

	// Print the prompt contents so we can see what we're validating
	validPromptJSON, err := json.MarshalIndent(validPrompt, "", "  ")
	if err != nil {
		log.Printf("Warning: Failed to format valid prompt JSON: %v", err)
	}
	log.Printf("\nPrompt contents:\n%s\n", string(validPromptJSON))

	// Validate the prompt
	err = validPrompt.Validate()
	if err != nil {
		log.Printf("\n❌ Unexpected validation error: %v\n", err)
	} else {
		log.Println("\n✓ Validation passed: Prompt is valid because it has the required input text")
	}

	// Now let's test an invalid prompt
	log.Println("\n2. Testing an Invalid Prompt:")
	log.Println("Creating a prompt with NO input text (empty string)")
	invalidPrompt := gollm.NewPrompt("")

	// Print the prompt contents so we can see what we're validating
	invalidPromptJSON, err := json.MarshalIndent(invalidPrompt, "", "  ")
	if err != nil {
		log.Printf("Warning: Failed to format invalid prompt JSON: %v", err)
	}
	log.Printf("\nPrompt contents:\n%s\n", string(invalidPromptJSON))

	// Validate the prompt
	err = invalidPrompt.Validate()
	if err != nil {
		log.Printf("\n✓ Validation failed as expected: %v\n", err)
		log.Println("   This is correct because a prompt must have input text")
	} else {
		log.Println("\n❌ Error: Invalid prompt unexpectedly passed validation!")
	}

	// Example 5: Using Chain of Thought
	log.Println("\nExample 5: Using Chain of Thought")
	cotPrompt := "Explain the process of photosynthesis step by step."
	cotResponse, err := llm.Generate(ctx, gollm.NewPrompt(cotPrompt))
	if err != nil {
		log.Fatalf("Failed to generate Chain of Thought response: %v", err)
	}
	log.Printf("Chain of Thought response:\n%s\n", cotResponse.AsText())

	log.Println("\nBasic usage example completed.")
}
