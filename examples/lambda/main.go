package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("LAMBDA_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set the LAMBDA_API_KEY environment variable")
		os.Exit(1)
	}

	ctx := context.Background()

	// Example 1: Basic Usage
	fmt.Println("\n=== Example 1: Basic Usage with Lambda Labs ===")
	llm, err := gollm.NewLLM(
		gollm.SetProvider("lambda"),
		gollm.SetAPIKey(apiKey),
		gollm.SetModel("hermes-3-llama-3.1-405b-fp8"), // Lambda Labs model
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt := gollm.NewPrompt("What are the main features of Lambda Labs' cloud GPU platform?")
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 2: Using a Different Model
	fmt.Println("\n=== Example 2: Using Llama 3.1 70B ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("lambda"),
		gollm.SetAPIKey(apiKey),
		gollm.SetModel("llama3.1-70b-instruct-fp8"),
		gollm.SetTemperature(0.5),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt = gollm.NewPrompt("Explain the concept of GPU memory bandwidth in simple terms.")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 3: JSON Schema Validation
	fmt.Println("\n=== Example 3: JSON Schema Validation ===")
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"gpu_name": map[string]interface{}{
				"type": "string",
			},
			"memory_gb": map[string]interface{}{
				"type": "integer",
			},
			"use_cases": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
		},
		"required": []string{"gpu_name", "memory_gb", "use_cases"},
	}

	prompt = gollm.NewPrompt("Describe the NVIDIA H100 GPU and its main use cases")
	response, err = llm.GenerateWithSchema(ctx, prompt, schema)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}
}
