package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	// AWS credentials are read from environment variables:
	// - AWS_ACCESS_KEY_ID
	// - AWS_SECRET_ACCESS_KEY
	// - AWS_SESSION_TOKEN (optional, for temporary credentials)
	// - AWS_REGION (defaults to "us-east-1")

	if os.Getenv("AWS_ACCESS_KEY_ID") == "" || os.Getenv("AWS_SECRET_ACCESS_KEY") == "" {
		fmt.Println("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
		os.Exit(1)
	}

	ctx := context.Background()

	// Example 1: Using Claude via Bedrock
	fmt.Println("\n=== Example 1: Claude 3 Sonnet via AWS Bedrock ===")
	llm, err := gollm.NewLLM(
		gollm.SetProvider("bedrock"),
		gollm.SetAPIKey(""), // Not used; credentials come from environment
		gollm.SetModel("anthropic.claude-3-sonnet-20240229-v1:0"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt := gollm.NewPrompt("What are the benefits of using AWS Bedrock for AI applications?")
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 2: Using Llama via Bedrock
	fmt.Println("\n=== Example 2: Llama 3 70B via AWS Bedrock ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("bedrock"),
		gollm.SetAPIKey(""),
		gollm.SetModel("meta.llama3-70b-instruct-v1:0"),
		gollm.SetTemperature(0.5),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt = gollm.NewPrompt("Explain how AWS Bedrock differs from direct API access to model providers.")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 3: Using Mistral via Bedrock
	fmt.Println("\n=== Example 3: Mistral via AWS Bedrock ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("bedrock"),
		gollm.SetAPIKey(""),
		gollm.SetModel("mistral.mistral-7b-instruct-v0:2"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	prompt = gollm.NewPrompt("What makes Mistral models unique in the LLM landscape?")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}

	// Example 4: Using a Different Region
	fmt.Println("\n=== Example 4: Using a Different Region ===")
	llm, err = gollm.NewLLM(
		gollm.SetProvider("bedrock"),
		gollm.SetAPIKey(""),
		gollm.SetModel("anthropic.claude-3-haiku-20240307-v1:0"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		fmt.Printf("Error creating LLM: %v\n", err)
		os.Exit(1)
	}

	// Set a specific AWS region
	llm.SetOption("region", "us-west-2")

	prompt = gollm.NewPrompt("What AWS regions support Amazon Bedrock?")
	response, err = llm.Generate(ctx, prompt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n\n", response)
	}
}
