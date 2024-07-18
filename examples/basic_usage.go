package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/goal/llm"
	"go.uber.org/zap"
)

func main() {
	// Set log level
	llm.SetLogLevel(zap.InfoLevel)

	// Ensure the appropriate API key is set in the environment
	provider := "anthropic"
	apiKeyEnv := fmt.Sprintf("%s_API_KEY", provider)
	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		llm.Logger.Fatal("API key not set", zap.String("env_var", apiKeyEnv))
	}

	// Create LLM provider
	model := "claude-3-opus-20240229"
	llmProvider, err := llm.GetProvider(provider, apiKey, model)
	if err != nil {
		llm.Logger.Fatal("Error creating LLM provider", zap.Error(err))
	}

	// Create LLM client
	llmClient := llm.NewLLM(llmProvider)

	// Set options
	llmClient.SetOption("temperature", 0.7)
	llmClient.SetOption("max_tokens", 100)

	// Generate text
	ctx := context.Background()
	prompt := "Explain the concept of recursion in programming."
	response, err := llmClient.Generate(ctx, prompt)
	if err != nil {
		llm.Logger.Fatal("Error generating text", zap.Error(err))
	}

	fmt.Println("Response:", response)
}
