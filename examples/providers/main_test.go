package main

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
)

func setupLLM(t *testing.T) gollm.LLM {
	t.Helper()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	require.NoError(t, err)
	return llm
}

func TestProviderBasicSetup(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	prompt := gollm.NewPrompt("Who was the first person to walk on the moon?")
	response, err := llm.Generate(ctx, prompt)
	require.NoError(t, err)
	require.NotEmpty(t, response)
	assert.Contains(t, response, "Armstrong", "Response should mention Armstrong")
}

func TestProviderConfigurationChange(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// Test with initial configuration
	prompt := gollm.NewPrompt("What is 2+2?")
	response, err := llm.Generate(ctx, prompt)
	require.NoError(t, err)
	require.NotEmpty(t, response)
	assert.Contains(t, response, "4", "Response should contain the answer 4")

	// Create new LLM with different configuration
	newLLM, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(50), // Reduced tokens
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	require.NoError(t, err)

	response, err = newLLM.Generate(ctx, prompt)
	require.NoError(t, err)
	require.NotEmpty(t, response)
	assert.Contains(t, response, "4", "Response should still contain the answer 4")
}

func TestProviderErrorHandling(t *testing.T) {
	// Test with invalid API key
	_, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey("invalid-key"),
		gollm.SetMaxTokens(500),
	)
	require.Error(t, err, "Should fail with invalid API key")
	assert.Contains(t, err.Error(), "APIKeys", "Error should mention API key validation")

	// Test with invalid provider
	_, err = gollm.NewLLM(
		gollm.SetProvider("invalid-provider"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey("test-key"),
	)
	require.Error(t, err, "Should error with invalid provider")
}

func TestProviderConcurrentRequests(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// Create multiple prompts
	prompts := []string{
		"What is 1+1?",
		"What is 2+2?",
		"What is 3+3?",
	}

	// Run requests concurrently
	results := make(chan string, len(prompts))
	errors := make(chan error, len(prompts))

	for _, p := range prompts {
		go func(prompt string) {
			response, err := llm.Generate(ctx, gollm.NewPrompt(prompt))
			if err != nil {
				errors <- err
				return
			}
			results <- response.AsText()
		}(p)
	}

	// Collect results
	for range prompts {
		select {
		case err := <-errors:
			t.Errorf("Concurrent request failed: %v", err)
		case response := <-results:
			require.NotEmpty(t, response)
		}
	}
}
