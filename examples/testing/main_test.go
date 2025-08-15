package main

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
)

func TestCreateLLM(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	// Test conservative settings
	llm1, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetTemperature(0.7),
		gollm.SetTopP(0.9),
		gollm.SetMaxTokens(300),
	)
	require.NoError(t, err, "Should create LLM with conservative settings")
	require.NotNil(t, llm1, "LLM instance should not be nil")

	// Test creative settings
	llm2, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetTemperature(0.9),
		gollm.SetTopP(0.5),
		gollm.SetMaxTokens(300),
	)
	require.NoError(t, err, "Should create LLM with creative settings")
	require.NotNil(t, llm2, "LLM instance should not be nil")

	// Test invalid settings
	_, err = gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetTemperature(-1.0),
		gollm.SetTopP(2.0),
		gollm.SetMaxTokens(-300),
	)
	assert.Error(t, err, "Should fail with invalid settings")
}

func TestLLMResponseComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping response comparison test in short mode")
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	ctx := context.Background()

	// Create LLMs with different settings
	llm1, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetTemperature(0.7),
		gollm.SetTopP(0.9),
		gollm.SetMaxTokens(300),
	)
	require.NoError(t, err)

	llm2, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetTemperature(0.9),
		gollm.SetTopP(0.5),
		gollm.SetMaxTokens(300),
	)
	require.NoError(t, err)

	// Test with a technical prompt
	prompt := gollm.NewPrompt("Explain the concept of quantum entanglement and its potential applications.")

	// Get response from conservative LLM
	response1, err := llm1.Generate(ctx, prompt)
	require.NoError(t, err, "Should generate response from conservative LLM")
	require.NotEmpty(t, response1, "Response from conservative LLM should not be empty")

	// Verify response contains key technical terms
	lowercaseResponse1 := strings.ToLower(response1.AsText())
	assert.Contains(t, lowercaseResponse1, "quantum", "Response should contain quantum concepts")
	assert.Contains(t, lowercaseResponse1, "entanglement", "Response should explain entanglement")

	// Get response from creative LLM
	response2, err := llm2.Generate(ctx, prompt)
	require.NoError(t, err, "Should generate response from creative LLM")
	require.NotEmpty(t, response2, "Response from creative LLM should not be empty")

	// Verify response contains key technical terms
	lowercaseResponse2 := strings.ToLower(response2.AsText())
	assert.Contains(t, lowercaseResponse2, "quantum", "Response should contain quantum concepts")
	assert.Contains(t, lowercaseResponse2, "entanglement", "Response should explain entanglement")

	// Responses should be different due to different settings
	assert.NotEqual(t, response1, response2, "Responses should differ with different settings")
}

func TestLLMErrorHandling(t *testing.T) {
	// Test with empty API key
	_, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(""),
	)
	assert.Error(t, err, "Should fail with empty API key")

	// Test with invalid API key
	_, err = gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey("invalid-key"),
	)
	assert.Error(t, err, "Should fail with invalid API key")

	// Test with invalid temperature
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey != "" {
		_, err = gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
			gollm.SetAPIKey(apiKey),
			gollm.SetTemperature(-1.0),
		)
		assert.Error(t, err, "Should fail with invalid temperature")
	}
}
