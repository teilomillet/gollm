package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
)

func TestChatbotMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping chatbot memory test in short mode")
	}

	// Create LLM client with memory enabled
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMemory(4000),
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(30*time.Second),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(5*time.Second),
	)
	require.NoError(t, err, "Should create LLM instance")

	ctx := context.Background()

	// Test conversation with context retention
	t.Run("conversation_memory", func(t *testing.T) {
		// First message about a specific topic
		prompt1 := gollm.NewPrompt("What is the capital of France?")
		response1, err := llm.Generate(ctx, prompt1)
		require.NoError(t, err, "Should generate first response")
		assert.Contains(t, strings.ToLower(response1.AsText()), "paris", "Response should mention Paris")

		// Follow-up question referring to the previous context
		prompt2 := gollm.NewPrompt("What is the population of that city?")
		response2, err := llm.Generate(ctx, prompt2)
		require.NoError(t, err, "Should generate second response")
		assert.Contains(
			t,
			strings.ToLower(response2.AsText()),
			"million",
			"Response should mention population in millions",
		)
	})

	// Test memory clearing
	// t.Run("clear_memory", func(t *testing.T) {
	// 	if memoryLLM, ok := llm.(interface{ ClearMemory() }); ok {
	// 		memoryLLM.ClearMemory()
	//
	// 		// After clearing memory, the context should be lost
	// 		prompt := gollm.NewPrompt("What was the city we were talking about?")
	// 		response, err := llm.Generate(ctx, prompt)
	// 		require.NoError(t, err, "Should generate response after clearing memory")
	// 		assert.NotContains(
	// 			t,
	// 			strings.ToLower(response.AsText()),
	// 			"paris",
	// 			"Response should not reference previous context",
	// 		)
	// 	}
	// })
}

func TestChatbotErrorHandling(t *testing.T) {
	t.Run("invalid api key", func(t *testing.T) {
		_, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
			gollm.SetAPIKey("invalid-key"),
			gollm.SetMaxTokens(500),
		)
		require.Error(t, err, "Should fail with invalid API key")
		assert.Contains(t, err.Error(), "APIKeys", "Error should mention API key validation")
	})

	// Test with empty prompt
	t.Run("empty_prompt", func(t *testing.T) {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
			gollm.SetLogLevel(gollm.LogLevelInfo),
			gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
			gollm.SetMemory(4000),
		)
		require.NoError(t, err, "Should create LLM instance")

		ctx := context.Background()
		prompt := gollm.NewPrompt("")
		_, err = llm.Generate(ctx, prompt)
		require.Error(t, err, "Should fail with empty prompt")
	})
}

func TestChatbotTokenLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping chatbot token limit test in short mode")
	}

	// Create LLM with small memory limit
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMemory(1000), // Small memory limit
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	require.NoError(t, err, "Should create LLM instance")

	ctx := context.Background()

	// Generate multiple responses to fill up memory
	prompts := []string{
		"Write a paragraph about artificial intelligence.",
		"Explain the concept of machine learning.",
		"Describe the impact of technology on society.",
		"Discuss the future of robotics.",
	}

	for _, promptText := range prompts {
		prompt := gollm.NewPrompt(promptText)
		_, err := llm.Generate(ctx, prompt)
		require.NoError(t, err, "Should generate response")
	}

	// Verify that older messages are truncated
	// if memoryLLM, ok := llm.(interface{ GetMemory() []gollm.MemoryMessage }); ok {
	// 	messages := memoryLLM.GetMemory()
	// 	totalTokens := 0
	// 	for _, msg := range messages {
	// 		totalTokens += msg.Tokens
	// 	}
	// 	assert.LessOrEqual(t, totalTokens, 1000, "Total tokens should not exceed memory limit")
	// }
}
