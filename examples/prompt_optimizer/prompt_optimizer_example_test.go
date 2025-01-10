package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/optimizer"
	"github.com/teilomillet/gollm/utils"
)

func TestPromptOptimizer(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping prompt optimizer test in short mode")
	}

	// Check for API key first, before any setup
	groqKey := os.Getenv("GROQ_API_KEY")
	if groqKey == "" {
		t.Fatal("GROQ_API_KEY environment variable must be set for this test")
	}

	// Create LLM client with settings similar to the example
	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama3-8b-8192"), // Same model as example
		gollm.SetMaxTokens(1024),         // Same token limit as example
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(45*time.Second),   // Reasonable timeout for test
		gollm.SetMaxRetries(3),             // More retries for stability
		gollm.SetRetryDelay(2*time.Second), // Conservative retry delay
		gollm.SetAPIKey(groqKey),           // API key from environment
	)
	if err != nil {
		t.Fatalf("Failed to create LLM client: %v", err)
	}

	// Create the same example from the main file
	example := optimizer.PromptExample{
		Name:        "Creative Writing",
		Prompt:      "Write the opening paragraph of a mystery novel set in a small coastal town.",
		Description: "Create an engaging and atmospheric opening that hooks the reader",
		Threshold:   0.9,
		Metrics: []optimizer.Metric{
			{Name: "Atmosphere", Description: "How well the writing evokes the setting"},
			{Name: "Intrigue", Description: "How effectively it sets up the mystery"},
			{Name: "Character Introduction", Description: "How well it introduces key characters"},
		},
	}

	// Create debug manager as in the example
	debugManager := utils.NewDebugManager(llm.GetLogger(), utils.DebugOptions{
		LogPrompts:   true,
		LogResponses: true,
	})

	// Create initial prompt
	initialPrompt := llm.NewPrompt(example.Prompt)

	// Create optimizer instance with same configuration as example
	optimizerInstance := optimizer.NewPromptOptimizer(
		llm,
		debugManager,
		initialPrompt,
		example.Description,
		optimizer.WithCustomMetrics(example.Metrics...),
		optimizer.WithRatingSystem("numerical"),
		optimizer.WithThreshold(example.Threshold),
		optimizer.WithMaxRetries(3),
		optimizer.WithRetryDelay(time.Second*2),
	)

	// Run optimization with reasonable timeout
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	// Test the optimization process
	t.Run("optimize_prompt", func(t *testing.T) {
		optimizedPrompt, err := optimizerInstance.OptimizePrompt(ctx)
		if err != nil {
			t.Fatalf("Optimization failed: %v", err)
		}

		// Verify optimization results
		assert.NotNil(t, optimizedPrompt, "Should return optimized prompt")
		assert.NotEmpty(t, optimizedPrompt.Input, "Optimized prompt should not be empty")

		// Check optimization history
		history := optimizerInstance.GetOptimizationHistory()
		assert.NotEmpty(t, history, "Should have optimization history")

		// Generate response with optimized prompt
		response, err := llm.Generate(ctx, optimizedPrompt)
		assert.NoError(t, err, "Should generate response")
		assert.NotEmpty(t, response, "Should have non-empty response")

		// Validate response content
		lowercaseResponse := strings.ToLower(response)
		assert.True(t,
			strings.Contains(lowercaseResponse, "coastal") ||
				strings.Contains(lowercaseResponse, "town") ||
				strings.Contains(lowercaseResponse, "mystery"),
			"Response should contain relevant content")

		// Log results for manual review
		t.Logf("Original Prompt: %s", example.Prompt)
		t.Logf("Optimized Prompt: %s", optimizedPrompt.Input)
		t.Logf("Generated Content: %s", response)
		t.Logf("Optimization iterations: %d", len(history))

		// Validate optimization metrics
		lastEntry := history[len(history)-1]
		assert.NotNil(t, lastEntry.Assessment, "Should have assessment data")
		assert.True(t, lastEntry.Assessment.OverallScore > 0, "Should have positive assessment score")
	})
}
