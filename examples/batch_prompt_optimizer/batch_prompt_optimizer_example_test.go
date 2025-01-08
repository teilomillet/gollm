package main

import (
	"context"
	"os"
	"testing"
	"time"

	"golang.org/x/time/rate"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/optimizer"
)

func TestBatchPromptOptimizer(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping batch prompt optimizer test in short mode")
	}

	// Create LLM client with settings similar to the example but more conservative
	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama3-8b-8192"), // Using a smaller model for tests
		gollm.SetMaxTokens(1024),         // Increased token limit for assessment prompt
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(45*time.Second),           // Longer timeout to handle rate limits
		gollm.SetMaxRetries(5),                     // More retries for rate limits
		gollm.SetRetryDelay(15*time.Second),        // Longer delay between retries
		gollm.SetAPIKey(os.Getenv("GROQ_API_KEY")), // API key from environment variable
	)
	assert.NoError(t, err, "Should create LLM instance")

	// Create optimizer with more conservative rate limit for TPM
	batchOptimizer := optimizer.NewBatchPromptOptimizer(llm)
	batchOptimizer.Verbose = true
	batchOptimizer.SetRateLimit(rate.Every(15*time.Second), 1) // More conservative to stay under TPM

	// Test examples (very concise to minimize token usage)
	examples := []optimizer.PromptExample{
		{
			Name:        "Creative Writing",
			Prompt:      "Write a six-word story about mystery.", // Hemingway style for minimal tokens
			Description: "Create a concise mystery",
			Threshold:   0.9,
			Metrics: []optimizer.Metric{
				{Name: "Impact", Description: "How memorable is the story"},
				{Name: "Mystery", Description: "Does it create intrigue"},
			},
		},
		{
			Name:        "Technical Writing",
			Prompt:      "Define 'variable' in one sentence.", // Minimal technical explanation
			Description: "Create clear definition",
			Threshold:   0.85,
			Metrics: []optimizer.Metric{
				{Name: "Clarity", Description: "Is it clear"},
				{Name: "Accuracy", Description: "Is it correct"},
			},
		},
	}

	// Run optimization with longer timeout for rate limits
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	results := batchOptimizer.OptimizePrompts(ctx, examples)
	assert.Len(t, results, len(examples), "Should get results for all examples")

	// Track rate limit errors to ensure we're handling them
	rateLimitErrors := 0

	// Validate results
	for i, result := range results {
		t.Run(result.Name, func(t *testing.T) {
			// Check basic result structure
			assert.Equal(t, examples[i].Name, result.Name, "Name should match")
			assert.Equal(t, examples[i].Prompt, result.OriginalPrompt, "Original prompt should match")

			if result.Error != nil {
				if result.Error.Error() == "rate limit exceeded" {
					rateLimitErrors++
					t.Logf("Rate limit hit (expected): %v", result.Error)
				} else {
					t.Errorf("Unexpected error: %v", result.Error)
				}
			} else {
				// Validate successful optimization
				assert.NotEmpty(t, result.GeneratedContent, "Should have generated content")

				// Check that optimization respects thresholds
				if examples[i].Threshold > 0.9 {
					assert.NotEmpty(t, result.OptimizedPrompt, "High threshold tasks should have optimized prompts")
				}

				// Log the results for manual review
				// t.Logf("Original Prompt: %s", result.OriginalPrompt)
				if result.OptimizedPrompt != "" {
					t.Logf("Optimized Prompt: %s", result.OptimizedPrompt)
				}
				// t.Logf("Generated Content: %s", result.GeneratedContent)
			}
		})
	}

	// Log rate limit handling statistics
	t.Logf("Rate limit errors encountered and handled: %d", rateLimitErrors)
}
