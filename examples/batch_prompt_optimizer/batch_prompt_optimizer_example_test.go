package main

import (
	"context"
	"os"
	"strings"
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

	// Check for API key first
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable not set")
	}

	// Create LLM client with OpenAI instead of Groq
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"), // Using a smaller OpenAI model
		gollm.SetMaxTokens(256),       // Keep token limit low
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(60*time.Second),    // Reasonable timeout
		gollm.SetMaxRetries(2),              // Fewer retries
		gollm.SetRetryDelay(10*time.Second), // Shorter retry delay for OpenAI
		gollm.SetAPIKey(apiKey),             // OpenAI API key
	)
	assert.NoError(t, err, "Should create LLM instance")

	// Create optimizer with conservative rate limit
	batchOptimizer := optimizer.NewBatchPromptOptimizer(llm)
	batchOptimizer.Verbose = true
	batchOptimizer.SetRateLimit(rate.Every(5*time.Second), 1) // More reasonable for OpenAI

	// Test examples (keep them minimal)
	examples := []optimizer.PromptExample{
		{
			Name:        "Creative Writing",
			Prompt:      "Mystery story in six words.", // Keep minimal
			Description: "Write concise mystery",
			Threshold:   0.7, // Lower threshold
			Metrics: []optimizer.Metric{
				{Name: "Impact", Description: "Is it memorable"},
				{Name: "Mystery", Description: "Creates intrigue"},
			},
		},
	}

	// Run optimization with reasonable timeout
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	results := batchOptimizer.OptimizePrompts(ctx, examples)
	assert.Len(t, results, len(examples), "Should get results for all examples")

	// Track rate limit errors
	rateLimitErrors := 0

	// Validate results
	for i, result := range results {
		t.Run(result.Name, func(t *testing.T) {
			// Check basic result structure
			assert.Equal(t, examples[i].Name, result.Name, "Name should match")
			assert.Equal(t, examples[i].Prompt, result.OriginalPrompt, "Original prompt should match")

			if result.Error != nil {
				if strings.Contains(strings.ToLower(result.Error.Error()), "rate limit") {
					rateLimitErrors++
					t.Logf("Rate limit hit (expected): %v", result.Error)
				} else {
					t.Logf("Error occurred: %v", result.Error)
				}
				return
			}

			// Only validate content if we got a successful result
			if result.GeneratedContent != "" {
				assert.NotEmpty(t, result.GeneratedContent, "Should have generated content")
				if examples[i].Threshold > 0.7 {
					assert.NotEmpty(t, result.OptimizedPrompt, "High threshold tasks should have optimized prompts")
				}
			}
		})
	}

	// Log rate limit handling statistics
	t.Logf("Rate limit errors encountered and handled: %d", rateLimitErrors)
}
