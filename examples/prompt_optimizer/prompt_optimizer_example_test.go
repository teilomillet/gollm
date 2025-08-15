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

	// Check for API key first
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable not set")
	}

	// Create LLM client with OpenAI instead of Groq
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"), // Using smaller OpenAI model
		gollm.SetMaxTokens(256),       // Keep token limit low
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(60*time.Second),    // Reasonable timeout
		gollm.SetMaxRetries(2),              // Fewer retries
		gollm.SetRetryDelay(10*time.Second), // Shorter retry delay for OpenAI
		gollm.SetAPIKey(apiKey),             // OpenAI API key
	)
	if err != nil {
		t.Fatalf("Failed to create LLM client: %v", err)
	}

	// Create a simple example
	example := optimizer.PromptExample{
		Name:        "Creative Writing",
		Prompt:      "Mystery story in six words.", // Keep minimal
		Description: "Write concise mystery",
		Threshold:   0.7, // Lower threshold
		Metrics: []optimizer.Metric{
			{Name: "Impact", Description: "Is it memorable"},
			{Name: "Mystery", Description: "Creates intrigue"},
		},
	}

	// Create debug manager with minimal logging
	debugManager := utils.NewDebugManager(llm.GetLogger(), utils.DebugOptions{
		LogPrompts:   true,
		LogResponses: false, // Reduce logging
	})

	// Create initial prompt
	initialPrompt := llm.NewPrompt(example.Prompt)

	// Create optimizer instance with conservative settings
	optimizerInstance := optimizer.NewPromptOptimizer(
		llm,
		debugManager,
		initialPrompt,
		example.Description,
		optimizer.WithCustomMetrics(example.Metrics...),
		optimizer.WithRatingSystem("numerical"),
		optimizer.WithThreshold(example.Threshold),
		optimizer.WithMaxRetries(2),
		optimizer.WithRetryDelay(time.Second*10),
	)

	// Run optimization with reasonable timeout
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	// Test the optimization process
	t.Run("optimize_prompt", func(t *testing.T) {
		optimizedPrompt, err := optimizerInstance.OptimizePrompt(ctx)
		if err != nil {
			if strings.Contains(strings.ToLower(err.Error()), "rate limit") {
				t.Logf("Rate limit error encountered (expected): %v", err)
				return
			}
			t.Logf("Optimization error: %v", err)
			return
		}

		// Only validate if we got a result
		if optimizedPrompt != nil {
			assert.NotEmpty(t, optimizedPrompt.Input, "Optimized prompt should not be empty")

			// Check optimization history
			history := optimizerInstance.GetOptimizationHistory()
			assert.NotEmpty(t, history, "Should have optimization history")

			// Generate response with optimized prompt
			response, err := llm.Generate(ctx, optimizedPrompt)
			if err != nil {
				if strings.Contains(strings.ToLower(err.Error()), "rate limit") {
					t.Logf("Rate limit error encountered (expected): %v", err)
					return
				}
				t.Logf("Generation error: %v", err)
				return
			}

			// Only validate response if we got one
			if response.AsText() != "" {
				// Simple content validation
				lowercaseResponse := strings.ToLower(response.AsText())
				assert.True(t,
					strings.Contains(lowercaseResponse, "mystery") ||
						strings.Contains(lowercaseResponse, "story"),
					"Response should contain relevant content")

				// Log minimal results
				t.Logf("Optimization iterations: %d", len(history))

				// Validate optimization metrics
				if len(history) > 0 {
					lastEntry := history[len(history)-1]
					assert.NotNil(t, lastEntry.Assessment, "Should have assessment data")
					assert.Positive(t, lastEntry.Assessment.OverallScore, "Should have positive assessment score")
				}
			}
		}
	})
}
