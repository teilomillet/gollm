// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"golang.org/x/time/rate"

	"github.com/weave-labs/gollm/llm"
)

// BatchPromptOptimizer handles concurrent optimization of multiple prompts with rate limiting.
// It provides efficient batch processing capabilities while managing API rate limits
// and resource utilization.
type BatchPromptOptimizer struct {
	LLM         llm.LLM
	rateLimiter *rate.Limiter
	Verbose     bool
}

// NewBatchPromptOptimizer creates a new BatchPromptOptimizer with default rate limiting.
// The default configuration allows one optimization every 3 seconds to prevent API
// rate limit issues while maintaining reasonable throughput.
//
// Parameters:
//   - llm: Language model instance to use for optimizations
//
// Returns:
//   - A configured BatchPromptOptimizer instance
//
// Example:
//
//	optimizer := NewBatchPromptOptimizer(llmInstance)
func NewBatchPromptOptimizer(llmInstance llm.LLM) *BatchPromptOptimizer {
	return &BatchPromptOptimizer{
		LLM: llmInstance,
		rateLimiter: rate.NewLimiter(
			rate.Every(DefaultRateLimitSeconds*time.Second),
			1,
		), // Default: 1 request per 3 seconds
	}
}

// SetRateLimit configures custom rate limiting for API calls.
// This allows fine-tuning of the optimization throughput based on
// API limits and available resources.
//
// Parameters:
//   - r: Rate limit (e.g., rate.Every(2*time.Second))
//   - b: Burst size (maximum number of tokens available immediately)
//
// Example:
//
//	optimizer.SetRateLimit(rate.Every(time.Second), 2) // 1 request/second, burst of 2
func (bpo *BatchPromptOptimizer) SetRateLimit(r rate.Limit, b int) {
	bpo.rateLimiter = rate.NewLimiter(r, b)
}

// PromptExample represents a single prompt to be optimized in a batch process.
// It contains all necessary information for optimization, including evaluation criteria.
type PromptExample struct {
	// Name identifies this prompt in the batch
	Name string

	// Prompt is the initial prompt text to optimize
	Prompt string

	// Description explains the intended use of the prompt
	Description string

	// Metrics defines custom evaluation criteria
	Metrics []Metric

	// Threshold sets the minimum acceptable quality score
	Threshold float64
}

// OptimizationResult contains the outcome of a single prompt optimization.
// It includes both the optimized content and any errors encountered.
type OptimizationResult struct {
	Error            error
	Name             string
	OriginalPrompt   string
	OptimizedPrompt  string
	GeneratedContent string
}

// OptimizePrompts performs concurrent optimization of multiple prompts.
// It manages concurrent processing while respecting rate limits and
// resource constraints.
//
// Features:
// - Concurrent optimization with controlled parallelism
// - Rate limiting to prevent API overload
// - Progress tracking and error handling
// - Optional verbose logging
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - examples: Slice of prompts to optimize
//
// Returns:
//   - Slice of optimization results, one per input prompt
//
// Example:
//
//	results := optimizer.OptimizePrompts(ctx, []PromptExample{
//	    {
//	        Name:        "Query Generation",
//	        Prompt:      "Generate SQL query...",
//	        Description: "SQL query generation optimization",
//	        Threshold:   15.0,
//	    },
//	    // More examples...
//	})
func (bpo *BatchPromptOptimizer) OptimizePrompts(ctx context.Context, examples []PromptExample) []OptimizationResult {
	results := make([]OptimizationResult, len(examples))
	var wg sync.WaitGroup

	for i, example := range examples {
		wg.Add(1)
		go func(i int, example PromptExample) {
			defer wg.Done()

			// Apply rate limiting
			err := bpo.rateLimiter.Wait(ctx)
			if err != nil {
				results[i] = OptimizationResult{
					Name:           example.Name,
					OriginalPrompt: example.Prompt,
					Error:          fmt.Errorf("rate limiter error: %w", err),
				}
				return
			}

			// Configure and perform optimization
			config := OptimizationConfig{
				Prompt:       example.Prompt,
				Description:  example.Description,
				Metrics:      example.Metrics,
				RatingSystem: "numerical",
				Threshold:    example.Threshold,
				MaxRetries:   DefaultMaxRetries,
				RetryDelay:   DefaultRetryDelay,
			}

			optimizedPrompt, response, err := OptimizePrompt(ctx, bpo.LLM, config)
			results[i] = OptimizationResult{
				Name:             example.Name,
				OriginalPrompt:   example.Prompt,
				OptimizedPrompt:  optimizedPrompt.String(),
				GeneratedContent: response.AsText(),
				Error:            err,
			}

			// Log progress if verbose mode is enabled
			if bpo.Verbose {
				slog.Info("Optimized prompt", "name", example.Name, "prompt", optimizedPrompt)
			}
		}(i, example)
	}

	wg.Wait()
	return results
}
