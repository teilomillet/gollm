// optimizer/batch_opimizer.go

package optimizer

import (
	"context"
	"fmt"
	"golang.org/x/time/rate"
	"sync"
	"time"

	"github.com/teilomillet/gollm/llm"
)

// BatchPromptOptimizer handles batch optimization of prompts
type BatchPromptOptimizer struct {
	LLM         llm.LLM
	Verbose     bool
	rateLimiter *rate.Limiter
}

// NewBatchPromptOptimizer creates a new BatchPromptOptimizer
func NewBatchPromptOptimizer(llm llm.LLM) *BatchPromptOptimizer {
	return &BatchPromptOptimizer{
		LLM:         llm,
		rateLimiter: rate.NewLimiter(rate.Every(3*time.Second), 1), // Adjust these values as needed
	}
}

func (bpo *BatchPromptOptimizer) SetRateLimit(r rate.Limit, b int) {
	bpo.rateLimiter = rate.NewLimiter(r, b)
}

type PromptExample struct {
	Name        string
	Prompt      string
	Description string
	Metrics     []Metric
	Threshold   float64
}

type OptimizationResult struct {
	Name             string
	OriginalPrompt   string
	OptimizedPrompt  string
	GeneratedContent string
	Error            error
}

// OptimizePrompts optimizes a batch of prompts concurrently
func (bpo *BatchPromptOptimizer) OptimizePrompts(ctx context.Context, examples []PromptExample) []OptimizationResult {
	results := make([]OptimizationResult, len(examples))
	var wg sync.WaitGroup
	for i, example := range examples {
		wg.Add(1)
		go func(i int, example PromptExample) {
			defer wg.Done()

			// Wait for rate limiter
			err := bpo.rateLimiter.Wait(ctx)
			if err != nil {
				results[i] = OptimizationResult{
					Name:           example.Name,
					OriginalPrompt: example.Prompt,
					Error:          fmt.Errorf("rate limiter error: %w", err),
				}
				return
			}

			config := OptimizationConfig{
				Prompt:       example.Prompt,
				Description:  example.Description,
				Metrics:      example.Metrics,
				RatingSystem: "numerical",
				Threshold:    example.Threshold,
				MaxRetries:   3,
				RetryDelay:   DefaultRetryDelay,
			}
			optimizedPrompt, response, err := OptimizePrompt(ctx, bpo.LLM, config)
			results[i] = OptimizationResult{
				Name:             example.Name,
				OriginalPrompt:   example.Prompt,
				OptimizedPrompt:  optimizedPrompt,
				GeneratedContent: response,
				Error:            err,
			}
			if bpo.Verbose {
				fmt.Printf("Optimized prompt for %s: %s\n", example.Name, optimizedPrompt)
			}
		}(i, example)
	}
	wg.Wait()
	return results
}
