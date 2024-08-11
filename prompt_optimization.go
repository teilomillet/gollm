// File: gollm/prompt_optimization.go

package gollm

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// OptimizationConfig holds the configuration for prompt optimization
type OptimizationConfig struct {
	Prompt       string
	Description  string
	Metrics      []Metric
	RatingSystem string
	Threshold    float64
	MaxRetries   int
	RetryDelay   time.Duration
}

// DefaultOptimizationConfig returns a default configuration for prompt optimization
func DefaultOptimizationConfig() OptimizationConfig {
	return OptimizationConfig{
		RatingSystem: "numerical",
		Threshold:    0.8,
		MaxRetries:   3,
		RetryDelay:   time.Second * 2,
		Metrics: []Metric{
			{Name: "Relevance", Description: "How relevant the prompt is to the task"},
			{Name: "Clarity", Description: "How clear and unambiguous the prompt is"},
			{Name: "Specificity", Description: "How specific and detailed the prompt is"},
		},
	}
}

// BatchPromptOptimizer handles batch optimization of prompts
type BatchPromptOptimizer struct {
	LLM     LLM
	Verbose bool
}

// NewBatchPromptOptimizer creates a new BatchPromptOptimizer
func NewBatchPromptOptimizer(llm LLM) *BatchPromptOptimizer {
	return &BatchPromptOptimizer{
		LLM: llm,
	}
}

// PromptExample represents a single prompt to be optimized
type PromptExample struct {
	Name        string
	Prompt      string
	Description string
	Metrics     []Metric
	Threshold   float64
}

// OptimizationResult represents the result of a single prompt optimization
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

			config := OptimizationConfig{
				Prompt:       example.Prompt,
				Description:  example.Description,
				Metrics:      example.Metrics,
				RatingSystem: "numerical",
				Threshold:    example.Threshold,
				MaxRetries:   3,
				RetryDelay:   time.Second * 2,
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

// OptimizePrompt optimizes the given prompt and generates a response
func OptimizePrompt(ctx context.Context, llm LLM, config OptimizationConfig) (optimizedPrompt string, response string, err error) {
	opts := []OptimizerOption{
		WithCustomMetrics(config.Metrics...),
		WithRatingSystem(config.RatingSystem),
		WithOptimizationGoal(fmt.Sprintf("Optimize the prompt for %s", config.Description)),
		WithMaxRetries(config.MaxRetries),
		WithRetryDelay(config.RetryDelay),
		WithThreshold(config.Threshold),
	}

	optimizer := NewPromptOptimizer(llm, config.Prompt, config.Description, opts...)

	optimizedPrompt, err = optimizer.OptimizePrompt(ctx)
	if err != nil {
		return "", "", fmt.Errorf("optimization failed: %w", err)
	}

	response, err = llm.Generate(ctx, NewPrompt(optimizedPrompt))
	if err != nil {
		return optimizedPrompt, "", fmt.Errorf("response generation failed: %w", err)
	}

	return optimizedPrompt, response, nil
}
