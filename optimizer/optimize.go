// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"errors"
	"fmt"

	"github.com/weave-labs/gollm/internal/debug"
	"github.com/weave-labs/gollm/llm"
	"github.com/weave-labs/gollm/providers"
)

// OptimizePrompt performs automated optimization of an LLM prompt and generates a response.
// It uses a sophisticated optimization process that includes:
// - Prompt quality assessment
// - Iterative refinement
// - Performance measurement
// - Response validation
//
// The optimization process follows these steps:
// 1. Initialize optimization with the given configuration
// 2. Assess initial prompt quality
// 3. Apply iterative improvements based on assessment
// 4. Validate against optimization goals
// 5. Generate response using the optimized prompt
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - llm: Language model instance to use for optimization
//   - config: Configuration controlling the optimization process
//
// Returns:
//   - optimizedPrompt: The refined and improved prompt text
//   - response: The LLM's response using the optimized prompt
//   - err: Any error encountered during optimization
//
// Example usage:
//
//	optimizedPrompt, response, err := OptimizePrompt(ctx, llmInstance, OptimizationConfig{
//	    Prompt:      "Initial prompt text...",
//	    Description: "Task description for optimization",
//	    Metrics:     []Metric{{Name: "Clarity", Description: "..."}},
//	    Threshold:   15.0, // Minimum acceptable quality score
//	})
//
// The function uses a PromptOptimizer internally and configures it with:
// - Debug logging for prompts and responses
// - Custom evaluation metrics
// - Configurable rating system
// - Retry mechanisms for reliability
// - Quality thresholds for acceptance
func OptimizePrompt(
	ctx context.Context,
	llmInstance llm.LLM,
	config OptimizationConfig,
) (*llm.Prompt, *providers.Response, error) {
	// Initialize debug manager for logging optimization process
	debugManager := debug.NewDebugManager(true, "./debug_output")

	// Create initial prompt object
	initialPrompt := llm.NewPrompt(config.Prompt)

	// Configure and create optimizer instance
	optimizer := NewPromptOptimizer(llmInstance, debugManager, initialPrompt, config.Description,
		WithCustomMetrics(config.Metrics...),
		WithRatingSystem(config.RatingSystem),
		WithOptimizationGoal("Optimize the prompt for "+config.Description),
		WithMaxRetries(config.MaxRetries),
		WithRetryDelay(config.RetryDelay),
		WithThreshold(config.Threshold),
	)

	// Perform prompt optimization
	optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("optimization failed: %w", err)
	}

	// Validate optimization result
	if optimizedPrompt == nil {
		return nil, nil, errors.New("optimized prompt is nil")
	}

	// Generate response using optimized prompt
	response, err := llmInstance.Generate(ctx, optimizedPrompt)
	if err != nil {
		return optimizedPrompt, nil, fmt.Errorf("response generation failed: %w", err)
	}

	return optimizedPrompt, response, nil
}
