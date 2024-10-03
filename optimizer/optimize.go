// File: optimizer/optimize.go
package optimizer

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/utils"
)

// OptimizePrompt optimizes the given prompt and generates a response
func OptimizePrompt(ctx context.Context, llm llm.LLM, config OptimizationConfig) (optimizedPrompt string, response string, err error) {
	debugManager := utils.NewDebugManager(llm.GetLogger(), utils.DebugOptions{LogPrompts: true, LogResponses: true})
	initialPrompt := llm.NewPrompt(config.Prompt)
	optimizer := NewPromptOptimizer(llm, debugManager, initialPrompt, config.Description,
		WithCustomMetrics(config.Metrics...),
		WithRatingSystem(config.RatingSystem),
		WithOptimizationGoal(fmt.Sprintf("Optimize the prompt for %s", config.Description)),
		WithMaxRetries(config.MaxRetries),
		WithRetryDelay(config.RetryDelay),
		WithThreshold(config.Threshold),
	)

	optimizedPromptObj, err := optimizer.OptimizePrompt(ctx)
	if err != nil {
		return "", "", fmt.Errorf("optimization failed: %w", err)
	}

	if optimizedPromptObj == nil {
		return "", "", fmt.Errorf("optimized prompt is nil")
	}

	optimizedPrompt = optimizedPromptObj.String()

	response, _, err = llm.Generate(ctx, optimizedPrompt)
	if err != nil {
		return optimizedPrompt, "", fmt.Errorf("response generation failed: %w", err)
	}

	return optimizedPrompt, response, nil
}
