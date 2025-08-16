// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import "time"

// OptimizationConfig holds the configuration parameters for prompt optimization.
// It controls various aspects of the optimization process including evaluation
// criteria, retry behavior, and quality thresholds.
type OptimizationConfig struct {
	Prompt       string
	Description  string
	RatingSystem string
	Metrics      []Metric
	Threshold    float64
	MaxRetries   int
	RetryDelay   time.Duration
}

// DefaultOptimizationConfig returns a default configuration for prompt optimization.
// The default configuration provides a balanced set of parameters suitable for
// most optimization scenarios.
//
// Default values:
//   - RatingSystem: "numerical" (0-20 scale)
//   - Threshold: 0.8 (requires 16/20 or better)
//   - MaxRetries: 3 attempts
//   - RetryDelay: 2 seconds
//   - Metrics: Relevance, Clarity, and Specificity
//
// Example usage:
//
//	config := DefaultOptimizationConfig()
//	config.Prompt = "Your prompt text..."
//	config.Description = "Task description..."
//
// The default metrics evaluate:
//   - Relevance: Alignment with the intended task
//   - Clarity: Unambiguous and understandable phrasing
//   - Specificity: Level of detail and precision
func DefaultOptimizationConfig() OptimizationConfig {
	return OptimizationConfig{
		RatingSystem: "numerical",
		Threshold:    DefaultThreshold, // Requires 16/20 or better
		MaxRetries:   DefaultMaxRetries,
		RetryDelay:   DefaultRetryDelay,
		Metrics: []Metric{
			{Name: "Relevance", Description: "How relevant the prompt is to the task"},
			{Name: "Clarity", Description: "How clear and unambiguous the prompt is"},
			{Name: "Specificity", Description: "How specific and detailed the prompt is"},
		},
	}
}
