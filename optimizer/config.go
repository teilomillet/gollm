// File: optimizer/config.go
package optimizer

import "time"

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
