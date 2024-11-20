// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import "time"

// OptimizationConfig holds the configuration parameters for prompt optimization.
// It controls various aspects of the optimization process including evaluation
// criteria, retry behavior, and quality thresholds.
type OptimizationConfig struct {
	// Prompt is the initial prompt text to be optimized
	Prompt string

	// Description explains the intended use and context of the prompt
	Description string

	// Metrics defines custom evaluation criteria for prompt assessment
	// Each metric should have a name and description
	Metrics []Metric

	// RatingSystem specifies the grading approach:
	// - "numerical": Uses a 0-20 scale
	// - "letter": Uses letter grades (F to A+)
	RatingSystem string

	// Threshold sets the minimum acceptable quality score (0.0 to 1.0)
	// For numerical ratings: score must be >= threshold * 20
	// For letter grades: requires grade equivalent to threshold
	// Example: 0.8 requires A- or better
	Threshold float64

	// MaxRetries is the maximum number of retry attempts for failed operations
	// Each retry includes a delay specified by RetryDelay
	MaxRetries int

	// RetryDelay is the duration to wait between retry attempts
	// This helps prevent rate limiting and allows for transient issues to resolve
	RetryDelay time.Duration
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
		Threshold:    0.8, // Requires 16/20 or better
		MaxRetries:   3,
		RetryDelay:   time.Second * 2,
		Metrics: []Metric{
			{Name: "Relevance", Description: "How relevant the prompt is to the task"},
			{Name: "Clarity", Description: "How clear and unambiguous the prompt is"},
			{Name: "Specificity", Description: "How specific and detailed the prompt is"},
		},
	}
}
