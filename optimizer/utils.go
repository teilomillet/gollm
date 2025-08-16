// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// DefaultRetryDelay is the standard duration to wait between retry attempts.
// This value provides a balance between quick retries and avoiding rate limits.
const (
	DefaultRetryDelay = time.Second * 2
)

// cleanJSONResponse processes raw LLM responses to extract valid JSON content.
// It handles common response formats including markdown code blocks and
// extraneous text surrounding the JSON object.
//
// The function:
// 1. Removes markdown code block delimiters
// 2. Trims whitespace
// 3. Extracts JSON object if embedded in other text
//
// Parameters:
//   - response: Raw response string from the LLM
//
// Returns:
//   - Clean JSON string ready for parsing
func cleanJSONResponse(response string) string {
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	if strings.HasPrefix(response, "{") {
		return response
	}

	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")

	if start != -1 && end != -1 && end > start {
		return response[start : end+1]
	}

	return response
}

// WithCustomMetrics configures custom evaluation metrics for the optimizer.
// These metrics are used to assess prompt quality during optimization.
//
// Parameters:
//   - metrics: Variable number of Metric structs defining evaluation criteria
func WithCustomMetrics(metrics ...Metric) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.customMetrics = metrics
	}
}

// WithOptimizationGoal sets the target outcome for prompt optimization.
// The goal helps guide the optimization process and determine success criteria.
//
// Parameters:
//   - goal: Description of the desired optimization outcome
func WithOptimizationGoal(goal string) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.optimizationGoal = goal
	}
}

// WithRatingSystem specifies the grading methodology for prompt assessment.
// Supports both numerical (0-20) and letter (F to A+) grading systems.
//
// Parameters:
//   - system: Either "numerical" or "letter"
func WithRatingSystem(system string) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.ratingSystem = system
	}
}

// WithThreshold sets the minimum acceptable quality score.
// For numerical ratings: threshold * 20 is the minimum score
// For letter grades: threshold determines minimum acceptable grade
//
// Parameters:
//   - threshold: Value between 0.0 and 1.0
func WithThreshold(threshold float64) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.threshold = threshold
	}
}

// WithIterationCallback registers a function to monitor optimization progress.
// The callback is invoked after each optimization iteration.
//
// Parameters:
//   - callback: Function receiving iteration count and optimization entry
func WithIterationCallback(callback IterationCallback) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.iterationCallback = callback
	}
}

// WithIterations sets the maximum number of optimization iterations.
// This limits the optimization process to prevent excessive API calls.
//
// Parameters:
//   - iterations: Maximum number of optimization attempts
func WithIterations(iterations int) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.iterations = iterations
	}
}

// WithMaxRetries configures the retry behavior for failed operations.
// Each retry includes a delay specified by RetryDelay.
//
// Parameters:
//   - maxRetries: Maximum number of retry attempts
func WithMaxRetries(maxRetries int) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.maxRetries = maxRetries
	}
}

// WithRetryDelay sets the duration to wait between retry attempts.
// Longer delays help prevent rate limiting and allow transient issues to resolve.
//
// Parameters:
//   - delay: Duration to wait between retries
func WithRetryDelay(delay time.Duration) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.retryDelay = delay
	}
}

// WithMemorySize sets the number of optimization entries to retain in history.
// This affects the context available for subsequent optimization iterations.
//
// Parameters:
//   - size: Number of historical entries to maintain
func WithMemorySize(size int) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.memorySize = size
	}
}

// normalizeGrade converts between numerical and letter grade formats.
// It ensures consistent grade representation across the optimization process.
//
// Conversion rules:
// - A+: >= 19/20 (95%)
// - A:  >= 17/20 (85%)
// - A-: >= 15/20 (75%)
// - B+: >= 13/20 (65%)
// - B:  >= 11/20 (55%)
// - B-: >= 9/20  (45%)
// - C+: >= 7/20  (35%)
// - C:  >= 5/20  (25%)
// - C-: >= 3/20  (15%)
// - D+: >= 2/20  (10%)
// - D:  >= 1/20  (5%)
// - F:  < 1/20   (<5%)
//
// Parameters:
//   - grade: Letter grade or numeric score string
//   - score: Numerical score (0-20 scale)
//
// Returns:
//   - Normalized letter grade
//   - Error if grade format is invalid
func normalizeGrade(grade string, _ float64) (string, error) {
	validGrades := map[string]bool{
		"A+": true, "A": true, "A-": true,
		"B+": true, "B": true, "B-": true,
		"C+": true, "C": true, "C-": true,
		"D+": true, "D": true, "D-": true,
		"F": true,
	}

	if validGrades[grade] {
		return grade, nil
	}

	numericGrade, err := strconv.ParseFloat(grade, 64)
	if err != nil {
		return "", fmt.Errorf("failed to parse grade '%s' as float: %w", grade, err)
	}

	switch {
	case numericGrade >= GradeThresholdAPlus:
		return "A+", nil // 95%+
	case numericGrade >= GradeThresholdA:
		return "A", nil // 85%+
	case numericGrade >= GradeThresholdAMinus:
		return "A-", nil // 75%+
	case numericGrade >= GradeThresholdBPlus:
		return "B+", nil // 65%+
	case numericGrade >= GradeThresholdB:
		return "B", nil // 55%+
	case numericGrade >= GradeThresholdBMinus:
		return "B-", nil // 45%+
	case numericGrade >= GradeThresholdCPlus:
		return "C+", nil // 35%+
	case numericGrade >= GradeThresholdC:
		return "C", nil // 25%+
	case numericGrade >= GradeThresholdCMinus:
		return "C-", nil // 15%+
	case numericGrade >= GradeThresholdDPlus:
		return "D+", nil // 10%+
	case numericGrade >= GradeThresholdD:
		return "D", nil // 5%+
	default:
		return "F", nil // <5%
	}
}

// Add any other utility functions here that might be used across the optimizer package
