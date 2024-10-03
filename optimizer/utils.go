// File: optimizer/utils.go

package optimizer

import (
	"strconv"
	"strings"
	"time"
)

const (
	DefaultRetryDelay = time.Second * 2
)

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

func WithCustomMetrics(metrics ...Metric) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.customMetrics = metrics
	}
}

func WithOptimizationGoal(goal string) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.optimizationGoal = goal
	}
}

func WithRatingSystem(system string) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.ratingSystem = system
	}
}

func WithThreshold(threshold float64) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.threshold = threshold
	}
}

func WithIterationCallback(callback IterationCallback) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.iterationCallback = callback
	}
}

func WithIterations(iterations int) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.iterations = iterations
	}
}

func WithMaxRetries(maxRetries int) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.maxRetries = maxRetries
	}
}

func WithRetryDelay(delay time.Duration) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.retryDelay = delay
	}
}

func WithMemorySize(size int) OptimizerOption {
	return func(po *PromptOptimizer) {
		po.memorySize = size
	}
}

func normalizeGrade(grade string, score float64) (string, error) {
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
		return "", err
	}

	switch {
	case numericGrade >= 19:
		return "A+", nil
	case numericGrade >= 17:
		return "A", nil
	case numericGrade >= 15:
		return "A-", nil
	case numericGrade >= 13:
		return "B+", nil
	case numericGrade >= 11:
		return "B", nil
	case numericGrade >= 9:
		return "B-", nil
	case numericGrade >= 7:
		return "C+", nil
	case numericGrade >= 5:
		return "C", nil
	case numericGrade >= 3:
		return "C-", nil
	case numericGrade >= 2:
		return "D+", nil
	case numericGrade >= 1:
		return "D", nil
	default:
		return "F", nil
	}
}

// Add any other utility functions here that might be used across the optimizer package
