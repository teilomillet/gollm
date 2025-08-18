// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/go-playground/validator/v10"

	"github.com/weave-labs/gollm/internal/debug"
	"github.com/weave-labs/gollm/llm"
)

//nolint:gochecknoglobals // This is too weird the mess with right now.
var registerValidationOnce sync.Once

// registerValidationFunctions registers custom validation functions for the optimizer package.
// This function is called when the optimizer is first used.
func registerValidationFunctions() {
	registerValidationOnce.Do(func() {
		err := llm.RegisterCustomValidation("validGrade", validGrade)
		if err != nil {
			panic(fmt.Sprintf("Failed to register validGrade function: %v", err))
		}
	})
}

// OptimizationRating defines the interface for different rating systems used in prompt optimization.
// Implementations can provide different ways to evaluate if an optimization goal has been met.
type OptimizationRating interface {
	// IsGoalMet determines if the optimization goal has been achieved
	IsGoalMet() bool
	// String returns a string representation of the rating
	String() string
}

// NumericalRating implements OptimizationRating using a numerical score system.
// It evaluates prompts on a scale from 0 to Max.
type NumericalRating struct {
	Score float64 // Current score
	Max   float64 // Maximum possible score
}

// IsGoalMet checks if the numerical score meets the optimization goal.
// Returns true if the score is 90% or higher of the maximum possible score.
func (nr NumericalRating) IsGoalMet() bool {
	return nr.Score >= nr.Max*DefaultGoalMetThreshold // Consider goal met if score is 90% or higher
}

// String formats the numerical rating as a string in the form "score/max".
func (nr NumericalRating) String() string {
	return fmt.Sprintf("%.1f/%.1f", nr.Score, nr.Max)
}

// LetterRating implements OptimizationRating using a letter grade system.
// It evaluates prompts using traditional academic grades (A+, A, B, etc.).
type LetterRating struct {
	Grade string // Letter grade (A+, A, B, etc.)
}

// IsGoalMet checks if the letter grade meets the optimization goal.
// Returns true for grades A, A+, or S.
func (lr LetterRating) IsGoalMet() bool {
	return lr.Grade == "A" || lr.Grade == "A+" || lr.Grade == "S"
}

// String returns the letter grade as a string.
func (lr LetterRating) String() string {
	return lr.Grade
}

// Configuration methods for PromptOptimizer

// WithCustomMetrics sets custom evaluation metrics for the optimizer.
func (po *PromptOptimizer) WithCustomMetrics(metrics ...Metric) {
	po.customMetrics = metrics
}

// WithOptimizationGoal sets the target goal for optimization.
func (po *PromptOptimizer) WithOptimizationGoal(goal string) {
	po.optimizationGoal = goal
}

// WithRatingSystem sets the rating system to use (numerical or letter grades).
func (po *PromptOptimizer) WithRatingSystem(system string) {
	po.ratingSystem = system
}

// WithThreshold sets the minimum acceptable score threshold.
func (po *PromptOptimizer) WithThreshold(threshold float64) {
	po.threshold = threshold
}

// WithIterationCallback sets a callback function to be called after each iteration.
func (po *PromptOptimizer) WithIterationCallback(callback IterationCallback) {
	po.iterationCallback = callback
}

// WithIterations sets the maximum number of optimization iterations.
func (po *PromptOptimizer) WithIterations(iterations int) {
	po.iterations = iterations
}

// WithMaxRetries sets the maximum number of retry attempts per iteration.
func (po *PromptOptimizer) WithMaxRetries(maxRetries int) {
	po.maxRetries = maxRetries
}

// WithRetryDelay sets the delay duration between retry attempts.
func (po *PromptOptimizer) WithRetryDelay(delay time.Duration) {
	po.retryDelay = delay
}

// WithMemorySize sets the number of recent optimization entries to keep in memory.
func (po *PromptOptimizer) WithMemorySize(size int) {
	po.memorySize = size
}

// recentHistory returns the most recent optimization entries based on memory size.
func (po *PromptOptimizer) recentHistory() []OptimizationEntry {
	if len(po.history) <= po.memorySize {
		return po.history
	}
	return po.history[len(po.history)-po.memorySize:]
}

// validGrade validates if a given grade string is a valid letter grade.
func validGrade(fl validator.FieldLevel) bool {
	grade := fl.Field().String()
	validGrades := map[string]bool{
		"A+": true, "A": true, "A-": true,
		"B+": true, "B": true, "B-": true,
		"C+": true, "C": true, "C-": true,
		"D+": true, "D": true, "D-": true,
		"F": true,
	}
	return validGrades[grade]
}

// NewPromptOptimizer creates a new instance of PromptOptimizer with the given configuration.
//
// Parameters:
//   - llm: Language Learning Model interface for generating and evaluating prompts
//   - debugManager: Debug manager for logging and debugging
//   - initialPrompt: Starting prompt to optimize
//   - taskDesc: Description of the optimization task
//   - opts: Optional configuration options
//
// Returns:
//   - Configured PromptOptimizer instance
func NewPromptOptimizer(
	llmInstance llm.LLM,
	debugManager *debug.Manager,
	initialPrompt *llm.Prompt,
	taskDesc string,
	opts ...OptimizerOption,
) *PromptOptimizer {
	// Register validation functions on first use
	registerValidationFunctions()

	optimizer := &PromptOptimizer{
		llm:           llmInstance,
		debugManager:  debugManager,
		initialPrompt: initialPrompt,
		taskDesc:      taskDesc,
		history:       []OptimizationEntry{},
		threshold:     DefaultThreshold,
		maxRetries:    DefaultMaxRetries,
		retryDelay:    DefaultRetryDelay,
		memorySize:    DefaultMemorySize,
		iterations:    DefaultIterations,
	}

	for _, opt := range opts {
		opt(optimizer)
	}

	return optimizer
}

// OptimizePrompt performs iterative optimization of the initial prompt to meet the specified goal.
//
// The optimization process:
// 1. Assesses the current prompt
// 2. Records assessment in history
// 3. Checks if optimization goal is met
// 4. Generates improved prompt if goal not met
// 5. Repeats until goal is met or max iterations reached
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//
// Returns:
//   - Optimized prompt
//   - Error if optimization fails
func (po *PromptOptimizer) OptimizePrompt(ctx context.Context) (*llm.Prompt, error) {
	currentPrompt := po.initialPrompt
	var bestPrompt *llm.Prompt
	var bestScore float64

	for i := range po.iterations {
		var entry OptimizationEntry
		var err error

		// Retry loop for assessment
		for attempt := range po.maxRetries {
			entry, err = po.assessPrompt(ctx, currentPrompt)
			if err == nil {
				break
			}

			po.debugManager.LogResponse("iteration_error",
				fmt.Sprintf("Error in iteration %d, attempt %d: %v", i+1, attempt+1, err))
			if attempt < po.maxRetries-1 {
				po.debugManager.LogResponse("retry_info", fmt.Sprintf("Retrying in %v...", po.retryDelay))
				time.Sleep(po.retryDelay)
			}
		}

		if err != nil {
			return bestPrompt, fmt.Errorf(
				"optimization failed at iteration %d after %d attempts: %w",
				i+1,
				po.maxRetries,
				err,
			)
		}

		po.history = append(po.history, entry)

		// Execute iteration callback if set
		if po.iterationCallback != nil {
			po.iterationCallback(i+1, entry)
		}

		// Update best prompt if current score is higher
		if entry.Assessment.OverallScore > bestScore {
			bestScore = entry.Assessment.OverallScore
			bestPrompt = currentPrompt
		}

		// Check if optimization goal is met
		goalMet, err := po.isOptimizationGoalMet(&entry.Assessment)
		if err != nil {
			po.debugManager.LogResponse("goal_check_error", fmt.Sprintf("Error checking optimization goal: %v", err))
		} else if goalMet {
			po.debugManager.LogResponse("optimization_complete",
				fmt.Sprintf("Optimization complete after %d iterations. Goal achieved.", i+1))
			return currentPrompt, nil
		}

		// Generate improved prompt
		improvedPrompt, err := po.generateImprovedPrompt(ctx, &entry)
		if err != nil {
			po.debugManager.LogResponse("improvement_failed",
				fmt.Sprintf("Failed to generate improved prompt at iteration %d: %v", i+1, err))
			continue
		}

		currentPrompt = improvedPrompt
		po.debugManager.LogResponse("iteration_complete",
			fmt.Sprintf("Iteration %d complete. New prompt: %s", i+1, currentPrompt.Input))
	}

	return bestPrompt, nil
}

// GetOptimizationHistory returns the complete history of optimization attempts.
func (po *PromptOptimizer) GetOptimizationHistory() []OptimizationEntry {
	return po.history
}
