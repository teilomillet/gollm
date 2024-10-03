// File: internal/llm/prompt_optimizer.go

package optimizer

import (
	"context"
	"fmt"
	"time"

	"github.com/go-playground/validator/v10"

	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/utils"
)

type OptimizationRating interface {
	IsGoalMet() bool
	String() string
}

type NumericalRating struct {
	Score float64
	Max   float64
}

func (nr NumericalRating) IsGoalMet() bool {
	return nr.Score >= nr.Max*0.9 // Consider goal met if score is 90% or higher
}

func (nr NumericalRating) String() string {
	return fmt.Sprintf("%.1f/%.1f", nr.Score, nr.Max)
}

type LetterRating struct {
	Grade string
}

func (lr LetterRating) IsGoalMet() bool {
	return lr.Grade == "A" || lr.Grade == "A+" || lr.Grade == "S"
}

func (lr LetterRating) String() string {
	return lr.Grade
}

func (po *PromptOptimizer) WithCustomMetrics(metrics ...Metric) {
	po.customMetrics = metrics
}

func (po *PromptOptimizer) WithOptimizationGoal(goal string) {
	po.optimizationGoal = goal
}

func (po *PromptOptimizer) WithRatingSystem(system string) {
	po.ratingSystem = system
}

func (po *PromptOptimizer) WithThreshold(threshold float64) {
	po.threshold = threshold
}

func (po *PromptOptimizer) WithIterationCallback(callback IterationCallback) {
	po.iterationCallback = callback
}

func (po *PromptOptimizer) WithIterations(iterations int) {
	po.iterations = iterations
}

func (po *PromptOptimizer) WithMaxRetries(maxRetries int) {
	po.maxRetries = maxRetries
}

func (po *PromptOptimizer) WithRetryDelay(delay time.Duration) {
	po.retryDelay = delay
}

func (po *PromptOptimizer) WithMemorySize(size int) {
	po.memorySize = size
}

func (po *PromptOptimizer) recentHistory() []OptimizationEntry {
	if len(po.history) <= po.memorySize {
		return po.history
	}
	return po.history[len(po.history)-po.memorySize:]
}

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

func NewPromptOptimizer(llm llm.LLM, debugManager *utils.DebugManager, initialPrompt *llm.Prompt, taskDesc string, opts ...OptimizerOption) *PromptOptimizer {
	optimizer := &PromptOptimizer{
		llm:           llm,
		debugManager:  debugManager,
		initialPrompt: initialPrompt,
		taskDesc:      taskDesc,
		history:       []OptimizationEntry{},
		threshold:     0.8,
		maxRetries:    3,
		retryDelay:    time.Second * 2,
		memorySize:    2,
		iterations:    5,
	}

	for _, opt := range opts {
		opt(optimizer)
	}

	return optimizer
}

func (po *PromptOptimizer) OptimizePrompt(ctx context.Context) (*llm.Prompt, error) {
	currentPrompt := po.initialPrompt
	var bestPrompt *llm.Prompt
	var bestScore float64

	for i := 0; i < po.iterations; i++ {
		var entry OptimizationEntry
		var err error

		for attempt := 0; attempt < po.maxRetries; attempt++ {
			entry, err = po.assessPrompt(ctx, currentPrompt)
			if err == nil {
				break
			}

			po.debugManager.LogResponse(fmt.Sprintf("Error in iteration %d, attempt %d: %v", i+1, attempt+1, err))
			if attempt < po.maxRetries-1 {
				po.debugManager.LogResponse(fmt.Sprintf("Retrying in %v...", po.retryDelay))
				time.Sleep(po.retryDelay)
			}
		}

		if err != nil {
			return bestPrompt, fmt.Errorf("optimization failed at iteration %d after %d attempts: %w", i+1, po.maxRetries, err)
		}

		po.history = append(po.history, entry)

		// Call the iteration callback if set
		if po.iterationCallback != nil {
			po.iterationCallback(i+1, entry)
		}

		if entry.Assessment.OverallScore > bestScore {
			bestScore = entry.Assessment.OverallScore
			bestPrompt = currentPrompt
		}

		goalMet, err := po.isOptimizationGoalMet(entry.Assessment)
		if err != nil {
			po.debugManager.LogResponse(fmt.Sprintf("Error checking optimization goal: %v", err))
		} else if goalMet {
			po.debugManager.LogResponse(fmt.Sprintf("Optimization complete after %d iterations. Goal achieved.", i+1))
			return currentPrompt, nil
		}

		improvedPrompt, err := po.generateImprovedPrompt(ctx, entry)
		if err != nil {
			po.debugManager.LogResponse(fmt.Sprintf("Failed to generate improved prompt at iteration %d: %v", i+1, err))
			continue
		}

		currentPrompt = improvedPrompt
		po.debugManager.LogResponse(fmt.Sprintf("Iteration %d complete. New prompt: %s", i+1, currentPrompt.Input))
	}

	return bestPrompt, nil
}

func (po *PromptOptimizer) GetOptimizationHistory() []OptimizationEntry {
	return po.history
}

