// Package optimizer provides a sophisticated prompt optimization system for Language Learning Models (LLMs).
// It implements various strategies for improving prompt quality, efficiency, and alignment with goals
// through iterative refinement and assessment.
package optimizer

import (
	"time"

	"github.com/weave-labs/gollm/internal/debug"
	"github.com/weave-labs/gollm/llm"
)

// Metric represents a quantitative or qualitative measure of prompt performance.
// Each metric provides a specific aspect of evaluation with supporting reasoning.
type Metric struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Reasoning   string  `json:"reasoning"`
	Value       float64 `json:"value" validate:"required,min=0,max=20"`
}

// Strength represents a positive aspect of the prompt with a concrete example.
type Strength struct {
	// Point describes the strength (e.g., "Clear task definition")
	Point string `json:"point"`

	// Example provides a specific instance demonstrating this strength
	Example string `json:"example"`
}

// Weakness represents an area for improvement in the prompt with a concrete example.
type Weakness struct {
	// Point describes the weakness (e.g., "Ambiguous constraints")
	Point string `json:"point"`

	// Example provides a specific instance demonstrating this weakness
	Example string `json:"example"`
}

// Suggestion represents a proposed improvement to the prompt with impact estimation.
type Suggestion struct {
	Description    string  `json:"description"`
	Reasoning      string  `json:"reasoning"`
	ExpectedImpact float64 `json:"expected_impact" validate:"required,min=0,max=20"`
}

// PromptAssessment provides a comprehensive evaluation of a prompt's quality
// including metrics, strengths, weaknesses, and suggestions for improvement.
type PromptAssessment struct {
	OverallGrade      string       `json:"overall_grade" validate:"required,validGrade"`
	Metrics           []Metric     `json:"metrics" validate:"required,min=1"`
	Strengths         []Strength   `json:"strengths" validate:"required,min=1"`
	Weaknesses        []Weakness   `json:"weaknesses" validate:"required,min=1"`
	Suggestions       []Suggestion `json:"suggestions" validate:"required,min=1"`
	OverallScore      float64      `json:"overall_score" validate:"required,min=0,max=20"`
	EfficiencyScore   float64      `json:"efficiency_score" validate:"required,min=0,max=20"`
	AlignmentWithGoal float64      `json:"alignment_with_goal" validate:"required,min=0,max=20"`
}

// OptimizationEntry represents a single step in the optimization process,
// containing both the prompt and its assessment.
type OptimizationEntry struct {
	// Prompt is the LLM prompt being evaluated
	Prompt *llm.Prompt

	// Assessment contains the comprehensive evaluation of the prompt
	Assessment PromptAssessment
}

// OptimizerOption is a function type for configuring the PromptOptimizer.
// It follows the functional options pattern for flexible configuration.
//
//nolint:revive // OptimizerOption follows the Option pattern naming convention commonly used in Go
type OptimizerOption func(*PromptOptimizer)

// IterationCallback is a function type for monitoring optimization progress.
// It's called after each iteration with the current state.
type IterationCallback func(iteration int, entry OptimizationEntry)

// PromptOptimizer orchestrates the prompt optimization process.
// It manages the iterative refinement of prompts through assessment,
// improvement suggestions, and validation.
type PromptOptimizer struct {
	llm               llm.LLM
	iterationCallback IterationCallback
	debugManager      *debug.Manager
	initialPrompt     *llm.Prompt
	taskDesc          string
	optimizationGoal  string
	ratingSystem      string
	history           []OptimizationEntry
	customMetrics     []Metric
	threshold         float64
	maxRetries        int
	retryDelay        time.Duration
	memorySize        int
	iterations        int
}
