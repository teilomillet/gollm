// Package optimizer provides a sophisticated prompt optimization system for Language Learning Models (LLMs).
// It implements various strategies for improving prompt quality, efficiency, and alignment with goals
// through iterative refinement and assessment.
package optimizer

import (
	"time"

	"github.com/weave-labs/gollm/llm"
	"github.com/weave-labs/gollm/utils"
)

// Metric represents a quantitative or qualitative measure of prompt performance.
// Each metric provides a specific aspect of evaluation with supporting reasoning.
type Metric struct {
	// Name identifies the metric (e.g., "Clarity", "Specificity")
	Name string `json:"name"`

	// Description explains what the metric measures and its significance
	Description string `json:"description"`

	// Value is the numerical score (0-20 scale) assigned to this metric
	Value float64 `json:"value" validate:"required,min=0,max=20"`

	// Reasoning provides the rationale behind the assigned value
	Reasoning string `json:"reasoning"`
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
	// Description outlines the suggested change
	Description string `json:"description"`

	// ExpectedImpact estimates the improvement's effect (0-20 scale)
	ExpectedImpact float64 `json:"expectedImpact" validate:"required,min=0,max=20"`

	// Reasoning explains why this suggestion would improve the prompt
	Reasoning string `json:"reasoning"`
}

// PromptAssessment provides a comprehensive evaluation of a prompt's quality
// including metrics, strengths, weaknesses, and suggestions for improvement.
type PromptAssessment struct {
	// Metrics contains specific performance measurements
	Metrics []Metric `json:"metrics" validate:"required,min=1"`

	// Strengths lists positive aspects of the prompt
	Strengths []Strength `json:"strengths" validate:"required,min=1"`

	// Weaknesses identifies areas needing improvement
	Weaknesses []Weakness `json:"weaknesses" validate:"required,min=1"`

	// Suggestions provides actionable improvements
	Suggestions []Suggestion `json:"suggestions" validate:"required,min=1"`

	// OverallScore represents the prompt's overall quality (0-20 scale)
	OverallScore float64 `json:"overallScore" validate:"required,min=0,max=20"`

	// OverallGrade provides a letter grade assessment (e.g., "A", "B+")
	OverallGrade string `json:"overallGrade" validate:"required,validGrade"`

	// EfficiencyScore measures token usage and processing efficiency
	EfficiencyScore float64 `json:"efficiencyScore" validate:"required,min=0,max=20"`

	// AlignmentWithGoal measures how well the prompt serves its intended purpose
	AlignmentWithGoal float64 `json:"alignmentWithGoal" validate:"required,min=0,max=20"`
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
type OptimizerOption func(*PromptOptimizer)

// IterationCallback is a function type for monitoring optimization progress.
// It's called after each iteration with the current state.
type IterationCallback func(iteration int, entry OptimizationEntry)

// PromptOptimizer orchestrates the prompt optimization process.
// It manages the iterative refinement of prompts through assessment,
// improvement suggestions, and validation.
type PromptOptimizer struct {
	// llm is the language model used for optimization
	llm llm.LLM

	// debugManager handles debug output and tracking
	debugManager *utils.DebugManager

	// initialPrompt is the starting point for optimization
	initialPrompt *llm.Prompt

	// taskDesc describes the intended use of the prompt
	taskDesc string

	// customMetrics defines additional evaluation criteria
	customMetrics []Metric

	// optimizationGoal specifies the target outcome
	optimizationGoal string

	// history tracks the optimization process
	history []OptimizationEntry

	// ratingSystem defines the scoring methodology
	ratingSystem string

	// threshold sets the minimum acceptable score
	threshold float64

	// iterationCallback monitors optimization progress
	iterationCallback IterationCallback

	// maxRetries specifies retry attempts for failed operations
	maxRetries int

	// retryDelay sets the wait time between retries
	retryDelay time.Duration

	// memorySize limits the optimization history length
	memorySize int

	// iterations counts the optimization steps performed
	iterations int
}
