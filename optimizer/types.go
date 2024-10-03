// File: optimizer/types.go

package optimizer

import (
	"time"

	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/utils"
)

type Metric struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Value       float64 `json:"value" validate:"required,min=0,max=20"`
	Reasoning   string  `json:"reasoning"`
}

type Strength struct {
	Point   string `json:"point"`
	Example string `json:"example"`
}

type Weakness struct {
	Point   string `json:"point"`
	Example string `json:"example"`
}

type Suggestion struct {
	Description    string  `json:"description"`
	ExpectedImpact float64 `json:"expectedImpact" validate:"required,min=0,max=20"`
	Reasoning      string  `json:"reasoning"`
}

type PromptAssessment struct {
	Metrics           []Metric     `json:"metrics" validate:"required,min=1"`
	Strengths         []Strength   `json:"strengths" validate:"required,min=1"`
	Weaknesses        []Weakness   `json:"weaknesses" validate:"required,min=1"`
	Suggestions       []Suggestion `json:"suggestions" validate:"required,min=1"`
	OverallScore      float64      `json:"overallScore" validate:"required,min=0,max=20"`
	OverallGrade      string       `json:"overallGrade" validate:"required,validGrade"`
	EfficiencyScore   float64      `json:"efficiencyScore" validate:"required,min=0,max=20"`
	AlignmentWithGoal float64      `json:"alignmentWithGoal" validate:"required,min=0,max=20"`
}

type OptimizationEntry struct {
	Prompt     *llm.Prompt
	Assessment PromptAssessment
}

type OptimizerOption func(*PromptOptimizer)

type IterationCallback func(iteration int, entry OptimizationEntry)

type PromptOptimizer struct {
	llm               llm.LLM
	logger            utils.Logger
	debugManager      *utils.DebugManager
	initialPrompt     *llm.Prompt
	taskDesc          string
	customMetrics     []Metric
	optimizationGoal  string
	history           []OptimizationEntry
	ratingSystem      string
	threshold         float64
	iterationCallback IterationCallback
	maxRetries        int
	retryDelay        time.Duration
	memorySize        int
	iterations        int
}

