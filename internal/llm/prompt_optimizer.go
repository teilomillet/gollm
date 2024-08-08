// File: internal/llm/prompt_optimizer.go

package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type Metric struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Value       float64 `json:"value"`
}

type PromptAssessment struct {
	Metrics      []Metric `json:"metrics"`
	Strengths    []string `json:"strengths"`
	Weaknesses   []string `json:"weaknesses"`
	Suggestions  []string `json:"suggestions"`
	OverallScore float64  `json:"overallScore"`
	OverallGrade string   `json:"overallGrade"`
}

type PromptOptimizer struct {
	llm              LLM
	debugManager     *DebugManager
	initialPrompt    *Prompt
	taskDesc         string
	customMetrics    []Metric
	optimizationGoal string
	history          []OptimizationEntry
	ratingSystem     string
}

type OptimizationEntry struct {
	Prompt     *Prompt          `json:"prompt"`
	Assessment PromptAssessment `json:"assessment"`
}

type OptimizerOption func(*PromptOptimizer)

type OptimizationRating interface {
	IsGoalMet() bool
	String() string
}

type NumericalRating struct {
	Score float64
	Max   float64
}

func (nr NumericalRating) IsGoalMet() bool {
	return nr.Score >= nr.Max*0.8 // Consider goal met if score is 90% or higher
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

func NewPromptOptimizer(llm LLM, debugManager *DebugManager, initialPrompt *Prompt, taskDesc string, opts ...OptimizerOption) *PromptOptimizer {
	optimizer := &PromptOptimizer{
		llm:           llm,
		debugManager:  debugManager,
		initialPrompt: initialPrompt,
		taskDesc:      taskDesc,
		history:       []OptimizationEntry{},
	}

	for _, opt := range opts {
		opt(optimizer)
	}

	return optimizer
}

func (po *PromptOptimizer) assessPrompt(ctx context.Context, prompt *Prompt) (OptimizationEntry, error) {
	assessPrompt := NewPrompt(fmt.Sprintf(`
        Assess the following prompt for the task: %s

        Full Prompt Structure:
        %+v

        Custom Metrics: %v

        Optimization Goal: %s

        Provide your assessment as a JSON object (not a string) with the following structure:
        {
            "metrics": [{"name": string, "value": number}, ...],
            "strengths": [string],
            "weaknesses": [string],
            "suggestions": [string],
            "overallScore": number,
            "overallGrade": string
        }

        IMPORTANT: Do not use any markdown formatting, code blocks, or backticks in your response.
        Return only the raw JSON object.
    `, po.taskDesc, prompt, po.customMetrics, po.optimizationGoal))

	po.debugManager.LogPrompt(assessPrompt.String())

	response, _, err := po.llm.Generate(ctx, assessPrompt.String())
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to assess prompt: %w", err)
	}

	po.debugManager.LogResponse(response)

	// Clean the response
	cleanedResponse := cleanJSONResponse(response)

	var assessment PromptAssessment
	err = json.Unmarshal([]byte(cleanedResponse), &assessment)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to parse assessment response: %w", err)
	}

	return OptimizationEntry{
		Prompt:     prompt,
		Assessment: assessment,
	}, nil
}

func cleanJSONResponse(response string) string {
	// Remove any markdown code block syntax
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	// If the response starts with a {, assume it's already JSON
	if strings.HasPrefix(response, "{") {
		return response
	}

	// Otherwise, try to find JSON object in the text
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")

	if start != -1 && end != -1 && end > start {
		return response[start : end+1]
	}

	// If we can't find a JSON object, return the original response
	// This will likely cause a JSON parsing error, which we can catch and handle
	return response
}

func (po *PromptOptimizer) generateImprovedPrompt(ctx context.Context, prevEntry OptimizationEntry) (*Prompt, error) {
	improvePrompt := NewPrompt(fmt.Sprintf(`
        Based on the following assessment, generate an improved version of the entire prompt structure:

        Previous prompt: %+v
        Assessment: %+v

        Task Description: %s
        Optimization Goal: %s

        IMPORTANT: Respond ONLY with a raw JSON object. Do not use any markdown formatting, code blocks, or backticks.
        The JSON object should have this structure:
        {
            "input": "improved prompt text",
            "directives": ["directive1", "directive2", ...],
            "examples": ["example1", "example2", ...]
        }
    `, prevEntry.Prompt, prevEntry.Assessment, po.taskDesc, po.optimizationGoal))

	po.debugManager.LogPrompt(improvePrompt.String())

	response, _, err := po.llm.Generate(ctx, improvePrompt.String())
	if err != nil {
		return nil, fmt.Errorf("failed to generate improved prompt: %w", err)
	}

	po.debugManager.LogResponse(response)

	// Clean the response
	cleanedResponse := cleanJSONResponse(response)

	var improvedPrompt Prompt
	err = json.Unmarshal([]byte(cleanedResponse), &improvedPrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse improved prompt: %w", err)
	}

	return &improvedPrompt, nil
}

func (po *PromptOptimizer) OptimizePrompt(ctx context.Context, iterations int) (*Prompt, error) {
	currentPrompt := po.initialPrompt

	for i := 0; i < iterations; i++ {
		entry, err := po.assessPrompt(ctx, currentPrompt)
		if err != nil {
			return nil, fmt.Errorf("optimization failed at iteration %d: %w", i+1, err)
		}

		po.history = append(po.history, entry)

		// Check if the optimization goal has been met
		goalMet, err := po.isOptimizationGoalMet(entry.Assessment)
		if err != nil {
			po.debugManager.LogResponse(fmt.Sprintf("Error checking optimization goal: %v", err))
		} else if goalMet {
			po.debugManager.LogResponse(fmt.Sprintf("Optimization complete after %d iterations. Goal achieved.", i+1))
			break
		}

		currentPrompt, err = po.generateImprovedPrompt(ctx, entry)
		if err != nil {
			return nil, fmt.Errorf("failed to generate improved prompt at iteration %d: %w", i+1, err)
		}

		po.debugManager.LogResponse(fmt.Sprintf("Iteration %d complete. New assessment: %+v", i+1, entry.Assessment))
	}

	return currentPrompt, nil
}

func (po *PromptOptimizer) isOptimizationGoalMet(assessment PromptAssessment) (bool, error) {
	if po.ratingSystem == "" {
		// If no rating system is set, consider the goal not met
		return false, nil
	}

	var rating OptimizationRating
	switch po.ratingSystem {
	case "numerical":
		rating = NumericalRating{Score: assessment.OverallScore, Max: 10}
	case "letter":
		rating = LetterRating{Grade: assessment.OverallGrade}
	default:
		return false, fmt.Errorf("unknown rating system: %s", po.ratingSystem)
	}
	return rating.IsGoalMet(), nil
}

func (po *PromptOptimizer) GetOptimizationHistory() []OptimizationEntry {
	return po.history
}
