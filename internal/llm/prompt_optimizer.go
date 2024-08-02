// File: internal/llm/prompt_optimizer.go

package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// PromptOptimizer represents the system for optimizing prompts
type PromptOptimizer struct {
	llm           LLM
	debugManager  *DebugManager
	initialPrompt string
	taskDesc      string
	history       []OptimizationEntry
}

// OptimizationEntry represents a single entry in the optimization history
type OptimizationEntry struct {
	Prompt     string           `json:"prompt"`
	Assessment PromptAssessment `json:"assessment"`
}

// PromptAssessment represents the assessment of a prompt
type PromptAssessment struct {
	Clarity       float64  `json:"clarity"`
	Relevance     float64  `json:"relevance"`
	Effectiveness float64  `json:"effectiveness"`
	OverallScore  float64  `json:"overall_score"`
	Strengths     []string `json:"strengths"`
	Weaknesses    []string `json:"weaknesses"`
	Improvements  []string `json:"improvements"`
}

// NewPromptOptimizer creates a new PromptOptimizer
func NewPromptOptimizer(llm LLM, debugManager *DebugManager, initialPrompt, taskDesc string) *PromptOptimizer {
	return &PromptOptimizer{
		llm:           llm,
		debugManager:  debugManager,
		initialPrompt: initialPrompt,
		taskDesc:      taskDesc,
		history:       []OptimizationEntry{},
	}
}

// assessPrompt evaluates the given prompt
func (po *PromptOptimizer) assessPrompt(ctx context.Context, prompt string) (OptimizationEntry, error) {
	assessPrompt := NewPrompt(fmt.Sprintf(`
		Assess the following prompt for the task: %s

		Prompt: %s

		Provide your assessment in the following JSON format:
		{
			"clarity": float64 (0-5, using half-point increments),
			"relevance": float64 (0-5, using half-point increments),
			"effectiveness": float64 (0-5, using half-point increments),
			"overall_score": float64 (0-5, using half-point increments),
			"strengths": [string] (1-3 items),
			"weaknesses": [string] (1-3 items),
			"improvements": [string] (1-3 items)
		}

		Ensure all scores are between 0 and 5, using only half-point increments (0, 0.5, 1, 1.5, ..., 5).
		Provide 1-3 items each for strengths, weaknesses, and improvements.
	`, po.taskDesc, prompt))

	po.debugManager.LogPrompt(assessPrompt.String())

	response, _, err := po.llm.Generate(ctx, assessPrompt.String())
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to assess prompt: %w", err)
	}

	po.debugManager.LogResponse(response)

	// Clean the response by removing markdown code block syntax
	cleanedResponse := strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(response, "```"), "```json"))

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

// generateImprovedPrompt creates an improved prompt based on the previous assessment
func (po *PromptOptimizer) generateImprovedPrompt(ctx context.Context, prevEntry OptimizationEntry) (string, error) {
	improvePrompt := NewPrompt(fmt.Sprintf(`
		Based on the following assessment, generate an improved version of the prompt:

		Previous prompt: %s
		Assessment:
		- Clarity: %.1f
		- Relevance: %.1f
		- Effectiveness: %.1f
		- Overall Score: %.1f
		- Strengths: %v
		- Weaknesses: %v
		- Suggested Improvements: %v

		Task Description: %s

		Provide only the improved prompt as your response, addressing the weaknesses and incorporating the suggested improvements.
	`, prevEntry.Prompt, prevEntry.Assessment.Clarity, prevEntry.Assessment.Relevance,
		prevEntry.Assessment.Effectiveness, prevEntry.Assessment.OverallScore,
		prevEntry.Assessment.Strengths, prevEntry.Assessment.Weaknesses,
		prevEntry.Assessment.Improvements, po.taskDesc))

	po.debugManager.LogPrompt(improvePrompt.String())

	response, _, err := po.llm.Generate(ctx, improvePrompt.String())
	if err != nil {
		return "", fmt.Errorf("failed to generate improved prompt: %w", err)
	}

	po.debugManager.LogResponse(response)

	return response, nil
}

// OptimizePrompt runs the optimization process
func (po *PromptOptimizer) OptimizePrompt(ctx context.Context, iterations int) (string, error) {
	currentPrompt := po.initialPrompt

	for i := 0; i < iterations; i++ {
		entry, err := po.assessPrompt(ctx, currentPrompt)
		if err != nil {
			return "", fmt.Errorf("optimization failed at iteration %d: %w", i+1, err)
		}

		po.history = append(po.history, entry)

		if entry.Assessment.OverallScore >= 4.5 {
			po.debugManager.LogResponse(fmt.Sprintf("Optimization complete after %d iterations. High score achieved.", i+1))
			break
		}

		currentPrompt, err = po.generateImprovedPrompt(ctx, entry)
		if err != nil {
			return "", fmt.Errorf("failed to generate improved prompt at iteration %d: %w", i+1, err)
		}

		po.debugManager.LogResponse(fmt.Sprintf("Iteration %d complete. New overall score: %.1f", i+1, entry.Assessment.OverallScore))
	}

	return currentPrompt, nil
}

// GetOptimizationHistory returns the history of optimization attempts
func (po *PromptOptimizer) GetOptimizationHistory() []OptimizationEntry {
	return po.history
}

