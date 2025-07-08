// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/guiperry/gollm_cerebras/llm"
)

// assessPrompt evaluates a prompt's quality and effectiveness using the configured LLM.
// It performs a comprehensive analysis considering multiple factors including custom metrics,
// optimization goals, and historical context.
//
// The assessment process:
// 1. Constructs an evaluation prompt incorporating task description and history
// 2. Requests LLM evaluation of the prompt
// 3. Parses and validates the assessment response
// 4. Normalizes grading for consistency
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - prompt: The prompt to be assessed
//
// Returns:
//   - OptimizationEntry containing the assessment results
//   - Error if assessment fails
//
// The assessment evaluates:
//   - Custom metrics specified in PromptOptimizer
//   - Prompt strengths with examples
//   - Weaknesses with improvement suggestions
//   - Overall effectiveness and efficiency
//   - Alignment with optimization goals
func (po *PromptOptimizer) assessPrompt(ctx context.Context, prompt *llm.Prompt) (OptimizationEntry, error) {
	recentHistory := po.recentHistory()
	assessPrompt := llm.NewPrompt(fmt.Sprintf(`
		Assess the following prompt for the task: %s

		Full Prompt Structure:
		%+v

		Recent History:
		%+v

		Custom Metrics: %v

		Optimization Goal: %s

		Consider the recent history when making your assessment.
		Provide your assessment as a JSON object with the following structure:
		{
			"metrics": [{"name": string, "value": number, "reasoning": string}, ...],
			"strengths": [{"point": string, "example": string}, ...],
			"weaknesses": [{"point": string, "example": string}, ...],
			"suggestions": [{"description": string, "expectedImpact": number, "reasoning": string}, ...],
			"overallScore": number,
			"overallGrade": string,
			"efficiencyScore": number,
			"alignmentWithGoal": number
		}

		IMPORTANT: 
		- Do not use any markdown formatting, code blocks, or backticks in your response.
		- Return only the raw JSON object.
		- For numerical ratings, use a scale of 0 to 20 (inclusive).
		- For the overallGrade:
		  - If using letter grades, use one of: F, D, C, B, A, A+
		  - If using numerical grades, use the same value as overallScore (0-20)
		- Include at least one item in each array (metrics, strengths, weaknesses, suggestions).
		- Provide specific examples and reasoning for each point.
		- Rate the prompt's efficiency and alignment with the optimization goal.
		- Rank suggestions by their expected impact (20 being highest impact).
		- Use clear, jargon-free language in your assessment.
		- Double-check that your response is valid JSON before submitting.
	`, po.taskDesc, prompt, recentHistory, po.customMetrics, po.optimizationGoal))

	// Generate assessment using LLM
	response, err := po.llm.Generate(ctx, assessPrompt)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to assess prompt: %w", err)
	}

	// Parse and validate assessment response
	cleanedResponse := cleanJSONResponse(response)
	var assessment PromptAssessment
	err = json.Unmarshal([]byte(cleanedResponse), &assessment)
	if err != nil {
		po.debugManager.LogResponse(fmt.Sprintf("Raw response: %s", response))
		return OptimizationEntry{}, fmt.Errorf("failed to parse assessment response: %w", err)
	}

	if err := llm.Validate(assessment); err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid assessment structure: %w", err)
	}

	// Normalize grading for consistency
	assessment.OverallGrade, err = normalizeGrade(assessment.OverallGrade, assessment.OverallScore)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid overall grade: %w", err)
	}

	return OptimizationEntry{
		Prompt:     prompt,
		Assessment: assessment,
	}, nil
}

// isOptimizationGoalMet determines if a prompt's assessment meets the configured
// optimization threshold. It supports both numerical and letter-based grading systems.
//
// For numerical ratings:
// - Uses a 0-20 scale
// - Compares against the configured threshold
//
// For letter grades:
// - Converts letter grades to GPA scale (0.0-4.3)
// - Requires A- (3.7) or better to meet goal
//
// Parameters:
//   - assessment: The PromptAssessment to evaluate
//
// Returns:
//   - bool: true if optimization goal is met
//   - error: if rating system is invalid or grade cannot be evaluated
//
// Example threshold values:
//   - Numerical: 0.75 requires score >= 15/20
//   - Letter: Requires A- or better
func (po *PromptOptimizer) isOptimizationGoalMet(assessment PromptAssessment) (bool, error) {
	if po.ratingSystem == "" {
		return false, nil
	}

	switch po.ratingSystem {
	case "numerical":
		return assessment.OverallScore >= 20*po.threshold, nil
	case "letter":
		gradeValues := map[string]float64{
			"A+": 4.3, "A": 4.0, "A-": 3.7,
			"B+": 3.3, "B": 3.0, "B-": 2.7,
			"C+": 2.3, "C": 2.0, "C-": 1.7,
			"D+": 1.3, "D": 1.0, "D-": 0.7,
			"F": 0.0,
		}
		gradeValue, exists := gradeValues[assessment.OverallGrade]
		if !exists {
			return false, fmt.Errorf("invalid grade: %s", assessment.OverallGrade)
		}
		return gradeValue >= 3.7, nil // Equivalent to A- or better
	default:
		return false, fmt.Errorf("unknown rating system: %s", po.ratingSystem)
	}
}
