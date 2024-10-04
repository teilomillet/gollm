// File: optimizer/assessment.go

package optimizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/llm"
)

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

	response, err := po.llm.Generate(ctx, assessPrompt)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to assess prompt: %w", err)
	}

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

	assessment.OverallGrade, err = normalizeGrade(assessment.OverallGrade, assessment.OverallScore)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid overall grade: %w", err)
	}

	return OptimizationEntry{
		Prompt:     prompt,
		Assessment: assessment,
	}, nil
}

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
