// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/weave-labs/gollm/llm"
)

// generateImprovedPrompt creates an enhanced version of a prompt based on its assessment
// and optimization history. It employs a dual-strategy approach, generating both
// incremental improvements and bold redesigns.
//
// The improvement process:
// 1. Analyzes previous assessment and optimization history
// 2. Generates two alternative improvements:
//   - Incremental: Refines existing approach
//   - Bold: Reimagines prompt structure
//
// 3. Evaluates expected impact of each version
// 4. Selects the version with higher potential impact
//
// The function considers:
// - Identified strengths and weaknesses
// - Historical optimization attempts
// - Task description and goals
// - Efficiency and clarity
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - prevEntry: Previous optimization entry containing prompt and assessment
//
// Returns:
//   - Improved prompt object
//   - Error if improvement generation fails
//
// Example improvement structure:
//
//	{
//	    "incrementalImprovement": {
//	        "input": "Refined prompt text...",
//	        "directives": ["Be more specific", "Add examples"],
//	        "examples": ["Example usage 1", "Example usage 2"],
//	        "reasoning": "Changes address clarity issues while maintaining strengths"
//	    },
//	    "boldRedesign": {
//	        "input": "Completely restructured prompt...",
//	        "directives": ["New approach", "Different perspective"],
//	        "examples": ["New example 1", "New example 2"],
//	        "reasoning": "Novel approach potentially offers better results"
//	    },
//	    "expectedImpact": {
//	        "incremental": 16.5,
//	        "bold": 18.0
//	    }
//	}
func (po *PromptOptimizer) generateImprovedPrompt(
	ctx context.Context,
	prevEntry *OptimizationEntry,
) (*llm.Prompt, error) {
	improvePrompt := po.createImprovementPrompt(prevEntry)

	// Log the improvement request for debugging
	po.debugManager.LogPrompt("improvement_request", improvePrompt.String())

	// Generate improvements using LLM
	response, err := po.llm.Generate(ctx, improvePrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate improved prompt: %w", err)
	}

	// Log the raw response for debugging
	po.debugManager.LogResponse("improvement_response", response.AsText())

	// Extract and parse JSON response
	cleanedResponse := cleanJSONResponse(response.AsText())

	var improvedPrompts struct {
		IncrementalImprovement llm.Prompt `json:"incremental_improvement"`
		BoldRedesign           llm.Prompt `json:"bold_redesign"`
		ExpectedImpact         struct {
			Incremental float64 `json:"incremental"`
			Bold        float64 `json:"bold"`
		} `json:"expected_impact"`
	}

	err = json.Unmarshal([]byte(cleanedResponse), &improvedPrompts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse improved prompts: %w", err)
	}

	// Select the improvement with higher expected impact
	if improvedPrompts.ExpectedImpact.Bold > improvedPrompts.ExpectedImpact.Incremental {
		return &improvedPrompts.BoldRedesign, nil
	}
	return &improvedPrompts.IncrementalImprovement, nil
}

// createImprovementPrompt creates the prompt for generating improvements
func (po *PromptOptimizer) createImprovementPrompt(prevEntry *OptimizationEntry) *llm.Prompt {
	recentHistory := po.recentHistory()
	return llm.NewPrompt(fmt.Sprintf(`
		Based on the following assessment and recent history, generate an improved version of the entire prompt structure:

		Previous prompt: %+v
		Assessment: %+v

		Recent History:
		%+v

		Task Description: %s
		Optimization Goal: %s

		Consider the recent history when generating improvements.

		Provide two versions of the improved prompt:
		1. An incremental improvement
		2. A bold reimagining

		IMPORTANT: Respond ONLY with a raw JSON object. Do not use any markdown formatting, code blocks, or backticks.
		The JSON object should have this structure:
		{
			"incrementalImprovement": {
				"input": "improved prompt text",
				"directives": ["directive1", "directive2", ...],
				"examples": ["example1", "example2", ...],
				"reasoning": "explanation of changes and their link to the assessment"
			},
			"boldRedesign": {
				"input": "reimagined prompt text",
				"directives": ["directive1", "directive2", ...],
				"examples": ["example1", "example2", ...],
				"reasoning": "explanation of the new approach and its potential benefits"
			},
			"expectedImpact": {
				"incremental": number,
				"bold": number
			}
		}

		For each improvement:
		- Directly address weaknesses identified in the assessment.
		- Build upon identified strengths.
		- Ensure alignment with the task description and optimization goal.
		- Strive for efficiency in language use.
		- Use clear, jargon-free language.
		- Provide a brief reasoning for major changes.
		- Rate the expected impact of each version on a scale of 0 to 20.

		Double-check that your response is valid JSON before submitting.
	`, prevEntry.Prompt, prevEntry.Assessment, recentHistory, po.taskDesc, po.optimizationGoal))
}
