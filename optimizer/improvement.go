// File: optimizer/improvement.go

package optimizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/llm"
)

func (po *PromptOptimizer) generateImprovedPrompt(ctx context.Context, prevEntry OptimizationEntry) (*llm.Prompt, error) {
	recentHistory := po.recentHistory()
	improvePrompt := llm.NewPrompt(fmt.Sprintf(`
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

	po.debugManager.LogPrompt(improvePrompt.String())

	response, err := po.llm.Generate(ctx, improvePrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate improved prompt: %w", err)
	}

	po.debugManager.LogResponse(response)

	cleanedResponse := cleanJSONResponse(response)

	var improvedPrompts struct {
		IncrementalImprovement llm.Prompt `json:"incrementalImprovement"`
		BoldRedesign           llm.Prompt `json:"boldRedesign"`
		ExpectedImpact         struct {
			Incremental float64 `json:"incremental"`
			Bold        float64 `json:"bold"`
		} `json:"expectedImpact"`
	}

	err = json.Unmarshal([]byte(cleanedResponse), &improvedPrompts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse improved prompts: %w", err)
	}

	if improvedPrompts.ExpectedImpact.Bold > improvedPrompts.ExpectedImpact.Incremental {
		return &improvedPrompts.BoldRedesign, nil
	}
	return &improvedPrompts.IncrementalImprovement, nil
}
