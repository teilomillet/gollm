// File: internal/opto/manager.go

package opto

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/teilomillet/gollm/internal/llm"
)

// OPTOManager coordinates the overall optimization process
type OPTOManager struct {
	oracle  TraceOracle
	context string
	llm     llm.LLM
}

type OptoPrimeResponse struct {
	Reasoning  string `json:"reasoning" validate:"required"`
	Suggestion string `json:"suggestion" validate:"required"`
	Prompt     string `json:"prompt" validate:"required"`
}

func NewOPTOManager(oracle TraceOracle, context string, l llm.LLM) (*OPTOManager, error) {
	if l == nil {
		return nil, fmt.Errorf("LLM instance cannot be nil")
	}

	return &OPTOManager{
		oracle:  oracle,
		context: context,
		llm:     l,
	}, nil
}

func (om *OPTOManager) Optimize(ctx context.Context, initialPrompt string, iterations int) (string, error) {
	currentPrompt := initialPrompt
	var bestPrompt string
	var bestScore float64

	for i := 0; i < iterations; i++ {
		log.Printf("Iteration %d - Current Prompt: %s", i, currentPrompt)

		// 1. Execute Trace Oracle
		graph, feedback, err := om.oracle.Execute(ctx, map[string]Parameter{"prompt": NewPromptParameter(currentPrompt, "")})
		if err != nil {
			return "", fmt.Errorf("optimization iteration %d failed: %w", i, err)
		}

		// 2. Update parameters using OptoPrime
		updatedPrompt, err := om.OptoPrime(currentPrompt, graph, feedback, om.context)
		if err != nil {
			return "", fmt.Errorf("parameter update failed at iteration %d: %w", err)
		}

		log.Printf("Iteration %d - Updated Prompt: %s", i, updatedPrompt)

		// 3. Evaluate and store best parameters
		if feedback.Score > bestScore {
			bestScore = feedback.Score
			bestPrompt = updatedPrompt
		}

		currentPrompt = updatedPrompt

		// Check for context cancellation
		select {
		case <-ctx.Done():
			return bestPrompt, ctx.Err()
		default:
		}
	}

	return bestPrompt, nil
}
func (om *OPTOManager) preparePrompt(prompt string, graph Graph, feedback *Feedback, ctx string) string {
	graphStr := graphToString(graph)

	basePrompt := fmt.Sprintf(`
Analyze the following information and suggest improvements to the algorithm. Your response must be a single JSON object with the following structure:

{
    "reasoning": "Your step-by-step reasoning for the suggested changes",
    "suggestion": "The updated algorithm",
    "prompt": "The new prompt to be used"
}

Context: %s

Current Algorithm:
%s

Execution Trace:
%s

Feedback:
Score: %.2f
Message: %s

Ensure your response is a valid JSON object and does not include any text or formatting outside the JSON structure.`,
		ctx, prompt, graphStr, feedback.Score, feedback.Message)

	return basePrompt
}

func (om *OPTOManager) OptoPrime(prompt string, graph Graph, feedback *Feedback, ctx string) (string, error) {
	promptStr := om.preparePrompt(prompt, graph, feedback, ctx)

	response, _, err := om.llm.Generate(context.Background(), promptStr)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	log.Printf("Raw LLM response:\n%s", response)

	// Extract JSON part of the response
	jsonStart := strings.Index(response, "{")
	jsonEnd := strings.LastIndex(response, "}")
	if jsonStart == -1 || jsonEnd == -1 || jsonEnd <= jsonStart {
		return "", fmt.Errorf("failed to find valid JSON in response")
	}
	jsonStr := response[jsonStart : jsonEnd+1]

	// Parse the JSON
	var parsedResponse OptoPrimeResponse
	if err := json.Unmarshal([]byte(jsonStr), &parsedResponse); err != nil {
		return "", fmt.Errorf("failed to parse OptoPrime response: %w", err)
	}

	// Return just the prompt
	return parsedResponse.Prompt, nil
}

func cleanJSONResponse(response string) string {
	// Remove markdown code block syntax if present
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")

	// Remove any text before the first '{' and after the last '}'
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		response = response[start : end+1]
	}

	return strings.TrimSpace(response)
}

func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start != -1 && end != -1 && end > start {
		return s[start : end+1]
	}
	return ""
}

func (om *OPTOManager) extractSuggestion(response string) (string, error) {
	// Split the response into lines
	lines := strings.Split(response, "\n")

	// Possible headers for the Suggestion section
	suggestionHeaders := []string{
		"Suggestion:",
		"4. Suggestion:",
		"### 4. Suggestion:",
		"### Suggestion:",
	}

	var suggestion strings.Builder
	inSuggestion := false

	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)

		// Check if we've reached the Suggestion section
		for _, header := range suggestionHeaders {
			if strings.HasPrefix(trimmedLine, header) {
				inSuggestion = true
				break
			}
		}

		// If we're in the Suggestion section, add the line to our suggestion
		if inSuggestion {
			// Check if we've reached the end of the Suggestion section
			if strings.HasPrefix(trimmedLine, "###") && !strings.Contains(trimmedLine, "Suggestion") {
				break
			}
			suggestion.WriteString(line + "\n")
		}
	}

	if suggestion.Len() == 0 {
		return "", fmt.Errorf("no suggestion found in LLM response")
	}

	return strings.TrimSpace(suggestion.String()), nil
}

func (om *OPTOManager) applySuggestion(currentPrompt, suggestionJSON string) (string, error) {
	var suggestion struct {
		Reasoning  string `json:"reasoning"`
		Suggestion string `json:"suggestion"`
	}

	err := json.Unmarshal([]byte(suggestionJSON), &suggestion)
	if err != nil {
		return "", fmt.Errorf("failed to parse suggestion JSON: %w", err)
	}

	if suggestion.Suggestion == "" {
		return "", fmt.Errorf("empty suggestion received")
	}

	return suggestion.Suggestion, nil
}

func graphToString(graph Graph) string {
	var result strings.Builder
	for _, node := range graph.Nodes() {
		result.WriteString(fmt.Sprintf("Node: %s, Type: %s, Value: %v\n", node.ID(), node.Type(), node.Value()))
	}
	return result.String()
}
