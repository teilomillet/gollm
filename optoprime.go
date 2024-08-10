// File: gollm/optoprime.go

package gollm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/internal/opto"
)

type OptoPrime struct {
	manager *opto.OPTOManager
	llm     LLM
}

func NewOptoPrime(l LLM) (*OptoPrime, error) {
	internalLLM, ok := l.(*llmImpl)
	if !ok {
		return nil, fmt.Errorf("invalid LLM implementation")
	}

	executor := &optoExecutor{llm: l}
	feedbackGenerator := &optoFeedbackGenerator{llm: l}

	traceOracle := opto.NewTraceOracle(
		executor.Execute,
		opto.NewDefaultGraphBuilder(),
		feedbackGenerator.GenerateFeedback,
	)

	context := "Optimize prompt for given task"
	manager, err := opto.NewOPTOManager(traceOracle, context, internalLLM.LLM)
	if err != nil {
		return nil, err
	}

	return &OptoPrime{
		manager: manager,
		llm:     l,
	}, nil
}

func (op *OptoPrime) OptimizePrompt(ctx context.Context, initialPrompt string, iterations int) (string, error) {
	return op.manager.Optimize(ctx, initialPrompt, iterations)
}

type optoExecutor struct {
	llm LLM
}

func (e *optoExecutor) Execute(ctx context.Context, prompt string) (interface{}, error) {
	response, err := e.llm.Generate(ctx, NewPrompt(prompt))
	if err != nil {
		return nil, err
	}
	return response, nil
}

type optoFeedbackGenerator struct {
	llm LLM
}

func (fg *optoFeedbackGenerator) GenerateFeedback(result interface{}) *opto.Feedback {
	response, ok := result.(string)
	if !ok {
		return &opto.Feedback{Score: 0, Message: "Invalid result type"}
	}

	feedbackPrompt := NewPrompt(fmt.Sprintf(`
Analyze the following AI-generated response and provide feedback:

Response:
%s

Provide feedback in the following JSON format:
{
    "relevance": float,    // 0.0 to 1.0
    "coherence": float,    // 0.0 to 1.0
    "creativity": float,   // 0.0 to 1.0
    "overall_score": float, // 0.0 to 1.0
    "suggestions": string  // Suggestions for improvement
}
`, response))

	feedbackResponse, err := fg.llm.Generate(context.Background(), feedbackPrompt)
	if err != nil {
		return &opto.Feedback{Score: 0, Message: fmt.Sprintf("Error generating feedback: %v", err)}
	}

	var feedbackData struct {
		Relevance    float64 `json:"relevance"`
		Coherence    float64 `json:"coherence"`
		Creativity   float64 `json:"creativity"`
		OverallScore float64 `json:"overall_score"`
		Suggestions  string  `json:"suggestions"`
	}

	err = json.Unmarshal([]byte(feedbackResponse), &feedbackData)
	if err != nil {
		return &opto.Feedback{Score: 0, Message: fmt.Sprintf("Error parsing feedback: %v", err)}
	}

	return &opto.Feedback{
		Score: feedbackData.OverallScore,
		Message: fmt.Sprintf("Relevance: %.2f, Coherence: %.2f, Creativity: %.2f\nSuggestions: %s",
			feedbackData.Relevance, feedbackData.Coherence, feedbackData.Creativity, feedbackData.Suggestions),
	}
}
