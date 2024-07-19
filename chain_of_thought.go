package goal

import (
	"context"
)

// ChainOfThought performs a chain of thought reasoning
func ChainOfThought(ctx context.Context, l LLM, question string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prompt := NewPrompt("Perform a chain of thought reasoning for the following question:").
		WithDirective("Break down the problem into steps").
		WithDirective("Show your reasoning for each step").
		WithOutput("Chain of Thought:").
		WithInput(question)
	response, _, err := l.Generate(ctx, prompt.String())
	return response, err
}
