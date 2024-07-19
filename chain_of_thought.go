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
		Directive("Break down the problem into steps").
		Directive("Show your reasoning for each step").
		Output("Chain of Thought:").
		Input(question)

	response, _, err := l.Generate(ctx, prompt.String())
	return response, err
}
