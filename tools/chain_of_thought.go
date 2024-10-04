package tools

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

var chainOfThoughtTemplate = gollm.NewPromptTemplate(
	"ChainOfThought",
	"Perform a chain of thought reasoning",
	"Perform a chain of thought reasoning for the following question:\n\n{{.Question}}",
	gollm.WithPromptOptions(
		gollm.WithDirectives(
			"Break down the problem into steps",
			"Show your reasoning for each step",
		),
		gollm.WithOutput("Chain of Thought:"),
	),
)

// ChainOfThought performs a chain of thought reasoning
func ChainOfThought(ctx context.Context, l gollm.LLM, question string, opts ...gollm.PromptOption) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prompt, err := chainOfThoughtTemplate.Execute(map[string]interface{}{
		"Question": question,
	})
	if err != nil {
		return "", fmt.Errorf("failed to execute chain of thought template: %w", err)
	}
	prompt.Apply(opts...)
	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response, nil
}
