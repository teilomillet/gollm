package goal

import (
	"context"
	"fmt"
)

var chainOfThoughtTemplate = &PromptTemplate{
	Name:        "ChainOfThought",
	Description: "Perform a chain of thought reasoning",
	Template:    "Perform a chain of thought reasoning for the following question:\n\n{{.Question}}",
	Options: []PromptOption{
		WithDirectives(
			"Break down the problem into steps",
			"Show your reasoning for each step",
		),
		WithOutput("Chain of Thought:"),
	},
}

// ChainOfThought performs a chain of thought reasoning
func ChainOfThought(ctx context.Context, l LLM, question string, opts ...PromptOption) (string, error) {
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

	response, _, err := l.Generate(ctx, prompt.String())
	if err != nil {
		return "", fmt.Errorf("failed to generate chain of thought: %w", err)
	}

	return response, nil
}
