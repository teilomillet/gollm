package tools

import (
	"context"
	"fmt"
	"github.com/teilomillet/gollm/llm"
)

var chainOfThoughtTemplate = llm.NewPromptTemplate(
	"ChainOfThought",
	"Perform a chain of thought reasoning",
	"Perform a chain of thought reasoning for the following question:\n\n{{.Question}}",
	llm.WithPromptOptions(
		llm.WithDirectives(
			"Break down the problem into steps",
			"Show your reasoning for each step",
		),
		llm.WithOutput("Chain of Thought:"),
	),
)

// ChainOfThought performs a chain of thought reasoning
func ChainOfThought(ctx context.Context, l llm.LLM, question string, opts ...llm.PromptOption) (string, error) {
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

	// Convert the prompt to a string before passing it to Generate
	promptString := prompt.String()

	response, _, err := l.Generate(ctx, promptString)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response, nil
}

