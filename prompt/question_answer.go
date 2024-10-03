package gollm

import (
	"context"
	"fmt"
)

var QuestionAnswerTemplate = NewPromptTemplate(
	"QuestionAnswer",
	"Answer the given question",
	"Answer the following question:\n\n{{.Question}}",
	WithPromptOptions(
		WithDirectives("Provide a clear and concise answer"),
		WithOutput("Answer:"),
	),
)

// QuestionAnswer performs question answering
func QuestionAnswer(ctx context.Context, l LLM, question string, opts ...PromptOption) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	prompt, err := QuestionAnswerTemplate.Execute(map[string]interface{}{
		"Question": question,
	})
	if err != nil {
		return "", fmt.Errorf("failed to execute question answer template: %w", err)
	}

	prompt.Apply(opts...)

	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate ...: %w", err)
	}

	return response, nil
}
