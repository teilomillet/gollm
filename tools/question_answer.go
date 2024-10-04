package tools

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

var QuestionAnswerTemplate = gollm.NewPromptTemplate(
	"QuestionAnswer",
	"Answer the given question",
	"Answer the following question:\n\n{{.Question}}",
	gollm.WithPromptOptions(
		gollm.WithDirectives("Provide a clear and concise answer"),
		gollm.WithOutput("Answer:"),
	),
)

// QuestionAnswer performs question answering
func QuestionAnswer(ctx context.Context, l gollm.LLM, question string, opts ...gollm.PromptOption) (string, error) {
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
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response, nil
}
