package tools

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm/llm"
)

var QuestionAnswerTemplate = llm.NewPromptTemplate(
	"QuestionAnswer",
	"Answer the given question",
	"Answer the following question:\n\n{{.Question}}",
	llm.WithPromptOptions(
		llm.WithDirectives("Provide a clear and concise answer"),
		llm.WithOutput("Answer:"),
	),
)

// QuestionAnswer performs question answering
func QuestionAnswer(ctx context.Context, l llm.LLM, question string, opts ...llm.PromptOption) (string, error) {
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

	// Convert the prompt to a string before passing it to Generate
	promptString := prompt.String()

	response, _, err := l.Generate(ctx, promptString)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	return response, nil
}
