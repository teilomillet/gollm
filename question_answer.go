package goal

import (
	"context"
	"fmt"
)

var QuestionAnswerTemplate = &PromptTemplate{
	Name:        "QuestionAnswer",
	Description: "Answer the given question",
	Template:    "Answer the following question:\n\n{{.Question}}",
	Directives:  []string{"Provide a clear and concise answer"},
	Output:      "Answer:",
}

// QuestionAnswer performs question answering with optional configurations
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

	// Apply all the provided options
	for _, opt := range opts {
		opt(prompt)
	}

	response, _, err := l.Generate(ctx, prompt.String())
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}

	return response, nil
}
