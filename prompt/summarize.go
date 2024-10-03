package gollm

import (
	"context"
	"fmt"
)

var summarizeTemplate = NewPromptTemplate(
	"Summarize",
	"Summarize the given text",
	"Summarize the following text:\n\n{{.Text}}",
	WithPromptOptions(
		WithDirectives(
			"Provide a concise summary",
			"Capture the main points and key details",
		),
		WithOutput("Summary:"),
	),
)

// Summarize performs text summarization
func Summarize(ctx context.Context, l LLM, text string, opts ...PromptOption) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	prompt, err := summarizeTemplate.Execute(map[string]interface{}{
		"Text": text,
	})
	if err != nil {
		return "", fmt.Errorf("failed to execute summarize template: %w", err)
	}

	prompt.Apply(opts...)

	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate ...: %w", err)
	}

	return response, nil
}
