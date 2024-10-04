package tools

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

var summarizeTemplate = gollm.NewPromptTemplate(
	"Summarize",
	"Summarize the given text",
	"Summarize the following text:\n\n{{.Text}}",
	gollm.WithPromptOptions(
		gollm.WithDirectives(
			"Provide a concise summary",
			"Capture the main points and key details",
		),
		gollm.WithOutput("Summary:"),
	),
)

// Summarize performs text summarization
func Summarize(ctx context.Context, l gollm.LLM, text string, opts ...gollm.PromptOption) (string, error) {
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
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response, nil
}
