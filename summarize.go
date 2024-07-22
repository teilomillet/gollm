package goal

import (
	"context"
	"fmt"
)

var summarizeTemplate = &PromptTemplate{
	Name:        "Summarize",
	Description: "Summarize the given text",
	Template:    "Summarize the following text:\n\n{{.Text}}",
	Options: []PromptOption{
		WithDirectives(
			"Provide a concise summary",
			"Capture the main points and key details",
		),
		WithOutput("Summary:"),
	},
}

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

	response, _, err := l.Generate(ctx, prompt.String())
	if err != nil {
		return "", fmt.Errorf("failed to generate summary: %w", err)
	}

	return response, nil
}
