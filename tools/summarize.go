package tools

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm/llm"
)

var summarizeTemplate = llm.NewPromptTemplate(
	"Summarize",
	"Summarize the given text",
	"Summarize the following text:\n\n{{.Text}}",
	llm.WithPromptOptions(
		llm.WithDirectives(
			"Provide a concise summary",
			"Capture the main points and key details",
		),
		llm.WithOutput("Summary:"),
	),
)

// Summarize performs text summarization
func Summarize(ctx context.Context, l llm.LLM, text string, opts ...llm.PromptOption) (string, error) {
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

	// Convert the prompt to a string before passing it to Generate
	promptString := prompt.String()

	response, _, err := l.Generate(ctx, promptString)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}
	return response, nil
}
