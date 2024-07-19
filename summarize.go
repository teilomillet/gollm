package goal

import (
	"context"
	"fmt"
)

// Summarize performs text summarization
func Summarize(ctx context.Context, l LLM, text string, maxLength int) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prompt := NewPrompt("Summarize the following text:").
		WithDirective(fmt.Sprintf("Provide a concise summary within %d words", maxLength)).
		WithDirective("Capture the main points and key details").
		WithOutput("Summary:").
		WithInput(text)
	response, _, err := l.Generate(ctx, prompt.String())
	return response, err
}
