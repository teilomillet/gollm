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
		Directive(fmt.Sprintf("Provide a concise summary within %d words", maxLength)).
		Directive("Capture the main points and key details").
		Output("Summary:").
		Input(text)

	response, _, err := l.Generate(ctx, prompt.String())
	return response, err
}
