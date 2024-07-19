package goal

import (
	"context"
)

// QuestionAnswer performs question answering
func QuestionAnswer(ctx context.Context, l LLM, question string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prompt := NewPrompt("Answer the following question:").
		Directive("Provide a clear and concise answer").
		Output("Answer:").
		Input(question)

	response, _, err := l.Generate(ctx, prompt.String())
	return response, err
}
