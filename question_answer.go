package goal

import (
	"context"
)

// QuestionAnswer performs question answering
func QuestionAnswer(ctx context.Context, l LLM, question string, contextInfo string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prompt := NewPrompt("Answer the following question based on the given context:").
		WithDirective("Provide a clear and concise answer").
		WithDirective("Use information from the context to support your answer").
		WithOutput("Answer:").
		WithInput("Context: " + contextInfo + "\n\nQuestion: " + question)
	response, _, err := l.Generate(ctx, prompt.String())
	return response, err
}
