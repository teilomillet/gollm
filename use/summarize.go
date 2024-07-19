package use

import (
	"fmt"
	"github.com/teilomillet/goal/llm"
)

func Summarize(question string) *llm.Prompt {
	return llm.NewPrompt(fmt.Sprintf("Summarize: %s", question)).
		WithOutput("Answer:").
		WithDirective("Summarize in 2-3 sentences")
}
