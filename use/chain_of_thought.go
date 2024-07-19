package use

import (
	"fmt"
	"github.com/teilomillet/goal/llm"
)

func ChainOfThought(question string) *llm.Prompt {
	return llm.NewPrompt(fmt.Sprintf("Question: %s", question)).
		WithOutput("Answer:").
		WithDirective("Think step-by-step").
		WithDirective("Show your reasoning")
}
