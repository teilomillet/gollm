package use

import (
	"fmt"
	"github.com/teilomillet/goal/llm"
)

func QuestionAnswer(question string) *llm.Prompt {
	return llm.NewPrompt(fmt.Sprintf("Question: %s", question)).
		WithOutput("Answer:")
}
