package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	llmClient, err := goal.NewLLM(
		goal.SetMaxTokens(300),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	question := "How might climate change affect global agriculture?"

	response, err := goal.ChainOfThought(ctx, llmClient, question,
		goal.WithMaxLength(300),
		goal.WithContext("Climate change is causing global temperature increases and changing precipitation patterns."),
		goal.WithExamples("Effect: Shifting growing seasons, Adaptation: Developing heat-resistant crops"),
		goal.WithDirectives(
			"Break down the problem into steps",
			"Show your reasoning for each step",
		),
	)
	if err != nil {
		log.Fatalf("ChainOfThought failed: %v", err)
	}

	fmt.Printf("Question: %s\n\n", question)
	fmt.Printf("Chain of Thought Response:\n%s\n", response)
}
