package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/gollm"
)

func main() {
	llmClient, err := gollm.NewLLM(
		gollm.SetMaxTokens(300),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	question := "How might climate change affect global agriculture?"

	response, err := gollm.ChainOfThought(ctx, llmClient, question,
		gollm.WithMaxLength(300),
		gollm.WithContext("Climate change is causing global temperature increases and changing precipitation patterns."),
		gollm.WithExamples("Effect: Shifting growing seasons, Adaptation: Developing heat-resistant crops"),
		gollm.WithDirectives(
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
