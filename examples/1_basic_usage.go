// File: examples/1_basic_usage.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	llmClient, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()
	question := "What is the capital of France?"

	answer, err := goal.QuestionAnswer(ctx, llmClient, question, "")
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Answer: %s\n", answer)
}
