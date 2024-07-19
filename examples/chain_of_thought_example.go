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

	response, err := goal.ChainOfThought(ctx, llmClient, "How might climate change affect global agriculture?")
	if err != nil {
		log.Fatalf("ChainOfThought failed: %v", err)
	}

	fmt.Printf("Chain of Thought Response:\n%s\n", response)
}
