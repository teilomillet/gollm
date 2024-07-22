// File: examples/3_compare_providers.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {

	providers := []string{"openai", "anthropic"}
	llms := make(map[string]goal.LLM)

	for _, provider := range providers {
		llm, err := goal.NewLLM()
		if err != nil {
			log.Fatalf("Failed to create LLM client for %s: %v", provider, err)
		}
		llms[provider] = llm
	}

	ctx := context.Background()

	questions := []string{
		"What is the capital of France?",
		"How does photosynthesis work?",
		"Who wrote 'To Kill a Mockingbird'?",
	}

	for _, question := range questions {
		fmt.Printf("Question: %s\n", question)

		for provider, llm := range llms {
			answer, err := goal.QuestionAnswer(ctx, llm, question,
				goal.WithMaxLength(50),
				goal.WithDirectives("Provide a concise answer"),
			)
			if err != nil {
				log.Printf("Failed to get answer from %s: %v\n", provider, err)
				continue
			}
			fmt.Printf("%s answer: %s\n", provider, answer)
		}
		fmt.Println()
	}
}
