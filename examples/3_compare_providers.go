// File: examples/3_compare_providers.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/joho/godotenv"
	"github.com/teilomillet/goal"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: Error loading .env file")
	}

	llm, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	questions := []string{
		"What is the capital of France?",
		"How does photosynthesis work?",
		"Who wrote 'To Kill a Mockingbird'?",
		"What are the main principles of object-oriented programming?",
	}

	for _, question := range questions {
		answer, err := goal.QuestionAnswer(ctx, llm, question)
		if err != nil {
			log.Printf("Failed to get answer for question '%s': %v\n", question, err)
			continue
		}
		fmt.Printf("Q: %s\nA: %s\n\n", question, answer)
	}
}
