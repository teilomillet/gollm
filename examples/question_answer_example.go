package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	cfg := goal.NewConfigBuilder().
		SetProvider("openai").
		SetModel("gpt-3.5-turbo").
		SetMaxTokens(200).
		SetAPIKey("your-api-key-here"). // Replace with your actual API key
		Build()

	llmClient, err := goal.NewLLM(cfg)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	question := "What are the main challenges in quantum computing?"
	contextInfo := `Quantum computing is an emerging field that uses quantum-mechanical phenomena such as 
	superposition and entanglement to perform computation. It has the potential to solve certain problems 
	much faster than classical computers, particularly in areas like cryptography, optimization, and 
	simulation of quantum systems.`

	response, err := goal.QuestionAnswer(ctx, llmClient, question,
		goal.WithContext(contextInfo),
		goal.WithExamples("Challenge: Decoherence, Solution: Error correction techniques"),
		goal.WithMaxLength(200),
	)
	if err != nil {
		log.Fatalf("QuestionAnswer failed: %v", err)
	}

	fmt.Printf("Question: %s\n\n", question)
	fmt.Printf("Context: %s\n\n", contextInfo)
	fmt.Printf("Answer: %s\n", response)
}
