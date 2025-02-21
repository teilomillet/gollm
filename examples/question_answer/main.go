package main

import (
	"context"
	"fmt"
	"log"

	"github.com/mauza/gollm"
	"github.com/mauza/gollm/presets"
)

func main() {
	llmClient, err := gollm.NewLLM(
		gollm.SetMaxTokens(500),
	)

	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	question := "What are the main challenges in quantum computing?"
	contextInfo := `Quantum computing is an emerging field that uses quantum-mechanical phenomena such as 
	superposition and entanglement to perform computation. It has the potential to solve certain problems 
	much faster than classical computers, particularly in areas like cryptography, optimization, and 
	simulation of quantum systems.`

	response, err := presets.QuestionAnswer(ctx, llmClient, question,
		gollm.WithContext(contextInfo),
		gollm.WithExamples("Challenge: Decoherence, Solution: Error correction techniques"),
		gollm.WithMaxLength(200),
		gollm.WithDirectives(
			"Provide a concise answer",
			"Address the main challenges mentioned in the question",
		),
	)
	if err != nil {
		log.Fatalf("QuestionAnswer failed: %v", err)
	}

	fmt.Printf("Question: %s\n\n", question)
	fmt.Printf("Context: %s\n\n", contextInfo)
	fmt.Printf("Answer: %s\n", response)
}
