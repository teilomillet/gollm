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

	question := "What are the main challenges in quantum computing?"
	contextInfo := `Quantum computing is an emerging field that uses quantum-mechanical phenomena such as 
	superposition and entanglement to perform computation. It has the potential to solve certain problems 
	much faster than classical computers, particularly in areas like cryptography, optimization, and 
	simulation of quantum systems.`

	prompt := goal.NewPrompt("Answer the following question:").
		Directive("Provide a clear and concise answer").
		Output("Answer:").
		Context(contextInfo).
		Input(question)

	response, _, err := llmClient.Generate(ctx, prompt.String())
	if err != nil {
		log.Fatalf("QuestionAnswer failed: %v", err)
	}

	fmt.Printf("Question: %s\n\n", question)
	fmt.Printf("Context: %s\n\n", contextInfo)
	fmt.Printf("Answer: %s\n", response)
}
