package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/gollm"
)

func main() {
	ollamaEndpoint := "http://localhost:11434"

	// Create a new LLM instance with Ollama provider and memory
	llm, err := gollm.NewLLM(
		gollm.SetProvider("ollama"),
		gollm.SetModel("llama3.1"),
		gollm.SetDebugLevel(gollm.LogLevelDebug),
		gollm.SetOllamaEndpoint(ollamaEndpoint),
		gollm.SetMemory(4000),
		gollm.SetTemperature(0.7),
		gollm.SetTopP(0.9),
		gollm.SetFrequencyPenalty(0.5),
		gollm.SetPresencePenalty(0.5),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	// First prompt
	prompt1 := gollm.NewPrompt("Who was the first person to walk on the moon?")
	response1, err := llm.Generate(ctx, prompt1)
	if err != nil {
		log.Printf("Error generating response for first prompt: %v", err)
	} else {
		fmt.Printf("Response to first prompt: %s\n\n", response1)
	}

	// Second prompt, referencing the first
	prompt2 := gollm.NewPrompt("What year did that event happen?")
	response2, err := llm.Generate(ctx, prompt2)
	if err != nil {
		log.Printf("Error generating response for second prompt: %v", err)
	} else {
		fmt.Printf("Response to second prompt: %s\n\n", response2)
	}

	// Third prompt, to demonstrate memory retention
	prompt3 := gollm.NewPrompt("Can you summarize the information you've provided about the moon landing?")
	response3, err := llm.Generate(ctx, prompt3)
	if err != nil {
		log.Printf("Error generating response for third prompt: %v", err)
	} else {
		fmt.Printf("Response to third prompt: %s\n\n", response3)
	}
}
