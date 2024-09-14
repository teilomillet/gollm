package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/gollm"
)

func main() {
	ctx := context.Background()

	// Create two LLM instances with different settings
	llm1, err := createLLM(0.7, 0.9, 0.05, 1.1, 64, 1, 0.1, 5.0, 1.0, 42)
	if err != nil {
		log.Fatalf("Failed to create LLM1: %v", err)
	}

	llm2, err := createLLM(0.9, 0.5, 0.1, 1.5, 32, 2, 0.2, 4.0, 0.5, 123)
	if err != nil {
		log.Fatalf("Failed to create LLM2: %v", err)
	}

	prompt := gollm.NewPrompt("Explain the concept of quantum entanglement and its potential applications.")

	// Generate responses from both LLMs
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response from LLM1: %v", err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response from LLM2: %v", err)
	}

	// Print responses
	fmt.Println("Response from LLM1 (more conservative settings):")
	fmt.Println(response1)
	fmt.Println("\nResponse from LLM2 (more creative settings):")
	fmt.Println(response2)
}

func createLLM(temperature, topP, minP, repeatPenalty float64, repeatLastN int, mirostat int, mirostatEta, mirostatTau, tfsZ float64, seed int) (gollm.LLM, error) {
	return gollm.NewLLM(
		gollm.SetProvider("ollama"),
		gollm.SetModel("llama3.1"),
		gollm.SetOllamaEndpoint("http://localhost:11434"),
		gollm.SetTemperature(temperature),
		gollm.SetTopP(topP),
		gollm.SetMinP(minP),
		gollm.SetRepeatPenalty(repeatPenalty),
		gollm.SetRepeatLastN(repeatLastN),
		gollm.SetMirostat(mirostat),
		gollm.SetMirostatEta(mirostatEta),
		gollm.SetMirostatTau(mirostatTau),
		gollm.SetTfsZ(tfsZ),
		gollm.SetSeed(seed),
	)
}

