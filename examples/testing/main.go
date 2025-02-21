package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/mauza/gollm"
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

	// Process LLM1
	fmt.Println("Generating response from LLM1 (more conservative settings):")
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response from LLM1: %v", err)
	}
	fmt.Println(response1)

	getUserFeedback("LLM1")

	// Process LLM2
	fmt.Println("\nGenerating response from LLM2 (more creative settings):")
	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response from LLM2: %v", err)
	}
	fmt.Println(response2)

	getUserFeedback("LLM2")
}

func createLLM(temperature, topP, minP, repeatPenalty float64, repeatLastN int, mirostat int, mirostatEta, mirostatTau, tfsZ float64, seed int) (gollm.LLM, error) {
	return gollm.NewLLM(
		gollm.SetProvider("ollama"),
		gollm.SetModel("llama3.1"),
		gollm.SetEndpoint("http://localhost:11434"),
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

func getUserFeedback(llmName string) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("\nPlease rate the response from %s (1-5): ", llmName)
	rating, _ := reader.ReadString('\n')
	rating = strings.TrimSpace(rating)

	fmt.Print("Any additional comments? ")
	comments, _ := reader.ReadString('\n')
	comments = strings.TrimSpace(comments)

	fmt.Printf("Feedback for %s - Rating: %s, Comments: %s\n\n", llmName, rating, comments)
}
