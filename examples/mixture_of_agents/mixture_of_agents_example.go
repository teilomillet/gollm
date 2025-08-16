package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
)

func main() {
	// Configure the MOA
	moaConfig := gollm.MOAConfig{
		Iterations: 2,
		Models: []gollm.ConfigOption{
			gollm.SetProvider("ollama"),
			gollm.SetModel("phi-3-medium-128k-instruct:Q8_0"),
			gollm.SetOllamaEndpoint(os.Getenv("OLLAMA_HOST")),
			gollm.SetMaxTokens(1024),
		},
		MaxParallel:  2,
		AgentTimeout: 30 * time.Second,
	}

	// Configure the aggregator
	aggregatorOpts := []gollm.ConfigOption{
		gollm.SetProvider("ollama"),
		gollm.SetModel("llama-3.1:8b-instruct-Q8_0"),
		gollm.SetOllamaEndpoint(os.Getenv("OLLAMA_HOST")),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1024),
		gollm.SetTimeout(45 * time.Second),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(2 * time.Second),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	}

	// Create the MOA
	moa, err := gollm.NewMOA(moaConfig, aggregatorOpts...)
	if err != nil {
		log.Fatalf("Failed to create MOA: %v", err)
	}

	// Use the MOA to generate a response
	ctx := context.Background()
	input := "Explain the concept of quantum entanglement and its potential applications in computing."
	output, err := moa.Generate(ctx, input)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	log.Printf("Input: %s\n\n", input)
	log.Printf("MOA Response:\n%s\n", output)
}
