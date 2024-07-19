// File: examples/5_advanced_prompt.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal/llm"
)

func main() {
	config, err := llm.LoadConfig("")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	llmClient, err := llm.NewLLMFromConfig(config)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	prompt := llm.NewPrompt("Analyze the following statement:").
		WithDirective("Consider both positive and negative implications").
		WithDirective("Use a formal, academic tone").
		WithDirective("Provide at least three key points").
		WithOutput("Analysis:")

	statement := "The widespread adoption of artificial intelligence in various industries."
	fullPrompt := fmt.Sprintf("%s\n\n%s", prompt.String(), statement)

	response, _, err := llmClient.Generate(context.Background(), fullPrompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Advanced Analysis:\n%s\n", response)
}
