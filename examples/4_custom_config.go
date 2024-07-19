// File: examples/4_custom_config.go

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

	// Custom prompt template
	analysisPrompt := goal.NewPrompt("Analyze the following topic:").
		Directive("Consider technological, economic, and social implications").
		Directive("Provide at least one potential positive and one potential negative outcome").
		Directive("Conclude with a balanced summary").
		Output("Analysis:")

	topics := []string{
		"The widespread adoption of artificial intelligence",
		"The implementation of a four-day work week",
		"The transition to renewable energy sources",
	}

	for _, topic := range topics {
		fullPrompt := analysisPrompt.Input(topic)
		analysis, _, err := llm.Generate(ctx, fullPrompt.String())
		if err != nil {
			log.Printf("Failed to generate analysis for topic '%s': %v\n", topic, err)
			continue
		}

		fmt.Printf("Topic: %s\nAnalysis:\n%s\n\n", topic, analysis)
	}
}
