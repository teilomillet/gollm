// File: examples/4_custom_config.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	llm, err := goal.NewLLM()
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Custom prompt template
	analysisPrompt := goal.NewPromptTemplate(
		"CustomAnalysis",
		"Analyze a given topic",
		"Analyze the following topic: {{.Topic}}",
		goal.WithPromptOptions(
			goal.WithDirectives(
				"Consider technological, economic, and social implications",
				"Provide at least one potential positive and one potential negative outcome",
				"Conclude with a balanced summary",
			),
			goal.WithOutput("Analysis:"),
		),
	)

	topics := []string{
		"The widespread adoption of artificial intelligence",
		"The implementation of a four-day work week",
		"The transition to renewable energy sources",
	}

	for _, topic := range topics {
		prompt, err := analysisPrompt.Execute(map[string]interface{}{
			"Topic": topic,
		})
		if err != nil {
			log.Printf("Failed to execute prompt template for topic '%s': %v\n", topic, err)
			continue
		}

		analysis, _, err := llm.Generate(ctx, prompt.String())
		if err != nil {
			log.Printf("Failed to generate analysis for topic '%s': %v\n", topic, err)
			continue
		}

		fmt.Printf("Topic: %s\nAnalysis:\n%s\n\n", topic, analysis)
	}
}
