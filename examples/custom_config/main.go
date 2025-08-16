// File: examples/enhanced_custom_config.go

package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
)

func main() {
	// Get the API_KEY
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a custom configuration
	customConfig := []gollm.ConfigOption{
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(150),
		gollm.SetTimeout(30 * time.Second),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(2 * time.Second),
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetAPIKey(apiKey),
	}

	// Create a new LLM instance with custom configuration
	llm, err := gollm.NewLLM(customConfig...)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Custom prompt template with formatting options
	analysisPrompt := gollm.NewPromptTemplate(
		"CustomAnalysis",
		"Analyze a given topic",
		"Analyze the following topic: {{.Topic}}",
		gollm.WithPromptOptions(
			gollm.WithDirectives(
				"Consider technological, economic, and social implications",
				"Provide at least one potential positive and one potential negative outcome",
				"Conclude with a balanced summary",
			),
			gollm.WithOutput("Analysis:"),
		),
	)

	topics := []string{
		"The widespread adoption of artificial intelligence",
		"The implementation of a four-day work week",
		"The transition to renewable energy sources",
	}

	for _, topic := range topics {
		prompt, err := analysisPrompt.Execute(map[string]any{
			"Topic": topic,
		})
		if err != nil {
			log.Printf("Failed to execute prompt template for topic '%s': %v\n", topic, err)
			continue
		}

		analysis, err := llm.Generate(ctx, prompt)
		if err != nil {
			log.Printf("Failed to generate analysis for topic '%s': %v\n", topic, err)
			continue
		}

		log.Printf("Topic: %s\nAnalysis:\n%s\n\n", topic, analysis.AsText())
	}

	// Demonstrate dynamic configuration changes
	llm.SetOption("temperature", 0.9)
	llm.SetOption("max_tokens", 200)

	// Get and print the current provider and model
	log.Printf("Current Provider: %s\n", llm.GetProvider())
	log.Printf("Current Model: %s\n", llm.GetModel())
}
