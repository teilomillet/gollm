// File: examples/enhanced_custom_config.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/guiperry/gollm_cerebras"
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
		prompt, err := analysisPrompt.Execute(map[string]interface{}{
			"Topic": topic,
		})
		if err != nil {
			log.Printf("Failed to execute prompt template for topic '%s': %v\n", topic, err)
			continue
		}

		// Use JSON schema validation for the prompt
		analysis, err := llm.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
		if err != nil {
			log.Printf("Failed to generate analysis for topic '%s': %v\n", topic, err)
			continue
		}

		fmt.Printf("Topic: %s\nAnalysis:\n%s\n\n", topic, analysis)
	}

	// Demonstrate dynamic configuration changes
	llm.SetOption("temperature", 0.9)
	llm.SetOption("max_tokens", 200)

	// Get and print the current provider and model
	fmt.Printf("Current Provider: %s\n", llm.GetProvider())
	fmt.Printf("Current Model: %s\n", llm.GetModel())

}
