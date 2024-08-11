package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/teilomillet/gollm"
)

func main() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama-3.1-70b-versatile"),
		gollm.SetAPIKey(os.Getenv("GROQ_API_KEY")),
		gollm.SetMaxTokens(1024),
		gollm.SetDebugLevel(gollm.LogLevelWarn), // Set to Warn level
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	// Create the BatchPromptOptimizer
	optimizer := gollm.NewBatchPromptOptimizer(llm)
	optimizer.Verbose = true // Enable verbose output

	examples := []gollm.PromptExample{
		{
			Name:        "Creative Writing",
			Prompt:      "Write the opening paragraph of a mystery novel set in a small coastal town.",
			Description: "Create an engaging and atmospheric opening that hooks the reader",
			Threshold:   0.9,
			Metrics: []gollm.Metric{
				{Name: "Atmosphere", Description: "How well the writing evokes the setting"},
				{Name: "Intrigue", Description: "How effectively it sets up the mystery"},
				{Name: "Character Introduction", Description: "How well it introduces key characters"},
			},
		},
		{
			Name:        "Technical Documentation",
			Prompt:      "Explain how blockchain technology works to a non-technical audience.",
			Description: "Create a clear, accessible explanation without sacrificing accuracy",
			Threshold:   0.85,
			Metrics: []gollm.Metric{
				{Name: "Clarity", Description: "How easy it is for a layperson to understand"},
				{Name: "Accuracy", Description: "How technically correct the explanation is"},
				{Name: "Engagement", Description: "How interesting and relevant the explanation is"},
			},
		},
		{
			Name:        "Cross-cultural Communication",
			Prompt:      "Write a business proposal for a potential partner in Japan, suggesting a collaboration on a new eco-friendly technology.",
			Description: "Craft a culturally sensitive and effective business proposal",
			Threshold:   0.95,
			Metrics: []gollm.Metric{
				{Name: "Cultural Sensitivity", Description: "How well it respects Japanese business customs"},
				{Name: "Clarity of Proposition", Description: "How clearly the collaboration idea is presented"},
				{Name: "Persuasiveness", Description: "How compelling the proposal is"},
			},
		},
	}

	results := optimizer.OptimizePrompts(ctx, examples)

	// Process and display results
	for i, result := range results {
		fmt.Printf("\nResults for: %s (Example %d)\n", result.Name, i+1)
		fmt.Printf("Original Prompt: %s\n", result.OriginalPrompt)
		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
		} else {
			fmt.Printf("Optimized Prompt: %s\n", result.OptimizedPrompt)
			fmt.Printf("Generated Content:\n%s\n", result.GeneratedContent)
		}
		fmt.Println(strings.Repeat("=", 100))
	}
}
