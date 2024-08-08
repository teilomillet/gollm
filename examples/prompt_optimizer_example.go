// File: examples/prompt_optimizer_example.go

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
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(512),
		gollm.SetDebugLevel(gollm.LogLevelWarn),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	examples := []struct {
		name        string
		prompt      string
		description string
		metrics     []gollm.Metric
	}{
		{
			name:        "Creative Writing",
			prompt:      "Write the opening paragraph of a mystery novel set in a small coastal town.",
			description: "Create an engaging and atmospheric opening that hooks the reader",
			metrics: []gollm.Metric{
				{Name: "Atmosphere", Description: "How well the writing evokes the setting"},
				{Name: "Intrigue", Description: "How effectively it sets up the mystery"},
				{Name: "Character Introduction", Description: "How well it introduces key characters"},
			},
		},
		{
			name:        "Technical Documentation",
			prompt:      "Explain how blockchain technology works to a non-technical audience.",
			description: "Create a clear, accessible explanation without sacrificing accuracy",
			metrics: []gollm.Metric{
				{Name: "Clarity", Description: "How easy it is for a layperson to understand"},
				{Name: "Accuracy", Description: "How technically correct the explanation is"},
				{Name: "Engagement", Description: "How interesting and relevant the explanation is"},
			},
		},
		{
			name:        "Cross-cultural Communication",
			prompt:      "Write a business proposal for a potential partner in Japan, suggesting a collaboration on a new eco-friendly technology.",
			description: "Craft a culturally sensitive and effective business proposal",
			metrics: []gollm.Metric{
				{Name: "Cultural Sensitivity", Description: "How well it respects Japanese business customs"},
				{Name: "Clarity of Proposition", Description: "How clearly the collaboration idea is presented"},
				{Name: "Persuasiveness", Description: "How compelling the proposal is"},
			},
		},
	}

	for _, ex := range examples {
		fmt.Printf("\nExample: %s\n", ex.name)
		fmt.Printf("Initial Prompt: %s\n", ex.prompt)

		optimizePrompt(ctx, llm, ex.prompt, ex.description,
			gollm.WithCustomMetrics(ex.metrics...),
			gollm.WithRatingSystem("numerical"),
			gollm.WithOptimizationGoal(fmt.Sprintf("Optimize the prompt for %s", ex.name)),
		)
	}
}

func optimizePrompt(ctx context.Context, llm gollm.LLM, initialPrompt, taskDesc string, opts ...gollm.OptimizerOption) {
	optimizer := gollm.NewPromptOptimizer(llm, initialPrompt, taskDesc, opts...)

	optimizedPrompt, err := optimizer.OptimizePrompt(ctx, 5) // Increased to 5 iterations
	if err != nil {
		log.Printf("Optimization failed: %v", err)
		fmt.Println("Optimization History (including failed attempts):")
		printOptimizationHistory(optimizer.GetOptimizationHistory())
		return
	}

	fmt.Printf("Optimized Prompt: %s\n", optimizedPrompt)

	fmt.Println("\nOptimization History:")
	printOptimizationHistory(optimizer.GetOptimizationHistory())

	response, err := llm.Generate(ctx, gollm.NewPrompt(optimizedPrompt))
	if err != nil {
		log.Printf("Failed to generate response: %v", err)
		return
	}

	fmt.Printf("Generated Response:\n%s\n\n", response)
	fmt.Printf("%s\n", strings.Repeat("=", 100))
}

func printOptimizationHistory(history []gollm.OptimizationEntry) {
	for i, entry := range history {
		fmt.Printf("Iteration %d:\n", i+1)
		fmt.Printf("  Prompt: %s\n", entry.Prompt.Input)
		fmt.Printf("  Overall Score: %.2f\n", entry.Assessment.OverallScore)
		fmt.Println("  Metrics:")
		for _, metric := range entry.Assessment.Metrics {
			fmt.Printf("    - %s: %.2f\n", metric.Name, metric.Value)
		}
		fmt.Printf("  Strengths: %v\n", entry.Assessment.Strengths)
		fmt.Printf("  Weaknesses: %v\n", entry.Assessment.Weaknesses)
		fmt.Printf("  Suggestions: %v\n\n", entry.Assessment.Suggestions)
		fmt.Printf("%s\n", strings.Repeat("-", 50))
	}
}
