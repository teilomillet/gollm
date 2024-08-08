// File: examples/prompt_optimizer_example.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
)

func main() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama-3.1-70b-versatile"),
		gollm.SetAPIKey(os.Getenv("GROQ_API_KEY")),
		gollm.SetMaxTokens(1024),
		gollm.SetDebugLevel(gollm.LogLevelWarn),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	examples := []struct {
		name         string
		prompt       string
		description  string
		metrics      []gollm.Metric
		ratingSystem string
		threshold    float64
	}{
		{
			name:         "Creative Writing (Numerical)",
			prompt:       "Write the opening paragraph of a mystery novel set in a small coastal town.",
			description:  "Create an engaging and atmospheric opening that hooks the reader",
			ratingSystem: "numerical",
			threshold:    0.9,
			metrics: []gollm.Metric{
				{Name: "Atmosphere", Description: "How well the writing evokes the setting"},
				{Name: "Intrigue", Description: "How effectively it sets up the mystery"},
				{Name: "Character Introduction", Description: "How well it introduces key characters"},
			},
		},
		{
			name:         "Technical Documentation (Letter)",
			prompt:       "Explain how blockchain technology works to a non-technical audience.",
			description:  "Create a clear, accessible explanation without sacrificing accuracy",
			ratingSystem: "letter",
			metrics: []gollm.Metric{
				{Name: "Clarity", Description: "How easy it is for a layperson to understand"},
				{Name: "Accuracy", Description: "How technically correct the explanation is"},
				{Name: "Engagement", Description: "How interesting and relevant the explanation is"},
			},
		},
		{
			name:         "Cross-cultural Communication (Numerical)",
			prompt:       "Write a business proposal for a potential partner in Japan, suggesting a collaboration on a new eco-friendly technology.",
			description:  "Craft a culturally sensitive and effective business proposal",
			ratingSystem: "numerical",
			threshold:    0.9,
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
		fmt.Printf("Rating System: %s\n", ex.ratingSystem)
		if ex.ratingSystem == "numerical" {
			fmt.Printf("Threshold: %.2f\n", ex.threshold)
		}

		opts := []gollm.OptimizerOption{
			gollm.WithCustomMetrics(ex.metrics...),
			gollm.WithRatingSystem(ex.ratingSystem),
			gollm.WithOptimizationGoal(fmt.Sprintf("Optimize the prompt for %s", ex.name)),
			gollm.WithMaxRetries(3),
			gollm.WithRetryDelay(time.Second * 2),
			gollm.WithIterationCallback(func(iteration int, entry gollm.OptimizationEntry) {
				fmt.Printf("\nIteration %d complete:\n", iteration)
				printIterationInfo(entry)
			}),
		}

		if ex.ratingSystem == "numerical" {
			opts = append(opts, gollm.WithThreshold(ex.threshold))
		}

		optimizer := gollm.NewPromptOptimizer(llm, ex.prompt, ex.description, opts...)

		fmt.Println("\nStarting prompt optimization...")
		fmt.Println("This may take a while. Please wait...")

		optimizedPrompt, err := optimizer.OptimizePrompt(ctx, 5) // 5 iterations
		if err != nil {
			log.Printf("Optimization error: %v", err)
			continue
		}

		fmt.Printf("\nFinal Optimized Prompt: %s\n", optimizedPrompt)

		fmt.Println("\nGenerating response with the optimized prompt...")
		response, err := llm.Generate(ctx, gollm.NewPrompt(optimizedPrompt))
		if err != nil {
			log.Printf("Failed to generate response: %v", err)
			continue
		}

		fmt.Printf("\nGenerated Response:\n%s\n\n", response)
		fmt.Printf("%s\n", strings.Repeat("=", 100))
	}
}

func printIterationInfo(entry gollm.OptimizationEntry) {
	fmt.Printf("  Prompt: %s\n", entry.Prompt.Input)
	fmt.Printf("  Overall Score: %.2f\n", entry.Assessment.OverallScore)
	fmt.Printf("  Overall Grade: %s\n", entry.Assessment.OverallGrade)
	fmt.Println("  Metrics:")
	for _, metric := range entry.Assessment.Metrics {
		fmt.Printf("    - %s: %.2f\n", metric.Name, metric.Value)
	}
	fmt.Println("  Strengths:")
	for _, strength := range entry.Assessment.Strengths {
		fmt.Printf("    - %s (Example: %s)\n", strength.Point, strength.Example)
	}
	fmt.Println("  Weaknesses:")
	for _, weakness := range entry.Assessment.Weaknesses {
		fmt.Printf("    - %s (Example: %s)\n", weakness.Point, weakness.Example)
	}
	fmt.Println("  Suggestions:")
	for _, suggestion := range entry.Assessment.Suggestions {
		fmt.Printf("    - %s (Expected Impact: %.2f, Reasoning: %s)\n", suggestion.Description, suggestion.ExpectedImpact, suggestion.Reasoning)
	}
	fmt.Printf("%s\n", strings.Repeat("-", 50))
}
