// File: examples/prompt_optimizer_example.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"
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

	example := gollm.PromptExample{
		Name:        "Creative Writing",
		Prompt:      "Write the opening paragraph of a mystery novel set in a small coastal town.",
		Description: "Create an engaging and atmospheric opening that hooks the reader",
		Threshold:   0.9,
		Metrics: []gollm.Metric{
			{Name: "Atmosphere", Description: "How well the writing evokes the setting"},
			{Name: "Intrigue", Description: "How effectively it sets up the mystery"},
			{Name: "Character Introduction", Description: "How well it introduces key characters"},
		},
	}

	fmt.Printf("Optimizing prompt for: %s\n", example.Name)
	fmt.Printf("Initial Prompt: %s\n", example.Prompt)

	optimizer := gollm.NewPromptOptimizer(llm, example.Prompt, example.Description,
		gollm.WithCustomMetrics(example.Metrics...),
		gollm.WithRatingSystem("numerical"),
		gollm.WithThreshold(example.Threshold),
		gollm.WithMaxRetries(3),
		gollm.WithRetryDelay(time.Second*2),
		gollm.WithVerbose(),
	)

	optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
	if err != nil {
		log.Fatalf("Optimization error: %v", err)
	}

	fmt.Printf("\nOptimized Prompt: %s\n", optimizedPrompt)

	response, err := llm.Generate(ctx, gollm.NewPrompt(optimizedPrompt))
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("\nGenerated Content:\n%s\n", response)
}
