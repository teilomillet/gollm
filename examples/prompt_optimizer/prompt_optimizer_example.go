package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/optimizer"
	"github.com/teilomillet/gollm/utils"
)

func main() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama-3.1-70b-versatile"),
		gollm.SetAPIKey(os.Getenv("GROQ_API_KEY")),
		gollm.SetMaxTokens(1024),
		gollm.SetLogLevel(gollm.LogLevelWarn),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	example := optimizer.PromptExample{
		Name:        "Creative Writing",
		Prompt:      "Write the opening paragraph of a mystery novel set in a small coastal town.",
		Description: "Create an engaging and atmospheric opening that hooks the reader",
		Threshold:   0.9,
		Metrics: []optimizer.Metric{
			{Name: "Atmosphere", Description: "How well the writing evokes the setting"},
			{Name: "Intrigue", Description: "How effectively it sets up the mystery"},
			{Name: "Character Introduction", Description: "How well it introduces key characters"},
		},
	}

	log.Printf("Optimizing prompt for: %s\n", example.Name)
	log.Printf("Initial Prompt: %s\n", example.Prompt)

	debugManager := utils.NewDebugManager(llm.GetLogger(), utils.DebugOptions{LogPrompts: true, LogResponses: true})
	initialPrompt := llm.NewPrompt(example.Prompt)

	optimizerInstance := optimizer.NewPromptOptimizer(
		llm,
		debugManager,
		initialPrompt,
		example.Description,
		optimizer.WithCustomMetrics(example.Metrics...),
		optimizer.WithRatingSystem("numerical"),
		optimizer.WithThreshold(example.Threshold),
		optimizer.WithMaxRetries(3),
		optimizer.WithRetryDelay(time.Second*2),
	)

	optimizedPrompt, err := optimizerInstance.OptimizePrompt(ctx)
	if err != nil {
		log.Fatalf("Optimization error: %v", err)
	}

	log.Printf("\nOptimized Prompt: %s\n", optimizedPrompt.Input)

	response, err := llm.Generate(ctx, optimizedPrompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	log.Printf("\nGenerated Content:\n%s\n", response.AsText())
}
