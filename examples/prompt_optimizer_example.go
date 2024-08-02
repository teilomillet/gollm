// File: examples/prompt_optimizer_example.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	// Create a new LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(200),
		gollm.SetDebugLevel(gollm.LogLevelDebug),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a PromptOptimizer
	initialPrompt := "Write a tweet about climate change."
	taskDescription := "Generate an informative and engaging tweet about climate change that includes a call to action."
	optimizer := gollm.NewPromptOptimizer(llm, initialPrompt, taskDescription)

	// Run the optimization process
	ctx := context.Background()
	optimizedPrompt, err := optimizer.OptimizePrompt(ctx, 3) // Run for 3 iterations
	if err != nil {
		log.Printf("Optimization failed: %v", err)
		// Even if optimization fails, we can still print the history of attempts
	} else {
		fmt.Printf("Optimized prompt: %s\n", optimizedPrompt)
	}

	// Print optimization history
	fmt.Println("\nOptimization History:")
	history := optimizer.GetOptimizationHistory()
	if len(history) == 0 {
		fmt.Println("No optimization history available.")
	} else {
		for i, entry := range history {
			fmt.Printf("Iteration %d:\n", i+1)
			fmt.Printf("  Prompt: %s\n", entry.Prompt)
			fmt.Printf("  Overall Score: %.1f\n", entry.Assessment.OverallScore)
			fmt.Printf("  Clarity: %.1f\n", entry.Assessment.Clarity)
			fmt.Printf("  Relevance: %.1f\n", entry.Assessment.Relevance)
			fmt.Printf("  Effectiveness: %.1f\n", entry.Assessment.Effectiveness)
			fmt.Printf("  Strengths: %v\n", entry.Assessment.Strengths)
			fmt.Printf("  Weaknesses: %v\n", entry.Assessment.Weaknesses)
			fmt.Printf("  Improvements: %v\n\n", entry.Assessment.Improvements)
		}
	}

	// Use the last prompt (optimized or not) to generate a response
	lastPrompt := initialPrompt
	if len(history) > 0 {
		lastPrompt = history[len(history)-1].Prompt
	}

	prompt := gollm.NewPrompt(lastPrompt)
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Generated tweet: %s\n", response)
}

