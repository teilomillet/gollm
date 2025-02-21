package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/mauza/gollm"
)

func main() {
	// Initialize LLM client
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),                  // Set the LLM provider
		gollm.SetModel("gpt-4o-mini"),                // Set the model (ensure it's compatible with the provider)
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")), // Set the API Key
		gollm.SetMaxTokens(500))                      // Limit the response length
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	// Step 1: Research phase
	// Generate a brief overview of the topic to use as context for later steps
	researchPrompt := gollm.NewPrompt(
		"Provide a brief overview of quantum computing",
		gollm.WithMaxLength(200), // Limit the research to 200 words
	)
	research, err := llm.Generate(ctx, researchPrompt)
	if err != nil {
		log.Fatalf("Research failed: %v", err)
	}
	fmt.Printf("Research:\n%s\n\n", research)

	// Step 2: Ideation phase
	// Generate article ideas based on the research
	ideaPrompt := gollm.NewPrompt(
		"Generate 3 article ideas about quantum computing for a general audience",
		gollm.WithContext(research), // Use the research as context for generating ideas
	)
	ideas, err := llm.Generate(ctx, ideaPrompt)
	if err != nil {
		log.Fatalf("Ideation failed: %v", err)
	}
	fmt.Printf("Article Ideas:\n%s\n\n", ideas)

	// Step 3: Writing refinement
	// Improve a paragraph using specific directives
	refinementPrompt := gollm.NewPrompt(
		"Improve the following paragraph about quantum computing:",
		gollm.WithContext(research), // Use the research as the paragraph to improve
		gollm.WithDirectives( // Provide specific instructions for improvement
			"Use simpler language for a general audience",
			"Add an engaging opening sentence",
			"Conclude with a thought-provoking question",
		),
	)
	refinedParagraph, err := llm.Generate(ctx, refinementPrompt)
	if err != nil {
		log.Fatalf("Refinement failed: %v", err)
	}
	fmt.Printf("Refined Paragraph:\n%s\n", refinedParagraph)
}
