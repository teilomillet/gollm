// File: examples/3_compare_providers.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	log.Println("Starting the enhanced GOLLM library example...")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create LLM clients for different models
	llmGPT3, err := createLLM("gpt-4o-mini", apiKey)
	if err != nil {
		log.Fatalf("Failed to create GPT-4o-mini LLM client: %v", err)
	}

	llmGPT4, err := createLLM("gpt-4o", apiKey)
	if err != nil {
		log.Fatalf("Failed to create GPT-4o LLM client: %v", err)
	}

	ctx := context.Background()

	// Example 1: Basic Prompt with Comparison
	log.Println("\nExample 1: Basic Prompt with Comparison")
	basicPrompt := gollm.NewPrompt("Explain the concept of machine learning in simple terms.")
	compareBasicPrompt(ctx, basicPrompt, llmGPT3, llmGPT4)

	// Example 2: Prompt with Directives and Output
	log.Println("\nExample 2: Prompt with Directives and Output")
	directivePrompt := gollm.NewPrompt("Explain the concept of blockchain technology",
		gollm.WithDirectives(
			"Use a simple analogy to illustrate",
			"Highlight key features",
			"Mention potential applications",
		),
		gollm.WithOutput("Explanation of blockchain:"),
	)
	compareDirectivePrompt(ctx, directivePrompt, llmGPT3, llmGPT4)

	// Example 3: Prompt Template and JSON Schema
	log.Println("\nExample 3: Prompt Template and JSON Schema")
	templatePrompt := gollm.NewPromptTemplate(
		"CustomAnalysis",
		"Analyze a given topic",
		"Analyze the following topic from multiple perspectives: {{.Topic}}",
		gollm.WithPromptOptions(
			gollm.WithDirectives(
				"Consider economic, social, and environmental impacts",
				"Provide pros and cons",
				"Conclude with a balanced summary",
			),
			gollm.WithOutput("Analysis:"),
		),
	)

	// Generate JSON schema for prompts
	schemaBytes, err := llmGPT3.GetPromptJSONSchema()
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}
	log.Printf("JSON Schema for Prompts:\n%s\n", string(schemaBytes))

	// Execute the template and generate responses
	prompt, err := templatePrompt.Execute(map[string]any{
		"Topic": "The adoption of autonomous vehicles",
	})
	if err != nil {
		log.Fatalf("Failed to execute prompt template: %v", err)
	}

	compareTemplatePrompt(ctx, prompt, llmGPT3, llmGPT4)

	log.Println("\nExample completed.")
}

func createLLM(model, apiKey string) (gollm.LLM, error) {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel(model),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM: %w", err)
	}
	return llm, nil
}

func compareBasicPrompt(ctx context.Context, prompt *gollm.Prompt, llm1, llm2 gollm.LLM) {
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm1.GetProvider(), llm1.GetModel(), err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm2.GetProvider(), llm2.GetModel(), err)
	}

	log.Printf("%s %s Response:\n%s\n\n", llm1.GetProvider(), llm1.GetModel(), response1.AsText())
	log.Printf("%s %s Response:\n%s\n", llm2.GetProvider(), llm2.GetModel(), response2.AsText())
}

func compareDirectivePrompt(ctx context.Context, prompt *gollm.Prompt, llm1, llm2 gollm.LLM) {
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm1.GetProvider(), llm1.GetModel(), err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm2.GetProvider(), llm2.GetModel(), err)
	}

	log.Printf("%s %s Response:\n%s\n\n", llm1.GetProvider(), llm1.GetModel(), response1.AsText())
	log.Printf("%s %s Response:\n%s\n", llm2.GetProvider(), llm2.GetModel(), response2.AsText())
}

func compareTemplatePrompt(ctx context.Context, prompt *gollm.Prompt, llm1, llm2 gollm.LLM) {
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm1.GetProvider(), llm1.GetModel(), err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm2.GetProvider(), llm2.GetModel(), err)
	}

	log.Printf("%s %s Response:\n%s\n\n", llm1.GetProvider(), llm1.GetModel(), response1.AsText())
	log.Printf("%s %s Response:\n%s\n", llm2.GetProvider(), llm2.GetModel(), response2.AsText())
}
