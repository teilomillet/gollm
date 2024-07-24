// File: examples/3_compare_providers.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/goal"
)

func main() {
	fmt.Println("Starting the enhanced Goal library example...")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create LLM clients for different models
	llmGPT3, err := createLLM("openai", "gpt-4o-mini", apiKey)
	if err != nil {
		log.Fatalf("Failed to create GPT-4o-mini LLM client: %v", err)
	}

	llmGPT4, err := createLLM("openai", "gpt-4o", apiKey)
	if err != nil {
		log.Fatalf("Failed to create GPT-4o LLM client: %v", err)
	}

	ctx := context.Background()

	// Example 1: Basic Prompt with Comparison
	fmt.Println("\nExample 1: Basic Prompt with Comparison")
	basicPrompt := goal.NewPrompt("Explain the concept of machine learning in simple terms.")
	compareBasicPrompt(ctx, basicPrompt, llmGPT3, llmGPT4)

	// Example 2: Prompt with Directives and Output
	fmt.Println("\nExample 2: Prompt with Directives and Output")
	directivePrompt := goal.NewPrompt("Explain the concept of blockchain technology",
		goal.WithDirectives(
			"Use a simple analogy to illustrate",
			"Highlight key features",
			"Mention potential applications",
		),
		goal.WithOutput("Explanation of blockchain:"),
	)
	compareDirectivePrompt(ctx, directivePrompt, llmGPT3, llmGPT4)

	// Example 3: Prompt Template and JSON Schema
	fmt.Println("\nExample 3: Prompt Template and JSON Schema")
	templatePrompt := goal.NewPromptTemplate(
		"CustomAnalysis",
		"Analyze a given topic",
		"Analyze the following topic from multiple perspectives: {{.Topic}}",
		goal.WithPromptOptions(
			goal.WithDirectives(
				"Consider economic, social, and environmental impacts",
				"Provide pros and cons",
				"Conclude with a balanced summary",
			),
			goal.WithOutput("Analysis:"),
		),
	)

	// Generate JSON schema for prompts
	schemaBytes, err := llmGPT3.GetPromptJSONSchema()
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}
	fmt.Printf("JSON Schema for Prompts:\n%s\n", string(schemaBytes))

	// Execute the template and generate responses
	prompt, err := templatePrompt.Execute(map[string]interface{}{
		"Topic": "The adoption of autonomous vehicles",
	})
	if err != nil {
		log.Fatalf("Failed to execute prompt template: %v", err)
	}

	compareTemplatePrompt(ctx, prompt, llmGPT3, llmGPT4)

	fmt.Println("\nExample completed.")
}

func createLLM(provider, model, apiKey string) (goal.LLM, error) {
	return goal.NewLLM(
		goal.SetProvider(provider),
		goal.SetModel(model),
		goal.SetAPIKey(apiKey),
		goal.SetMaxTokens(300),
		goal.SetMaxRetries(3),
		goal.SetDebugLevel(goal.LogLevelInfo),
	)
}

func compareBasicPrompt(ctx context.Context, prompt *goal.Prompt, llm1, llm2 goal.LLM) {
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm1.GetProvider(), llm1.GetModel(), err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm2.GetProvider(), llm2.GetModel(), err)
	}

	fmt.Printf("%s %s Response:\n%s\n\n", llm1.GetProvider(), llm1.GetModel(), response1)
	fmt.Printf("%s %s Response:\n%s\n", llm2.GetProvider(), llm2.GetModel(), response2)
}

func compareDirectivePrompt(ctx context.Context, prompt *goal.Prompt, llm1, llm2 goal.LLM) {
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm1.GetProvider(), llm1.GetModel(), err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm2.GetProvider(), llm2.GetModel(), err)
	}

	fmt.Printf("%s %s Response:\n%s\n\n", llm1.GetProvider(), llm1.GetModel(), response1)
	fmt.Printf("%s %s Response:\n%s\n", llm2.GetProvider(), llm2.GetModel(), response2)
}

func compareTemplatePrompt(ctx context.Context, prompt *goal.Prompt, llm1, llm2 goal.LLM) {
	response1, err := llm1.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm1.GetProvider(), llm1.GetModel(), err)
	}

	response2, err := llm2.Generate(ctx, prompt)
	if err != nil {
		log.Printf("Failed to generate response from %s %s: %v", llm2.GetProvider(), llm2.GetModel(), err)
	}

	fmt.Printf("%s %s Response:\n%s\n\n", llm1.GetProvider(), llm1.GetModel(), response1)
	fmt.Printf("%s %s Response:\n%s\n", llm2.GetProvider(), llm2.GetModel(), response2)
}
