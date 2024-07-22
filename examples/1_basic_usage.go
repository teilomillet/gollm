package main

import (
	"context"
	"fmt"
	"log"

	"github.com/joho/godotenv"
	"github.com/teilomillet/goal"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: Error loading .env file")
	}

	cfg := goal.NewConfigBuilder().
		SetProvider("openai").
		SetModel("gpt-3.5-turbo").
		SetMaxTokens(100).
		SetAPIKey("your-api-key-here"). // Replace with your actual API key
		Build()

	llm, err := goal.NewLLM(cfg)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	// Example 1: Using a basic prompt with functional options
	basicPrompt := goal.NewPrompt("Tell me a short joke about programming.",
		goal.WithMaxLength(50),
	)
	response, _, err := llm.Generate(ctx, basicPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Basic prompt response: %s\n\n", response)

	// Example 2: Using a prompt with directives and output specification
	advancedPrompt := goal.NewPrompt("Describe the benefits of using Go for web development",
		goal.WithDirectives("List at least three key advantages"),
		goal.WithOutput("Benefits of Go for web development:"),
	)

	response, _, err = llm.Generate(ctx, advancedPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Advanced prompt response:\n%s\n\n", response)

	// Example 3: Using a PromptTemplate
	haikuTemplate := &goal.PromptTemplate{
		Name:        "HaikuTemplate",
		Description: "Generate a haiku about a given topic",
		Template:    "Write a haiku about {{.Topic}}",
		Options: []goal.PromptOption{
			goal.WithDirectives("Follow the 5-7-5 syllable pattern"),
			goal.WithOutput("Haiku:"),
		},
	}

	haikuPrompt, err := haikuTemplate.Execute(map[string]interface{}{
		"Topic": "Go programming",
	})
	if err != nil {
		log.Fatalf("Failed to execute haiku template: %v", err)
	}

	response, _, err = llm.Generate(ctx, haikuPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Haiku response:\n%s\n", response)
}
