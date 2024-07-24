package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/goal"
)

func main() {
	fmt.Println("Starting the enhanced prompt types example...")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-3.5-turbo"),
		goal.SetAPIKey(apiKey),
		goal.SetMaxTokens(300),
		goal.SetMaxRetries(3),
		goal.SetDebugLevel(goal.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Example 1: Basic Prompt with Structured Output
	fmt.Println("\nExample 1: Basic Prompt with Structured Output")
	basicPrompt := goal.NewPrompt("List the top 3 benefits of exercise",
		goal.WithOutput("JSON array of benefits, each with a 'title' and 'description'"),
	)
	basicResponse, err := llm.Generate(ctx, basicPrompt)
	if err != nil {
		log.Printf("Failed to generate basic response: %v", err)
	} else {
		fmt.Printf("Basic Prompt Response:\n%s\n", basicResponse)
	}

	// Example 2: Prompt with Directives, Output, and Context
	fmt.Println("\nExample 2: Prompt with Directives, Output, and Context")
	directivePrompt := goal.NewPrompt("Propose a solution to reduce urban traffic congestion",
		goal.WithDirectives(
			"Consider both technological and policy-based approaches",
			"Address environmental concerns",
			"Consider cost-effectiveness",
		),
		goal.WithOutput("Solution proposal in markdown format with headings"),
		goal.WithContext("The city has a population of 2 million and limited public transportation."),
	)
	directiveResponse, err := llm.Generate(ctx, directivePrompt)
	if err != nil {
		log.Fatalf("Failed to generate directive response: %v", err)
	}
	fmt.Printf("Directive Prompt Response:\n%s\n", directiveResponse)

	// Example 3: Prompt with Examples and Max Length
	fmt.Println("\nExample 3: Prompt with Examples and Max Length")
	examplesPrompt := goal.NewPrompt("Write a short, engaging tweet about climate change",
		goal.WithExamples(
			"🌍 Small actions, big impact! Reduce, reuse, recycle to fight climate change. #ClimateAction",
			"🌡️ Climate change is real, and it's happening now. Let's act before it's too late! #ClimateEmergency",
		),
		goal.WithMaxLength(30),
	)
	examplesResponse, err := llm.Generate(ctx, examplesPrompt)
	if err != nil {
		log.Fatalf("Failed to generate examples response: %v", err)
	}
	fmt.Printf("Examples Prompt Response:\n%s\n", examplesResponse)

	// Example 4: Prompt Template with Dynamic Content
	fmt.Println("\nExample 4: Prompt Template with Dynamic Content")
	templatePrompt := goal.NewPromptTemplate(
		"ProductDescription",
		"Generate a product description",
		"Create an engaging product description for a {{.ProductType}} named '{{.ProductName}}'. "+
			"Target audience: {{.TargetAudience}}. Highlight {{.NumFeatures}} key features.",
		goal.WithPromptOptions(
			goal.WithDirectives(
				"Use persuasive language",
				"Include a call-to-action",
			),
			goal.WithOutput("Product description in HTML format"),
		),
	)

	prompt, err := templatePrompt.Execute(map[string]interface{}{
		"ProductType":    "smartwatch",
		"ProductName":    "TimeWise X1",
		"TargetAudience": "fitness enthusiasts",
		"NumFeatures":    3,
	})
	if err != nil {
		log.Fatalf("Failed to execute prompt template: %v", err)
	}

	templateResponse, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate template response: %v", err)
	}
	fmt.Printf("Template Prompt Response:\n%s\n", templateResponse)

	// Example 5: JSON Schema Generation and Validation
	fmt.Println("\nExample 5: JSON Schema Generation and Validation")
	schemaPrompt := goal.NewPrompt("Generate a user profile",
		goal.WithOutput(`JSON object with name, age, and interests`),
		goal.WithDirectives(
			"Name should be a string",
			"Age should be an integer",
			"Interests should be an array of strings",
		),
	)
	schemaBytes, err := schemaPrompt.GenerateJSONSchema()
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}
	fmt.Printf("JSON Schema for User Profile Prompt:\n%s\n", string(schemaBytes))

	// Demonstrate validation
	validPrompt := goal.NewPrompt("Valid prompt", goal.WithMaxLength(100))
	err = validPrompt.Validate()
	if err != nil {
		fmt.Printf("Unexpected validation error: %v\n", err)
	} else {
		fmt.Println("Valid prompt passed validation.")
	}

	invalidPrompt := goal.NewPrompt("", goal.WithMaxLength(-1))
	err = invalidPrompt.Validate()
	if err != nil {
		fmt.Printf("Validation error (expected): %v\n", err)
	}

	// Example 6: Chained Prompts
	fmt.Println("\nExample 6: Chained Prompts")
	ideaPrompt := goal.NewPrompt("Generate a unique business idea in the sustainability sector")
	ideaResponse, err := llm.Generate(ctx, ideaPrompt)
	if err != nil {
		log.Fatalf("Failed to generate idea: %v", err)
	}

	analysisPrompt := goal.NewPrompt(fmt.Sprintf("Analyze the following business idea: %s", ideaResponse),
		goal.WithDirectives(
			"Identify potential challenges",
			"Suggest target market",
			"Propose a monetization strategy",
		),
		goal.WithOutput("Analysis in JSON format with 'challenges', 'targetMarket', and 'monetization' keys"),
	)
	analysisResponse, err := llm.Generate(ctx, analysisPrompt)
	if err != nil {
		log.Fatalf("Failed to generate analysis: %v", err)
	}

	fmt.Printf("Chained Prompts Response:\nIdea: %s\nAnalysis: %s\n", ideaResponse, analysisResponse)

	// Example 7: Prompt with JSON Schema Validation
	fmt.Println("\nExample 7: Prompt with JSON Schema Validation")
	jsonSchemaPrompt := goal.NewPrompt("Generate a user profile",
		goal.WithOutput(`JSON object with the following schema:
		{
			"type": "object",
			"properties": {
				"name": {"type": "string"},
				"age": {"type": "integer", "minimum": 18},
				"interests": {"type": "array", "items": {"type": "string"}}
			},
			"required": ["name", "age", "interests"]
		}`),
	)
	jsonSchemaResponse, err := llm.Generate(ctx, jsonSchemaPrompt, goal.WithJSONSchemaValidation())
	if err != nil {
		log.Fatalf("Failed to generate JSON schema validated response: %v", err)
	}

	var userProfile map[string]interface{}
	err = json.Unmarshal([]byte(jsonSchemaResponse), &userProfile)
	if err != nil {
		log.Fatalf("Failed to parse JSON response: %v", err)
	}
	fmt.Printf("JSON Schema Validated Response:\n%+v\n", userProfile)

	fmt.Println("\nExample completed.")
}

