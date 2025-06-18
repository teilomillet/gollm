package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/guiperry/gollm_cerebras"
)

func main() {
	fmt.Println("Starting JSON handling examples...")

	// Setup LLM client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(200),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second*2),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	// Example 1: Simple JSON Output
	fmt.Println("\nExample 1: Simple JSON Output")
	fmt.Println("This example requests JSON output without enforcing a specific structure")

	simplePrompt := gollm.NewPrompt("List three colors",
		gollm.WithOutput("JSON array of colors"),
	)

	response, err := llm.Generate(ctx, simplePrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Printf("Simple JSON response:\n%s\n", response)

	// Example 2: Structured JSON with Schema Validation
	colorSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"colors": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"name": map[string]interface{}{"type": "string"},
						"hex":  map[string]interface{}{"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
					},
					"required": []string{"name", "hex"},
				},
			},
		},
		"required": []string{"colors"},
	}

	schemaPrompt := gollm.NewPrompt("List three colors with their hex codes")
	response, err = llm.GenerateWithSchema(ctx, schemaPrompt, colorSchema)
	if err != nil {
		log.Fatalf("Failed to generate with schema: %v", err)
	}
	fmt.Printf("Schema-validated response:\n%s\n", response)

	// Example 3: Complex Nested JSON Structure
	fmt.Println("\nExample 3: Complex Nested JSON Structure")
	fmt.Println("This example shows handling of nested JSON objects with strict validation")

	userSchema := `{
		"type": "object",
		"properties": {
			"user": {
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"age": {"type": "integer", "minimum": 0},
					"preferences": {
						"type": "object",
						"properties": {
							"favoriteColors": {
								"type": "array",
								"items": {"type": "string"}
							},
							"settings": {
								"type": "object",
								"properties": {
									"darkMode": {"type": "boolean"},
									"notifications": {"type": "boolean"}
								},
								"required": ["darkMode", "notifications"]
							}
						},
						"required": ["favoriteColors", "settings"]
					}
				},
				"required": ["name", "age", "preferences"]
			}
		},
		"required": ["user"]
	}`

	userPrompt := gollm.NewPrompt("Generate a user profile with preferences")
	response, err = llm.GenerateWithSchema(ctx, userPrompt, userSchema)
	if err != nil {
		log.Fatalf("Failed to generate complex JSON: %v", err)
	}
	fmt.Printf("Complex nested JSON response:\n%s\n", response)

	// Example 4: Mixed Format Response
	fmt.Println("\nExample 4: Mixed Format Response")
	fmt.Println("This example shows how to request specific sections in JSON format")

	mixedPrompt := gollm.NewPrompt("Analyze the color red",
		gollm.WithDirectives(
			"Provide a general description",
			"List color psychology effects",
			"Include common RGB and HEX values",
		),
		gollm.WithOutput(`Response should include a JSON object for technical details:
{
    "technical": {
        "rgb": [R, G, B],
        "hex": "string",
        "hsl": [H, S, L]
    }
}`),
	)

	response, err = llm.Generate(ctx, mixedPrompt)
	if err != nil {
		log.Fatalf("Failed to generate mixed format: %v", err)
	}
	fmt.Printf("Mixed format response:\n%s\n", response)

	fmt.Println("\nJSON examples completed.")
}
