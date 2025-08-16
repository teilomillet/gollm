package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
)

func main() {
	log.Println("Starting JSON handling examples...")

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
	log.Println("\nExample 1: Simple JSON Output")
	log.Println("This example requests JSON output without enforcing a specific structure")

	simplePrompt := gollm.NewPrompt("List three colors",
		gollm.WithOutput("JSON array of colors"),
	)

	response, err := llm.Generate(ctx, simplePrompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	log.Printf("Simple JSON response:\n%s\n", response.AsText())

	// Example 2: Structured JSON with Schema Validation
	colorSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"colors": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"hex":  map[string]any{"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
					},
					"required": []string{"name", "hex"},
				},
			},
		},
		"required": []string{"colors"},
	}

	schemaPrompt := gollm.NewPrompt("List three colors with their hex codes")
	response, err = llm.Generate(ctx, schemaPrompt, gollm.WithStructuredResponse(colorSchema))
	if err != nil {
		log.Fatalf("Failed to generate with schema: %v", err)
	}
	log.Printf("Schema-validated response:\n%s\n", response.AsText())

	// Example 3: Complex Nested JSON Structure
	log.Println("\nExample 3: Complex Nested JSON Structure")
	log.Println("This example shows handling of nested JSON objects with strict validation")

	userPrompt := gollm.NewPrompt("Generate a user profile with preferences")
	response, err = llm.Generate(ctx, userPrompt, gollm.WithStructuredResponseSchema[UserEnvelope]())
	if err != nil {
		log.Fatalf("Failed to generate complex JSON: %v", err)
	}
	log.Printf("Complex nested JSON response:\n%s\n", response.AsText())

	// Example 4: Mixed Format Response
	log.Println("\nExample 4: Mixed Format Response")
	log.Println("This example shows how to request specific sections in JSON format")

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
	log.Printf("Mixed format response:\n%s\n", response.AsText())

	log.Println("\nJSON examples completed.")
}

// UserEnvelope is the top-level object with a required "user" field.
type UserEnvelope struct {
	User User `json:"user"`
}

// User corresponds to the schema's "user" object.
// Required fields: name, age, preferences.
type User struct {
	Name        string      `json:"name"`
	Preferences Preferences `json:"preferences"`
	Age         int         `json:"age"`
}

// Preferences corresponds to the "preferences" object.
// Required fields: favoriteColors, settings.
type Preferences struct {
	FavoriteColors []string `json:"favoriteColors"`
	Settings       Settings `json:"settings"`
}

// Settings corresponds to the "settings" object.
// Required fields: darkMode, notifications.
type Settings struct {
	DarkMode      bool `json:"darkMode"`
	Notifications bool `json:"notifications"`
}
