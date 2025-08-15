package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/teilomillet/gollm"
)

type PersonInfo struct {
	Name       string   `json:"name" validate:"required"`
	Age        int      `json:"age" validate:"required,gte=0,lte=150"`
	Occupation string   `json:"occupation" validate:"required"`
	Hobbies    []string `json:"hobbies" validate:"required,min=1,max=5"`
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second*2),
		gollm.SetLogLevel(gollm.LogLevelWarn),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	prompt := gollm.NewPrompt(
		"Generate information about a fictional person",
		gollm.WithDirectives(
			"Create a fictional person with a name, age, occupation, and hobbies",
			"Ensure the age is realistic for the occupation",
			"Include 1 to 5 hobbies",
			"Return ONLY the JSON data for the person, not the schema",
		),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt, gollm.WithStructuredResponseSchema[PersonInfo]())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}

	fmt.Printf("Generated PersonInfo:\n%s\n", response.AsText())

	var person PersonInfo
	err = json.Unmarshal([]byte(response.AsText()), &person)
	if err != nil {
		log.Fatalf("Failed to parse response as JSON: %v", err)
	}

	if err := gollm.Validate(&person); err != nil {
		log.Fatalf("Generated data does not match schema: %v", err)
	}

	fmt.Println("\nValidated PersonInfo:")
	fmt.Printf("Name: %s\n", person.Name)
	fmt.Printf("Age: %d\n", person.Age)
	fmt.Printf("Occupation: %s\n", person.Occupation)
	fmt.Printf("Hobbies: %v\n", person.Hobbies)
}
