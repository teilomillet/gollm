package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/teilomillet/goal"
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

	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-3.5-turbo"),
		goal.SetAPIKey(apiKey),
		goal.SetMaxTokens(300),
		goal.SetMaxRetries(3),
		goal.SetRetryDelay(time.Second*2),
		goal.SetDebugLevel(goal.LogLevelWarn),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	schemaJSON, err := goal.GenerateJSONSchema(&PersonInfo{}, goal.WithExpandedStruct(true))
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}

	prompt := goal.NewPrompt(
		"Generate information about a fictional person",
		goal.WithDirectives(
			"Create a fictional person with a name, age, occupation, and hobbies",
			"Ensure the age is realistic for the occupation",
			"Include 1 to 5 hobbies",
			"Return ONLY the JSON data for the person, not the schema",
		),
		goal.WithOutput(fmt.Sprintf("Generate a JSON object that adheres to this schema:\n%s\nDo not include the schema in your response, only the generated data.", string(schemaJSON))),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt, goal.WithJSONSchemaValidation())
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}

	fmt.Printf("Generated PersonInfo:\n%s\n", response)

	var person PersonInfo
	err = json.Unmarshal([]byte(response), &person)
	if err != nil {
		log.Fatalf("Failed to parse response as JSON: %v", err)
	}

	if err := goal.Validate(&person); err != nil {
		log.Fatalf("Generated data does not match schema: %v", err)
	}

	fmt.Println("\nValidated PersonInfo:")
	fmt.Printf("Name: %s\n", person.Name)
	fmt.Printf("Age: %d\n", person.Age)
	fmt.Printf("Occupation: %s\n", person.Occupation)
	fmt.Printf("Hobbies: %v\n", person.Hobbies)
}
