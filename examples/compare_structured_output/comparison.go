package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/presets"
)

type ComplexPerson struct {
	Name          string   `json:"name" validate:"required"`
	Occupation    string   `json:"occupation" validate:"required"`
	City          string   `json:"city" validate:"required"`
	Country       string   `json:"country" validate:"required"`
	FavoriteColor string   `json:"favoriteColor" validate:"required"`
	Education     string   `json:"education" validate:"required"`
	PetName       string   `json:"petName" validate:"required"`
	Hobbies       []string `json:"hobbies" validate:"required,min=1,max=5"`
	Age           int      `json:"age" validate:"required,gte=0,lte=150"`
	LuckyNumber   int      `json:"luckyNumber" validate:"required,gte=1,lte=100"`
}

var debugLevel = gollm.LogLevelWarn // Set to LogLevelDebug for verbose output

func debugLog(format string, args ...any) {
	if debugLevel == gollm.LogLevelDebug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

func main() {
	ctx := context.Background()

	fmt.Println("Starting structured output comparison...")

	// Define models to compare
	models := []struct {
		provider string
		model    string
	}{
		{"openai", "gpt-4o-mini"},
		{"openai", "gpt-4o"},
		{"anthropic", "claude-3-haiku-20240307"},
		{"anthropic", "claude-3-5-sonnet-20240620"},
	}

	// Create configs for each model
	configs := make([]*gollm.Config, 0, len(models))
	for _, m := range models {
		apiKeyEnv := strings.ToUpper(m.provider) + "_API_KEY"
		apiKey := os.Getenv(apiKeyEnv)
		if apiKey == "" {
			fmt.Printf(
				"Skipping %s %s: API key not set. Please set %s environment variable.\n",
				m.provider,
				m.model,
				apiKeyEnv,
			)
			continue
		}

		config := gollm.NewConfig() // Use NewConfig to create a properly initialized Config struct
		gollm.SetProvider(m.provider)(config)
		gollm.SetModel(m.model)(config)
		gollm.SetAPIKey(apiKey)(config)
		gollm.SetMaxTokens(500)(config)
		gollm.SetMaxRetries(3)(config)
		gollm.SetRetryDelay(time.Second * 2)(config)
		gollm.SetLogLevel(debugLevel)(config)

		configs = append(configs, config)
		debugLog("Created configuration for %s %s", m.provider, m.model)
	}

	if len(configs) == 0 {
		log.Fatalf("No valid configurations created. Please check your API keys.")
	}

	debugLog("Created %d valid configurations", len(configs))

	// Generate JSON schema for ComplexPerson
	schema, err := gollm.GenerateJSONSchema(ComplexPerson{})
	if err != nil {
		log.Fatalf("Failed to generate JSON schema: %v", err)
	}

	debugLog("Generated JSON schema for ComplexPerson")

	// Create prompt for generating ComplexPerson data
	promptText := "Generate information about a fictional person.\nCreate a fictional person with the following attributes: name, age, occupation, city, country, favorite color, hobbies (1-5), education, pet name, and lucky number (1-100).\nEnsure all fields are filled and adhere to the specified constraints.\nReturn the data as a JSON object that adheres to this schema:\n" + string(
		schema,
	)

	debugLog("Created prompt for generating ComplexPerson data")

	// Define a validation function for ComplexPerson
	validateComplexPerson := func(person ComplexPerson) error {
		// Add any additional validation logic here
		if person.Age < 0 || person.Age > 150 {
			return errors.New("age must be between 0 and 150")
		}
		if len(person.Hobbies) < 1 || len(person.Hobbies) > 5 {
			return errors.New("number of hobbies must be between 1 and 5")
		}
		if person.LuckyNumber < 1 || person.LuckyNumber > 100 {
			return errors.New("lucky number must be between 1 and 100")
		}
		return nil
	}

	// Compare model outputs
	fmt.Println("Starting model comparison...")
	results, err := presets.CompareModels(ctx, promptText, validateComplexPerson, configs...)
	if err != nil {
		log.Fatalf("Error comparing models: %v", err)
	}

	// Print results as they come in
	for _, result := range results {
		fmt.Printf("\n%s\n", strings.Repeat("=", 50))
		fmt.Printf("Results for %s %s\n", result.Provider, result.Model)
		fmt.Printf("Attempts: %d\n", result.Attempts)

		if result.Error != nil {
			fmt.Printf("Error: %v\n", result.Error)
			debugLog("Raw response:\n%s", result.Response)
			continue
		}

		fmt.Println("Valid output generated:")
		prettyJSON, err := json.MarshalIndent(result.Data, "", "  ")
		if err != nil {
			fmt.Printf("Error prettifying JSON: %v\n", err)
			debugLog("Raw response:\n%s", result.Response)
		} else {
			fmt.Printf("%s\n", string(prettyJSON))
		}

		debugLog("Processed result for %s %s", result.Provider, result.Model)

		fmt.Printf("%s\n", strings.Repeat("=", 50))
	}

	// Print the final analysis
	fmt.Println("\nFinal Analysis of Results:")
	analysis := presets.AnalyzeComparisonResults(results)
	fmt.Println(analysis)

	debugLog("Structured output comparison completed")
}
