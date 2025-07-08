// Package presets provides utilities for enhancing Language Learning Model interactions
// with specific reasoning patterns and comparison capabilities.
package presets

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/guiperry/gollm_cerebras/config"
	"github.com/guiperry/gollm_cerebras/llm"
	"github.com/guiperry/gollm_cerebras/providers"
	"github.com/guiperry/gollm_cerebras/utils"
)

// ComparisonResult represents the outcome of a model comparison for a specific provider.
// It is a generic type that can hold any structured response data along with metadata
// about the generation attempt.
type ComparisonResult[T any] struct {
	Provider string // The LLM provider (e.g., "openai", "anthropic")
	Model    string // The specific model used (e.g., "gpt-4", "claude-2")
	Response string // Raw response from the model
	Data     T      // Parsed and validated response data
	Error    error  // Any error encountered during generation or validation
	Attempts int    // Number of attempts made to get a valid response
}

// debugLog outputs debug information when debug logging is enabled in the config.
// It helps track the comparison process and troubleshoot issues.
func debugLog(config *config.Config, format string, args ...interface{}) {
	if config.LogLevel == utils.LogLevelDebug {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

// cleanResponse processes the raw model response to extract valid JSON.
// It handles common response formats including:
// - Responses wrapped in markdown code blocks
// - Responses with additional text before/after JSON
// - Responses with multiple JSON objects
func cleanResponse(response string) string {
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		response = response[start : end+1]
	}
	return strings.TrimSpace(response)
}

// ValidateFunc is a type for custom validation functions that verify
// the parsed response data meets specific requirements.
type ValidateFunc[T any] func(T) error

// CompareModels executes the same prompt across multiple LLM configurations
// and returns structured comparison results. It supports automatic retries
// and validation of responses.
//
// The function:
// 1. Attempts to generate responses from all models
// 2. Cleans and parses JSON responses
// 3. Validates responses using the provided validation function
// 4. Retries failed attempts up to 3 times
//
// Type parameter T represents the expected response structure.
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - prompt: The prompt to send to all models
//   - validateFunc: Custom validation function for parsed responses
//   - configs: List of LLM configurations to compare
//
// Returns:
//   - []ComparisonResult[T]: Results from all models, including any errors
//   - error: Any error that prevented the comparison from completing
//
// Example usage with complex structured data and validation:
//
//	// Define a complex data structure with validation tags
//	type ComplexPerson struct {
//	    Name          string   `json:"name" validate:"required"`
//	    Age           int      `json:"age" validate:"required,gte=0,lte=150"`
//	    Occupation    string   `json:"occupation" validate:"required"`
//	    City          string   `json:"city" validate:"required"`
//	    Country       string   `json:"country" validate:"required"`
//	    FavoriteColor string   `json:"favoriteColor" validate:"required"`
//	    Hobbies       []string `json:"hobbies" validate:"required,min=1,max=5"`
//	    Education     string   `json:"education" validate:"required"`
//	    PetName       string   `json:"petName" validate:"required"`
//	    LuckyNumber   int      `json:"luckyNumber" validate:"required,gte=1,lte=100"`
//	}
//
//	// Create validation function with custom rules
//	validatePerson := func(person ComplexPerson) error {
//	    if person.Age < 0 || person.Age > 150 {
//	        return fmt.Errorf("age must be between 0 and 150")
//	    }
//	    if len(person.Hobbies) < 1 || len(person.Hobbies) > 5 {
//	        return fmt.Errorf("number of hobbies must be between 1 and 5")
//	    }
//	    if person.LuckyNumber < 1 || person.LuckyNumber > 100 {
//	        return fmt.Errorf("lucky number must be between 1 and 100")
//	    }
//	    return nil
//	}
//
//	// Generate JSON schema for validation
//	schema, _ := gollm.GenerateJSONSchema(ComplexPerson{})
//
//	// Create structured prompt with schema
//	prompt := fmt.Sprintf(`Generate information about a fictional person.
//	Create a fictional person with the following attributes: name, age,
//	occupation, city, country, favorite color, hobbies (1-5), education,
//	pet name, and lucky number (1-100).
//	Return the data as a JSON object that adheres to this schema:
//	%s`, string(schema))
//
//	// Configure multiple models for comparison
//	configs := []*config.Config{
//	    {Provider: "openai", Model: "gpt-4"},
//	    {Provider: "anthropic", Model: "claude-2"},
//	    {Provider: "openai", Model: "gpt-3.5-turbo"},
//	}
//
//	// Compare model outputs
//	results, err := CompareModels[ComplexPerson](
//	    ctx, prompt, validatePerson, configs...)
//
//	// Analyze and display results
//	analysis := AnalyzeComparisonResults(results)
//	fmt.Println(analysis)
func CompareModels[T any](ctx context.Context, prompt string, validateFunc ValidateFunc[T], configs ...*config.Config) ([]ComparisonResult[T], error) {
	// Validate inputs
	if prompt == "" {
		return nil, fmt.Errorf("prompt cannot be empty")
	}
	if validateFunc == nil {
		return nil, fmt.Errorf("validator function cannot be nil")
	}
	if len(configs) == 0 {
		return nil, fmt.Errorf("at least one config must be provided")
	}

	results := make([]ComparisonResult[T], len(configs))
	remainingConfigs := make([]*config.Config, len(configs))
	copy(remainingConfigs, configs)

	logger := utils.NewLogger(utils.LogLevelDebug)

	for attempt := 1; attempt <= 3; attempt++ {
		if len(remainingConfigs) == 0 {
			break
		}

		newRemainingConfigs := []*config.Config{}

		for _, config := range remainingConfigs {
			debugLog(config, "Attempting generation for %s %s (Attempt %d)", config.Provider, config.Model, attempt)

			registry := providers.NewProviderRegistry()
			llmInstance, err := llm.NewLLM(config, logger, registry)
			if err != nil {
				return nil, fmt.Errorf("failed to create LLM for %s: %w", config.Provider, err)
			}

			response, err := llmInstance.Generate(ctx, llm.NewPrompt(prompt))

			index := findConfigIndex(configs, config)
			results[index].Provider = config.Provider
			results[index].Model = config.Model
			results[index].Response = response
			results[index].Error = err
			results[index].Attempts = attempt

			if err != nil {
				debugLog(config, "Error generating response: %v", err)
				// Immediately propagate API errors (like invalid keys)
				if strings.Contains(err.Error(), "API error") {
					return nil, fmt.Errorf("API error for %s %s: %w", config.Provider, config.Model, err)
				}
				if attempt == 3 {
					return nil, fmt.Errorf("failed to generate response after all attempts: %w", err)
				}
				newRemainingConfigs = append(newRemainingConfigs, config)
				continue
			}

			debugLog(config, "Raw response received: %s", response)

			cleanedResponse := cleanResponse(response)
			debugLog(config, "Cleaned response: %s", cleanedResponse)

			results[index].Response = cleanedResponse

			var data T
			if err := json.Unmarshal([]byte(cleanedResponse), &data); err != nil {
				debugLog(config, "Invalid JSON: %v", err)
				results[index].Error = fmt.Errorf("invalid JSON: %w", err)
				if attempt == 3 {
					return nil, fmt.Errorf("failed to parse JSON after all attempts: %w", err)
				}
				newRemainingConfigs = append(newRemainingConfigs, config)
				continue
			}

			if err := validateFunc(data); err != nil {
				debugLog(config, "Validation failed: %v", err)
				results[index].Error = fmt.Errorf("validation failed: %w", err)
				if attempt == 3 {
					return nil, fmt.Errorf("validation failed after all attempts: %w", err)
				}
				newRemainingConfigs = append(newRemainingConfigs, config)
				continue
			}

			results[index].Data = data
			debugLog(config, "Valid response received for %s %s", config.Provider, config.Model)
		}

		remainingConfigs = newRemainingConfigs
	}

	return results, nil
}

// findConfigIndex finds the index of a config in the original config slice.
// This helps maintain result ordering consistent with input configs.
func findConfigIndex(configs []*config.Config, target *config.Config) int {
	for i, config := range configs {
		if config.Provider == target.Provider && config.Model == target.Model {
			return i
		}
	}
	return -1
}

// AnalyzeComparisonResults generates a formatted analysis of comparison results.
// It creates a human-readable report showing:
// - Provider and model information
// - Number of generation attempts
// - Any errors encountered
// - Prettified JSON responses
//
// The analysis is particularly useful for:
// - Comparing model performance
// - Debugging generation issues
// - Evaluating response quality
// - Identifying systematic errors
//
// Example output for complex structured data:
//
//	----------------------------------------
//	Provider: openai, Model: gpt-4
//	Attempts: 1
//	Response: {
//	  "name": "Alexandra Chen",
//	  "age": 28,
//	  "occupation": "Environmental Scientist",
//	  "city": "Vancouver",
//	  "country": "Canada",
//	  "favoriteColor": "Emerald Green",
//	  "hobbies": [
//	    "Rock Climbing",
//	    "Urban Gardening",
//	    "Wildlife Photography"
//	  ],
//	  "education": "Master's in Environmental Science",
//	  "petName": "Luna",
//	  "luckyNumber": 42
//	}
//	----------------------------------------
//	Provider: anthropic, Model: claude-2
//	Attempts: 1
//	Response: {
//	  "name": "Marcus Rodriguez",
//	  "age": 34,
//	  "occupation": "Software Architect",
//	  "city": "Austin",
//	  "country": "United States",
//	  "favoriteColor": "Navy Blue",
//	  "hobbies": [
//	    "Jazz Piano",
//	    "Drone Racing",
//	    "Cooking",
//	    "Hiking"
//	  ],
//	  "education": "Bachelor's in Computer Science",
//	  "petName": "Pixel",
//	  "luckyNumber": 23
//	}
//	----------------------------------------
func AnalyzeComparisonResults[T any](results []ComparisonResult[T]) string {
	var analysis strings.Builder

	for _, result := range results {
		analysis.WriteString(strings.Repeat("-", 40) + "\n")
		analysis.WriteString(fmt.Sprintf("Provider: %s, Model: %s\n", result.Provider, result.Model))
		analysis.WriteString(fmt.Sprintf("Attempts: %d\n", result.Attempts))
		if result.Error != nil {
			analysis.WriteString(fmt.Sprintf("Error: %v\n", result.Error))
		} else {
			prettyJSON, err := json.MarshalIndent(result.Data, "", "  ")
			if err != nil {
				analysis.WriteString(fmt.Sprintf("Error prettifying JSON: %v\n", err))
				analysis.WriteString(fmt.Sprintf("Raw response: %s\n", result.Response))
			} else {
				analysis.WriteString(fmt.Sprintf("Response: %s\n", string(prettyJSON)))
			}
		}
		analysis.WriteString(strings.Repeat("-", 40) + "\n")
	}

	return analysis.String()
}
