// Package presets provide utilities for enhancing Language Learning Model interactions
// with specific reasoning patterns and comparison capabilities.
package presets

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
	"github.com/weave-labs/gollm/llm"
	"github.com/weave-labs/gollm/providers"
)

// ComparisonResult represents the outcome of a model comparison for a specific provider.
// It is a generic type that can hold any structured response data along with metadata
// about the generation attempt.
type ComparisonResult[T any] struct {
	Data     T
	Error    error
	Response *providers.Response
	Provider string
	Model    string
	Attempts int
}

// debugLog outputs debug information when debug logging is enabled in the config.
// It helps track the comparison process and troubleshoot issues.
func debugLog(cfg *config.Config, format string, args ...any) {
	if cfg.LogLevel == logging.LogLevelDebug {
		slog.Debug(fmt.Sprintf(format, args...))
	}
}

// cleanResponse processes the raw model response to extract valid JSON.
// It handles common response formats including
// - Responses wrapped in Markdown code blocks
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
//
// validateCompareInputs validates the inputs for CompareModels
func validateCompareInputs[T any](prompt string, validateFunc ValidateFunc[T], configs []*config.Config) error {
	if prompt == "" {
		return errors.New("prompt cannot be empty")
	}
	if validateFunc == nil {
		return errors.New("validator function cannot be nil")
	}
	if len(configs) == 0 {
		return errors.New("at least one config must be provided")
	}
	return nil
}

// processConfig processes a single configuration and returns whether it succeeded
func processConfig[T any](
	ctx context.Context,
	prompt string,
	validateFunc ValidateFunc[T],
	remainingConfig *config.Config,
	configs []*config.Config,
	results []ComparisonResult[T],
	attempt int,
	logger logging.Logger,
) (bool, error) {
	debugLog(
		remainingConfig,
		"Attempting generation for %s %s (Attempt %d)",
		remainingConfig.Provider,
		remainingConfig.Model,
		attempt,
	)

	// Generate response
	response, err := generateResponse(ctx, prompt, remainingConfig, logger)
	if err != nil {
		return handleGenerationError(err, remainingConfig, attempt)
	}

	// Store result
	index := findConfigIndex(configs, remainingConfig)
	updateResult(&results[index], remainingConfig, response, attempt)

	// Process and validate response
	return processAndValidateResponse[T](&results[index], response, validateFunc, remainingConfig, attempt)
}

// generateResponse generates a response using the LLM
func generateResponse(
	ctx context.Context,
	prompt string,
	cfg *config.Config,
	logger logging.Logger,
) (*providers.Response, error) {
	registry := providers.NewProviderRegistry()
	llmInstance, err := llm.NewLLM(cfg, logger, registry)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM for %s: %w", cfg.Provider, err)
	}
	resp, err := llmInstance.Generate(ctx, llm.NewPrompt(prompt))
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}
	return resp, nil
}

// handleGenerationError handles errors from response generation
func handleGenerationError(err error, cfg *config.Config, attempt int) (bool, error) {
	debugLog(cfg, "Error generating response: %v", err)

	// Immediately propagate API errors (like invalid keys)
	if strings.Contains(err.Error(), "API error") {
		return false, fmt.Errorf(
			"API error for %s %s: %w",
			cfg.Provider,
			cfg.Model,
			err,
		)
	}

	if attempt == MaxRetryAttempts {
		return false, fmt.Errorf("failed to generate response after all attempts: %w", err)
	}

	return false, nil // Will retry
}

// updateResult updates the comparison result with response data
func updateResult[T any](
	result *ComparisonResult[T],
	cfg *config.Config,
	response *providers.Response,
	attempt int,
) {
	result.Provider = cfg.Provider
	result.Model = cfg.Model
	result.Response = response
	result.Attempts = attempt
}

// processAndValidateResponse processes the response and validates it
func processAndValidateResponse[T any](
	result *ComparisonResult[T],
	response *providers.Response,
	validateFunc ValidateFunc[T],
	cfg *config.Config,
	attempt int,
) (bool, error) {
	debugLog(cfg, "Raw response received: %v", response)

	cleanedResponse := cleanResponse(response.AsText())
	debugLog(cfg, "Cleaned response: %s", cleanedResponse)

	result.Response.Content = providers.Text{Value: cleanedResponse}

	// Parse JSON
	var data T
	if err := json.Unmarshal([]byte(cleanedResponse), &data); err != nil {
		debugLog(cfg, "Invalid JSON: %v", err)
		result.Error = fmt.Errorf("invalid JSON: %w", err)
		if attempt == MaxRetryAttempts {
			return false, fmt.Errorf("failed to parse JSON after all attempts: %w", err)
		}
		return false, nil // Will retry
	}

	// Validate data
	if err := validateFunc(data); err != nil {
		debugLog(cfg, "Validation failed: %v", err)
		result.Error = fmt.Errorf("validation failed: %w", err)
		if attempt == MaxRetryAttempts {
			return false, fmt.Errorf("validation failed after all attempts: %w", err)
		}
		return false, nil // Will retry
	}

	result.Data = data
	return true, nil // Success
}

// results, err := CompareModels[ComplexPerson](
//
//	ctx, prompt, validatePerson, configs...)
//
// // Analyze and display results
// analysis := AnalyzeComparisonResults(results)
// fmt.Println(analysis)
func CompareModels[T any](
	ctx context.Context,
	prompt string,
	validateFunc ValidateFunc[T],
	configs ...*config.Config,
) ([]ComparisonResult[T], error) {
	// Validate inputs
	if err := validateCompareInputs(prompt, validateFunc, configs); err != nil {
		return nil, err
	}

	results := make([]ComparisonResult[T], len(configs))
	remainingConfigs := make([]*config.Config, len(configs))
	copy(remainingConfigs, configs)

	logger := logging.NewLogger(logging.LogLevelDebug)

	for attempt := 1; attempt <= MaxRetryAttempts; attempt++ {
		if len(remainingConfigs) == 0 {
			break
		}

		var newRemainingConfigs []*config.Config

		for _, remainingConfig := range remainingConfigs {
			success, err := processConfig[T](
				ctx,
				prompt,
				validateFunc,
				remainingConfig,
				configs,
				results,
				attempt,
				logger,
			)
			if err != nil {
				return nil, err
			}
			if !success {
				newRemainingConfigs = append(newRemainingConfigs, remainingConfig)
			}
		}

		remainingConfigs = newRemainingConfigs
	}

	return results, nil
}

// findConfigIndex finds the index of a config in the original config slice.
// This helps maintain result ordering consistent with input configs.
func findConfigIndex(configs []*config.Config, target *config.Config) int {
	for i, llmConfig := range configs {
		if llmConfig.Provider == target.Provider && llmConfig.Model == target.Model {
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
		analysis.WriteString(strings.Repeat("-", SeparatorLineLength) + "\n")
		analysis.WriteString(fmt.Sprintf("Provider: %s, Model: %s\n", result.Provider, result.Model))
		analysis.WriteString(fmt.Sprintf("Attempts: %d\n", result.Attempts))
		if result.Error != nil {
			analysis.WriteString(fmt.Sprintf("Error: %v\n", result.Error))
		} else {
			prettyJSON, err := json.MarshalIndent(result.Data, "", "  ")
			if err != nil {
				analysis.WriteString(fmt.Sprintf("Error prettifying JSON: %v\n", err))
				analysis.WriteString(fmt.Sprintf("Raw response: %s\n", result.Response.AsText()))
			} else {
				analysis.WriteString(fmt.Sprintf("Response: %s\n", string(prettyJSON)))
			}
		}
		analysis.WriteString(strings.Repeat("-", SeparatorLineLength) + "\n")
	}

	return analysis.String()
}
