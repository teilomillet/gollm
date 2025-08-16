package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/teilomillet/gollm/providers"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/assess"
)

// validateSingleFunctionCall validates a single function call response
func validateSingleFunctionCall(response string, expectedName string, expectedLocation string) error {
	functionCalls, err := providers.ExtractFunctionCalls(response)
	if err != nil {
		return fmt.Errorf("failed to extract function calls: %w", err)
	}

	if len(functionCalls) != 1 {
		return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
	}

	call := functionCalls[0]
	if call["name"] != expectedName {
		return fmt.Errorf("expected function name '%s', got '%s'", expectedName, call["name"])
	}

	args, ok := call["arguments"].(map[string]any)
	if !ok {
		return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
	}

	location, ok := args["location"].(string)
	if !ok {
		return fmt.Errorf("expected location to be a string, got %T", args["location"])
	}

	if !strings.HasPrefix(strings.ToLower(location), strings.ToLower(expectedLocation)) {
		return fmt.Errorf("location should start with '%s', got '%s'", expectedLocation, location)
	}

	return nil
}

// validateMultipleFunctionCalls validates multiple function calls
func validateMultipleFunctionCalls(response string) error {
	functionCalls, err := providers.ExtractFunctionCalls(response)
	if err != nil {
		return fmt.Errorf("failed to extract function calls: %w", err)
	}

	if len(functionCalls) != 2 {
		return fmt.Errorf("expected 2 function calls, got %d", len(functionCalls))
	}

	foundWeather := false
	foundTime := false
	for _, call := range functionCalls {
		switch call["name"] {
		case "get_weather":
			foundWeather = true
			if err := validateWeatherCall(call); err != nil {
				return err
			}
		case "get_time":
			foundTime = true
			if err := validateTimeCall(call); err != nil {
				return err
			}
		}
	}

	if !foundWeather || !foundTime {
		return fmt.Errorf("missing required function calls: weather=%v, time=%v", foundWeather, foundTime)
	}

	return nil
}

// validateWeatherCall validates a weather function call
func validateWeatherCall(call map[string]any) error {
	args, ok := call["arguments"].(map[string]any)
	if !ok {
		return errors.New("expected weather arguments to be a map")
	}
	location, ok := args["location"].(string)
	if !ok || !strings.HasPrefix(strings.ToLower(location), "new york") {
		return fmt.Errorf("invalid weather location: %v", args["location"])
	}
	return nil
}

// validateTimeCall validates a time function call
func validateTimeCall(call map[string]any) error {
	args, ok := call["arguments"].(map[string]any)
	if !ok {
		return errors.New("expected time arguments to be a map")
	}
	timezone, ok := args["timezone"].(string)
	if !ok || !strings.Contains(timezone, "New_York") {
		return fmt.Errorf("invalid timezone: %v", args["timezone"])
	}
	return nil
}

// validateOptionalParam validates optional parameter handling
func validateOptionalParam(response string) error {
	functionCalls, err := providers.ExtractFunctionCalls(response)
	if err != nil {
		return fmt.Errorf("failed to extract function calls: %w", err)
	}

	if len(functionCalls) != 1 {
		return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
	}

	call := functionCalls[0]
	args, ok := call["arguments"].(map[string]any)
	if !ok {
		return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
	}

	location, ok := args["location"].(string)
	if !ok || !strings.Contains(strings.ToLower(location), "london") {
		return fmt.Errorf("invalid location: %v", args["location"])
	}

	if unit, ok := args["unit"].(string); ok {
		if unit != "celsius" && unit != "fahrenheit" {
			return fmt.Errorf("if unit is provided, it must be celsius or fahrenheit, got: %s", unit)
		}
	}

	return nil
}

func TestFunctionCalling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping function calling test in short mode")
	}

	// Define the weather function
	getWeatherFunction := gollm.Function{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
			},
			"required": []string{"location"},
		},
	}

	// Define time function
	getTimeFunction := gollm.Function{
		Name:        "get_time",
		Description: "Get the current time in a given timezone",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"timezone": map[string]any{
					"type":        "string",
					"description": "The timezone, e.g. America/New_York",
				},
			},
			"required": []string{"timezone"},
		},
	}

	// Define optional parameter function
	getWeatherWithOptionalFunction := gollm.Function{
		Name:        "get_weather_with_unit",
		Description: "Get the current weather in a given location with optional unit",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
				"unit": map[string]any{
					"type":        "string",
					"enum":        []string{"celsius", "fahrenheit"},
					"description": "The unit of temperature (optional, defaults to celsius)",
				},
			},
			"required": []string{"location"},
		},
	}

	// Define record summary function
	recordSummaryFunction := gollm.Function{
		Name:        "record_summary",
		Description: "Record summary using well-structured JSON",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"summary": map[string]any{
					"type":        "string",
					"description": "Brief summary text",
				},
				"tags": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "string",
					},
					"description": "List of relevant tags",
				},
			},
			"required": []string{"summary"},
		},
	}

	// Create separate test runners for each provider
	testOpenAI := assess.NewTest(t).
		WithProviders(map[string]string{
			"openai": "gpt-4o-mini",
		}).
		WithBatchConfig(assess.BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  1,
			BatchTimeout: 5 * time.Minute,
		})

	testAnthropic := assess.NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
		}).
		WithBatchConfig(assess.BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  1,
			BatchTimeout: 5 * time.Minute,
		})

	// Add common test cases for both providers
	for _, runner := range []*assess.TestRunner{testOpenAI, testAnthropic} {
		// Test single function call generation
		runner.AddCase("single_function_call", "What's the weather like in New York?").
			WithTimeout(30*time.Second).
			WithOption("tools", []gollm.Tool{
				{
					Type:     "function",
					Function: getWeatherFunction,
				},
			}).
			WithOption("tool_choice", "auto").
			Validate(func(response string) error {
				return validateSingleFunctionCall(response, "get_weather", "new york")
			})

		// Test multiple function calls
		runner.AddCase("multiple_function_calls", "What's the weather and time in New York?").
			WithTimeout(30*time.Second).
			WithOption("tools", []gollm.Tool{
				{Type: "function", Function: getWeatherFunction},
				{Type: "function", Function: getTimeFunction},
			}).
			WithOption("tool_choice", "auto").
			Validate(validateMultipleFunctionCalls)

		// Test missing required parameter
		runner.AddCase("missing_required_param", "What's the weather like?").
			WithTimeout(30*time.Second).
			WithOption("tools", []gollm.Tool{
				{
					Type:     "function",
					Function: getWeatherFunction,
				},
			}).
			WithOption("tool_choice", "auto").
			Validate(func(response string) error {
				// Validation logic remains the same
				functionCalls, err := providers.ExtractFunctionCalls(response)
				if err != nil {
					return fmt.Errorf("failed to extract function calls: %w", err)
				}

				// We expect no function calls since location is required but not provided
				if len(functionCalls) > 0 {
					return fmt.Errorf(
						"expected no function calls due to missing required parameter, got %d",
						len(functionCalls),
					)
				}

				// The response should ask for the location
				if !strings.Contains(strings.ToLower(response), "location") {
					return fmt.Errorf("response should ask for location, got: %s", response)
				}

				return nil
			})

		// Test optional parameter handling
		runner.AddCase("optional_param_handling", "What's the weather like in London?").
			WithTimeout(30*time.Second).
			WithOption("tools", []gollm.Tool{
				{
					Type:     "function",
					Function: getWeatherWithOptionalFunction,
				},
			}).
			WithOption("tool_choice", "auto").
			Validate(validateOptionalParam)
	}

	// Provider-specific test cases

	// OpenAI-specific test for forced tool use
	testOpenAI.AddCase("forced_tool_use", "Tell me about the weather.").
		WithTimeout(30*time.Second).
		WithOption("tools", []gollm.Tool{
			{
				Type:     "function",
				Function: getWeatherFunction,
			},
		}).
		WithOption("tool_choice", "required"). // OpenAI uses "required" as a simple string value
		Validate(func(response string) error {
			functionCalls, err := providers.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %w", err)
			}

			// With forced tool use, we expect at least one function call
			if len(functionCalls) == 0 {
				return errors.New("expected at least one function call with forced tool use")
			}

			// Verify the get_weather function was called
			call := functionCalls[0]
			if call["name"] != "get_weather" {
				return fmt.Errorf("expected function name 'get_weather', got '%s'", call["name"])
			}

			return nil
		})

	// Anthropic-specific test for forced tool use
	testAnthropic.AddCase("forced_tool_use", "Tell me about the weather.").
		WithTimeout(30*time.Second).
		WithOption("tools", []gollm.Tool{
			{
				Type:     "function",
				Function: getWeatherFunction,
			},
		}).
		WithOption("tool_choice", "tool"). // Anthropic uses "tool" as a string value
		Validate(func(response string) error {
			functionCalls, err := providers.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %w", err)
			}

			// With forced tool use, we expect at least one function call
			if len(functionCalls) == 0 {
				return errors.New("expected at least one function call with forced tool use")
			}

			// Verify the get_weather function was called
			call := functionCalls[0]
			if call["name"] != "get_weather" {
				return fmt.Errorf("expected function name 'get_weather', got '%s'", call["name"])
			}

			return nil
		})

	// OpenAI-specific test for JSON schema output
	testOpenAI.AddCase("json_schema_output", "Summarize: The weather is sunny and warm today.").
		WithTimeout(30*time.Second).
		WithOption("tools", []gollm.Tool{
			{
				Type:     "function",
				Function: recordSummaryFunction,
			},
		}).
		WithOption("tool_choice", map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "record_summary",
			},
		}).
		Validate(func(response string) error {
			// Validation logic remains the same
			functionCalls, err := providers.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %w", err)
			}

			if len(functionCalls) != 1 {
				return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
			}

			call := functionCalls[0]
			if call["name"] != "record_summary" {
				return fmt.Errorf("expected function name 'record_summary', got '%s'", call["name"])
			}

			args, ok := call["arguments"].(map[string]any)
			if !ok {
				return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
			}

			// Verify required summary field
			summary, ok := args["summary"].(string)
			if !ok || summary == "" {
				return errors.New("missing or invalid summary field")
			}

			// Optional tags field should be an array if present
			if tags, ok := args["tags"].([]any); ok {
				for i, tag := range tags {
					if _, ok := tag.(string); !ok {
						return fmt.Errorf("tag at index %d is not a string", i)
					}
				}
			}

			return nil
		})

	// Anthropic-specific test for JSON schema output
	testAnthropic.AddCase("json_schema_output", "Summarize: The weather is sunny and warm today.").
		WithTimeout(30*time.Second).
		WithOption("tools", []gollm.Tool{
			{
				Type:     "function",
				Function: recordSummaryFunction,
			},
		}).
		WithOption("tool_choice", "tool"). // Use simple string value for Anthropic
		Validate(func(response string) error {
			// Validation logic remains the same
			functionCalls, err := providers.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %w", err)
			}

			if len(functionCalls) != 1 {
				return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
			}

			call := functionCalls[0]
			if call["name"] != "record_summary" {
				return fmt.Errorf("expected function name 'record_summary', got '%s'", call["name"])
			}

			args, ok := call["arguments"].(map[string]any)
			if !ok {
				return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
			}

			// Verify required summary field
			summary, ok := args["summary"].(string)
			if !ok || summary == "" {
				return errors.New("missing or invalid summary field")
			}

			// Optional tags field should be an array if present
			if tags, ok := args["tags"].([]any); ok {
				for i, tag := range tags {
					if _, ok := tag.(string); !ok {
						return fmt.Errorf("tag at index %d is not a string", i)
					}
				}
			}

			return nil
		})

	// Run the test cases for each provider
	ctx := context.Background()
	testOpenAI.RunBatch(ctx)
	testAnthropic.RunBatch(ctx)

	// Check metrics for both providers
	openAIMetrics := testOpenAI.GetBatchMetrics()
	anthropicMetrics := testAnthropic.GetBatchMetrics()

	// Verify batch execution completed for both
	assert.Positive(t, openAIMetrics.BatchTiming.TotalDuration)
	assert.True(t, openAIMetrics.BatchTiming.EndTime.After(openAIMetrics.BatchTiming.StartTime))
	assert.Positive(t, anthropicMetrics.BatchTiming.TotalDuration)
	assert.True(t, anthropicMetrics.BatchTiming.EndTime.After(anthropicMetrics.BatchTiming.StartTime))

	// Check provider latencies
	for provider, latency := range openAIMetrics.BatchTiming.ProviderLatency {
		t.Logf("Provider %s average latency: %v", provider, latency)
		assert.Positive(t, latency)
	}
	for provider, latency := range anthropicMetrics.BatchTiming.ProviderLatency {
		t.Logf("Provider %s average latency: %v", provider, latency)
		assert.Positive(t, latency)
	}

	// Verify error handling
	for provider, errs := range openAIMetrics.Errors {
		for _, err := range errs {
			t.Logf("Provider %s error: %v", provider, err)
		}
	}
	for provider, errs := range anthropicMetrics.Errors {
		for _, err := range errs {
			t.Logf("Provider %s error: %v", provider, err)
		}
	}
}
