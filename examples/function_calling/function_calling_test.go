package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/guiperry/gollm_cerebras"
	"github.com/guiperry/gollm_cerebras/assess"
	"github.com/guiperry/gollm_cerebras/utils"
)

func TestFunctionCalling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping function calling test in short mode")
	}

	// Define the weather function
	getWeatherFunction := gollm.Function{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
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
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"timezone": map[string]interface{}{
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
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
				"unit": map[string]interface{}{
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
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"summary": map[string]interface{}{
					"type":        "string",
					"description": "Brief summary text",
				},
				"tags": map[string]interface{}{
					"type": "array",
					"items": map[string]interface{}{
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
			WithOption("tools", []gollm.Tool{{
				Type:     "function",
				Function: getWeatherFunction,
			}}).
			WithOption("tool_choice", "auto").
			Validate(func(response string) error {
				// The validation logic is the same as before
				functionCalls, err := utils.ExtractFunctionCalls(response)
				if err != nil {
					return fmt.Errorf("failed to extract function calls: %v", err)
				}

				if len(functionCalls) != 1 {
					return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
				}

				call := functionCalls[0]
				if call["name"] != "get_weather" {
					return fmt.Errorf("expected function name 'get_weather', got '%s'", call["name"])
				}

				args, ok := call["arguments"].(map[string]interface{})
				if !ok {
					return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
				}

				location, ok := args["location"].(string)
				if !ok {
					return fmt.Errorf("expected location to be a string, got %T", args["location"])
				}

				if !strings.HasPrefix(strings.ToLower(location), "new york") {
					return fmt.Errorf("location should start with 'New York', got '%s'", location)
				}

				return nil
			})

		// Test multiple function calls
		runner.AddCase("multiple_function_calls", "What's the weather and time in New York?").
			WithTimeout(30*time.Second).
			WithOption("tools", []gollm.Tool{
				{Type: "function", Function: getWeatherFunction},
				{Type: "function", Function: getTimeFunction},
			}).
			WithOption("tool_choice", "auto").
			Validate(func(response string) error {
				// Validation logic remains the same
				functionCalls, err := utils.ExtractFunctionCalls(response)
				if err != nil {
					return fmt.Errorf("failed to extract function calls: %v", err)
				}

				if len(functionCalls) != 2 {
					return fmt.Errorf("expected 2 function calls, got %d", len(functionCalls))
				}

				// Check that we have both types of function calls
				foundWeather := false
				foundTime := false
				for _, call := range functionCalls {
					switch call["name"] {
					case "get_weather":
						foundWeather = true
						args, ok := call["arguments"].(map[string]interface{})
						if !ok {
							return fmt.Errorf("expected weather arguments to be a map")
						}
						location, ok := args["location"].(string)
						if !ok || !strings.HasPrefix(strings.ToLower(location), "new york") {
							return fmt.Errorf("invalid weather location: %v", args["location"])
						}
					case "get_time":
						foundTime = true
						args, ok := call["arguments"].(map[string]interface{})
						if !ok {
							return fmt.Errorf("expected time arguments to be a map")
						}
						timezone, ok := args["timezone"].(string)
						if !ok || !strings.Contains(timezone, "New_York") {
							return fmt.Errorf("invalid timezone: %v", args["timezone"])
						}
					}
				}

				if !foundWeather || !foundTime {
					return fmt.Errorf("missing required function calls: weather=%v, time=%v", foundWeather, foundTime)
				}

				return nil
			})

		// Test missing required parameter
		runner.AddCase("missing_required_param", "What's the weather like?").
			WithTimeout(30*time.Second).
			WithOption("tools", []gollm.Tool{{
				Type:     "function",
				Function: getWeatherFunction,
			}}).
			WithOption("tool_choice", "auto").
			Validate(func(response string) error {
				// Validation logic remains the same
				functionCalls, err := utils.ExtractFunctionCalls(response)
				if err != nil {
					return fmt.Errorf("failed to extract function calls: %v", err)
				}

				// We expect no function calls since location is required but not provided
				if len(functionCalls) > 0 {
					return fmt.Errorf("expected no function calls due to missing required parameter, got %d", len(functionCalls))
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
			WithOption("tools", []gollm.Tool{{
				Type:     "function",
				Function: getWeatherWithOptionalFunction,
			}}).
			WithOption("tool_choice", "auto").
			Validate(func(response string) error {
				// Validation logic remains the same
				functionCalls, err := utils.ExtractFunctionCalls(response)
				if err != nil {
					return fmt.Errorf("failed to extract function calls: %v", err)
				}

				if len(functionCalls) != 1 {
					return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
				}

				call := functionCalls[0]
				args, ok := call["arguments"].(map[string]interface{})
				if !ok {
					return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
				}

				// Check that location is present and correct
				location, ok := args["location"].(string)
				if !ok || !strings.Contains(strings.ToLower(location), "london") {
					return fmt.Errorf("invalid location: %v", args["location"])
				}

				// Optional unit parameter might or might not be present
				if unit, ok := args["unit"].(string); ok {
					if unit != "celsius" && unit != "fahrenheit" {
						return fmt.Errorf("if unit is provided, it must be celsius or fahrenheit, got: %s", unit)
					}
				}

				return nil
			})
	}

	// Provider-specific test cases

	// OpenAI-specific test for forced tool use
	testOpenAI.AddCase("forced_tool_use", "Tell me about the weather.").
		WithTimeout(30*time.Second).
		WithOption("tools", []gollm.Tool{{
			Type:     "function",
			Function: getWeatherFunction,
		}}).
		WithOption("tool_choice", "required"). // OpenAI uses "required" as a simple string value
		Validate(func(response string) error {
			functionCalls, err := utils.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %v", err)
			}

			// With forced tool use, we expect at least one function call
			if len(functionCalls) == 0 {
				return fmt.Errorf("expected at least one function call with forced tool use")
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
		WithOption("tools", []gollm.Tool{{
			Type:     "function",
			Function: getWeatherFunction,
		}}).
		WithOption("tool_choice", "tool"). // Anthropic uses "tool" as a string value
		Validate(func(response string) error {
			functionCalls, err := utils.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %v", err)
			}

			// With forced tool use, we expect at least one function call
			if len(functionCalls) == 0 {
				return fmt.Errorf("expected at least one function call with forced tool use")
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
		WithOption("tools", []gollm.Tool{{
			Type:     "function",
			Function: recordSummaryFunction,
		}}).
		WithOption("tool_choice", map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "record_summary",
			},
		}).
		Validate(func(response string) error {
			// Validation logic remains the same
			functionCalls, err := utils.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %v", err)
			}

			if len(functionCalls) != 1 {
				return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
			}

			call := functionCalls[0]
			if call["name"] != "record_summary" {
				return fmt.Errorf("expected function name 'record_summary', got '%s'", call["name"])
			}

			args, ok := call["arguments"].(map[string]interface{})
			if !ok {
				return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
			}

			// Verify required summary field
			summary, ok := args["summary"].(string)
			if !ok || summary == "" {
				return fmt.Errorf("missing or invalid summary field")
			}

			// Optional tags field should be an array if present
			if tags, ok := args["tags"].([]interface{}); ok {
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
		WithOption("tools", []gollm.Tool{{
			Type:     "function",
			Function: recordSummaryFunction,
		}}).
		WithOption("tool_choice", "tool"). // Use simple string value for Anthropic
		Validate(func(response string) error {
			// Validation logic remains the same
			functionCalls, err := utils.ExtractFunctionCalls(response)
			if err != nil {
				return fmt.Errorf("failed to extract function calls: %v", err)
			}

			if len(functionCalls) != 1 {
				return fmt.Errorf("expected 1 function call, got %d", len(functionCalls))
			}

			call := functionCalls[0]
			if call["name"] != "record_summary" {
				return fmt.Errorf("expected function name 'record_summary', got '%s'", call["name"])
			}

			args, ok := call["arguments"].(map[string]interface{})
			if !ok {
				return fmt.Errorf("expected arguments to be a map, got %T", call["arguments"])
			}

			// Verify required summary field
			summary, ok := args["summary"].(string)
			if !ok || summary == "" {
				return fmt.Errorf("missing or invalid summary field")
			}

			// Optional tags field should be an array if present
			if tags, ok := args["tags"].([]interface{}); ok {
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
	assert.True(t, openAIMetrics.BatchTiming.TotalDuration > 0)
	assert.True(t, openAIMetrics.BatchTiming.EndTime.After(openAIMetrics.BatchTiming.StartTime))
	assert.True(t, anthropicMetrics.BatchTiming.TotalDuration > 0)
	assert.True(t, anthropicMetrics.BatchTiming.EndTime.After(anthropicMetrics.BatchTiming.StartTime))

	// Check provider latencies
	for provider, latency := range openAIMetrics.BatchTiming.ProviderLatency {
		t.Logf("Provider %s average latency: %v", provider, latency)
		assert.True(t, latency > 0)
	}
	for provider, latency := range anthropicMetrics.BatchTiming.ProviderLatency {
		t.Logf("Provider %s average latency: %v", provider, latency)
		assert.True(t, latency > 0)
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
