// File: examples/mcp_integration/main_test.go
package main

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/utils"
)

// setupLLM creates and configures an LLM client for testing.
// It handles API key validation and common configuration settings.
//
// Parameters:
//   - provider: The LLM provider to use ("openai" or "anthropic")
//   - model: The specific model to use (e.g., "gpt-4o-mini" for OpenAI)
//
// The function will skip the test if the required API key is not set.
func setupLLM(t *testing.T, provider, model string) gollm.LLM {
	var apiKey string
	switch provider {
	case "openai":
		apiKey = os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			t.Skip("OPENAI_API_KEY environment variable not set")
		}
	case "anthropic":
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			t.Skip("ANTHROPIC_API_KEY environment variable not set")
		}
	default:
		t.Fatalf("Unsupported provider: %s", provider)
	}

	// Create a new LLM instance with standard test configuration
	llm, err := gollm.NewLLM(
		gollm.SetProvider(provider),
		gollm.SetModel(model),
		gollm.SetMaxTokens(256), // Limit response length for testing
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(60*time.Second),    // Reasonable timeout for API calls
		gollm.SetMaxRetries(2),              // Retry failed API calls
		gollm.SetRetryDelay(10*time.Second), // Wait between retries
		gollm.SetAPIKey(apiKey),
	)
	require.NoError(t, err, "Should create LLM instance")
	return llm
}

// TestMCPToolIntegration tests the integration of MCP-formatted tools with different LLM providers.
// This test verifies that:
//  1. Tools can be defined in MCP format and converted to internal format
//  2. Different providers can understand and use these tools
//  3. Tools with varying complexity work as expected
func TestMCPToolIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping MCP integration test in short mode")
	}

	// Define the providers to test with
	providers := []struct {
		name  string
		model string
	}{
		{"openai", "gpt-4o-mini"},
		{"anthropic", "claude-3-5-haiku-latest"},
	}

	// Define test tools in MCP format
	// We test both simple and complex tool definitions
	tools := []struct {
		name string
		json string
	}{
		{
			name: "Simple Weather Tool",
			json: `{
				"type": "function",
				"name": "get_weather",
				"description": "Get the current weather for a location",
				"parameters": {
					"type": "object",
					"properties": {
						"location": {
							"type": "string",
							"description": "The city and state, e.g. San Francisco, CA"
						},
						"unit": {
							"type": "string",
							"enum": ["celsius", "fahrenheit"],
							"description": "The unit for the temperature"
						}
					},
					"required": ["location"]
				}
			}`,
		},
		{
			name: "Complex Data Processing Tool", // Tests nested parameter structures
			json: `{
				"type": "function",
				"name": "process_data",
				"description": "Process complex data structure",
				"parameters": {
					"type": "object",
					"properties": {
						"data": {
							"type": "array",
							"items": {
								"type": "object",
								"properties": {
									"id": {
										"type": "integer"
									},
									"metadata": {
										"type": "object",
										"additionalProperties": true
									}
								}
							}
						},
						"options": {
							"type": "object",
							"additionalProperties": true
						}
					},
					"required": ["data"]
				}
			}`,
		},
	}

	// Define test cases that exercise different aspects of tool usage
	testCases := []struct {
		name    string
		prompt  string
		toolIdx int
		verify  func(t *testing.T, response string)
	}{
		{
			name:    "Weather Query",
			prompt:  "What's the weather like in London?",
			toolIdx: 0, // Uses the simple weather tool
			verify: func(t *testing.T, response string) {
				assert.Contains(t, response, "get_weather", "Response should include tool call")
				assert.Contains(t, response, "London", "Response should include the location")
			},
		},
		{
			name:    "Data Processing",
			prompt:  "Process this data: [{\"id\": 1, \"metadata\": {\"type\": \"test\"}}]",
			toolIdx: 1, // Uses the complex data processing tool
			verify: func(t *testing.T, response string) {
				assert.Contains(t, response, "process_data", "Response should include tool call")
				assert.Contains(t, response, "data", "Response should include data field")
			},
		},
	}

	// Run tests for each provider
	for _, provider := range providers {
		t.Run(provider.name, func(t *testing.T) {
			llm := setupLLM(t, provider.name, provider.model)

			// Run each test case with the current provider
			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					// Convert MCP JSON to internal Tool format
					var tool utils.Tool
					err := json.Unmarshal([]byte(tools[tc.toolIdx].json), &tool)
					require.NoError(t, err, "Should unmarshal MCP JSON")

					// Create a prompt that includes the tool
					prompt := gollm.NewPrompt(
						tc.prompt,
						gollm.WithTools([]utils.Tool{tool}),
					)

					// Generate response from the LLM
					ctx := context.Background()
					response, err := llm.Generate(ctx, prompt)
					require.NoError(t, err, "Should generate response")

					// Verify the response meets our expectations
					tc.verify(t, response)
				})
			}
		})
	}
}

// TestMCPToolValidation tests advanced scenarios for tool usage:
//  1. Multiple tools in a single prompt
//  2. Tool choice enforcement (forced vs automatic)
func TestMCPToolValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping MCP tool validation test in short mode")
	}

	llm := setupLLM(t, "openai", "gpt-4o-mini")

	// Test using multiple tools in a single prompt
	t.Run("MultipleTools", func(t *testing.T) {
		// Define two related tools that might be used together
		weatherTool := utils.Tool{
			Type: "function",
			Function: utils.Function{
				Name:        "get_weather",
				Description: "Get the current weather for a location",
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
			},
		}

		timeTool := utils.Tool{
			Type: "function",
			Function: utils.Function{
				Name:        "get_time",
				Description: "Get the current time for a location",
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
			},
		}

		// Create a prompt that should trigger both tools
		prompt := gollm.NewPrompt(
			"What's the weather and time in London?",
			gollm.WithTools([]utils.Tool{weatherTool, timeTool}),
		)

		ctx := context.Background()
		response, err := llm.Generate(ctx, prompt)
		require.NoError(t, err, "Should generate response")

		// Verify both tools were used
		assert.Contains(t, response, "get_weather", "Response should include weather tool call")
		assert.Contains(t, response, "get_time", "Response should include time tool call")
		assert.Contains(t, response, "London", "Response should include the location")
	})

	// Test different tool choice behaviors
	t.Run("ToolChoice", func(t *testing.T) {
		// Define a simple weather tool for testing
		var tool utils.Tool
		err := json.Unmarshal([]byte(`{
			"type": "function",
			"name": "get_weather",
			"description": "Get the current weather for a location",
			"parameters": {
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. San Francisco, CA"
					}
				},
				"required": ["location"]
			}
		}`), &tool)
		require.NoError(t, err, "Should unmarshal MCP JSON")

		// Test when we force the model to use the tool
		t.Run("ForcedToolUsage", func(t *testing.T) {
			prompt := gollm.NewPrompt(
				"What's the current weather in London? Use the weather tool to find out.",
				gollm.WithTools([]utils.Tool{tool}),
				gollm.WithToolChoice("any"), // Force the model to use a tool
			)

			ctx := context.Background()
			response, err := llm.Generate(ctx, prompt)
			require.NoError(t, err, "Should generate response")

			assert.Contains(t, response, "get_weather", "Response should include tool call with forced choice")
			assert.Contains(t, response, "London", "Response should include the location")
		})

		// Test when we let the model decide whether to use the tool
		t.Run("AutoToolUsage", func(t *testing.T) {
			prompt := gollm.NewPrompt(
				"Tell me about London.",
				gollm.WithTools([]utils.Tool{tool}),
				gollm.WithToolChoice("auto"), // Let the model decide whether to use the tool
			)

			ctx := context.Background()
			response, err := llm.Generate(ctx, prompt)
			require.NoError(t, err, "Should generate response")

			// We don't assert tool usage here because it's up to the model
			// The model might choose to give a general description without using the tool
			assert.Contains(t, response, "London", "Response should include information about London")
		})
	})
}
