package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// TestOpenRouterIntegration performs integration tests against the actual OpenRouter API.
// These tests require a valid API key set in the OPENROUTER_API_KEY environment variable.
func TestOpenRouterIntegration(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping OpenRouter integration tests. Set OPENROUTER_API_KEY to run them.")
	}

	// Create a logger to see detailed output
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("Basic Chat Completion", func(t *testing.T) {
		// Create provider with Claude model
		provider := NewOpenRouterProvider(apiKey, "anthropic/claude-3-haiku", nil)
		provider.SetLogger(logger)

		// Create a simple prompt
		prompt := "What is the capital of France? Answer in one word."

		// Prepare the request
		requestBody, err := provider.PrepareRequest(prompt, map[string]any{
			"temperature": 0.0, // Use 0 temperature for deterministic results
			"max_tokens":  10,  // Short response
		})
		require.NoError(t, err)

		// Make the actual API call
		respBody, err := makeAPICall(t, provider.Endpoint(), provider.Headers(), requestBody)
		require.NoError(t, err)
		t.Logf("Response body: %s", string(respBody))

		// Parse the response
		response, err := provider.ParseResponse(respBody)
		require.NoError(t, err)

		// Validate the response
		require.Contains(t, response, "Paris")
		t.Logf("Response: %s", response.AsText())
	})

	t.Run("Model Fallback", func(t *testing.T) {
		// Create provider with an intentionally invalid model and fallbacks
		provider := NewOpenRouterProvider(apiKey, "invalid-model", nil)
		provider.SetLogger(logger)

		// Prepare the request with fallbacks
		requestBody, err := provider.PrepareRequest("What is 2+2?", map[string]any{
			"temperature":     0.0,
			"max_tokens":      10,
			"fallback_models": []string{"anthropic/claude-3-haiku", "openai/gpt-3.5-turbo"},
		})
		require.NoError(t, err)

		// Make the actual API call
		respBody, err := makeAPICall(t, provider.Endpoint(), provider.Headers(), requestBody)
		require.NoError(t, err)
		t.Logf("Response body: %s", string(respBody))

		// Parse the response
		response, err := provider.ParseResponse(respBody)
		require.NoError(t, err)

		// Validate the response - we should get an answer despite the invalid primary model
		require.Contains(t, response, "4")

		// Check which model was actually used from the response
		var parsed map[string]any
		err = json.Unmarshal(respBody, &parsed)
		require.NoError(t, err)

		model, ok := parsed["model"].(string)
		require.True(t, ok)
		t.Logf("Fallback used model: %s", model)

		// Verify it's one of our fallback models
		require.Contains(t, []string{"anthropic/claude-3-haiku", "openai/gpt-3.5-turbo"}, model)
	})

	t.Run("JSON Schema Validation", func(t *testing.T) {
		// Create provider with Claude model (supports JSON schema)
		provider := NewOpenRouterProvider(apiKey, "anthropic/claude-3-haiku", nil)
		provider.SetLogger(logger)

		// Define the schema
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type": "string",
				},
				"age": map[string]any{
					"type": "integer",
				},
			},
			"required": []string{"name", "age"},
		}

		// Prepare the request with schema
		requestBody, err := provider.PrepareRequestWithSchema(
			"Create a JSON object for a person named Alex who is 25 years old.",
			map[string]any{
				"temperature": 0.0,
				"max_tokens":  100,
			},
			schema,
		)
		require.NoError(t, err)

		// Make the actual API call
		respBody, err := makeAPICall(t, provider.Endpoint(), provider.Headers(), requestBody)
		require.NoError(t, err)
		t.Logf("Response body: %s", string(respBody))

		// Parse the response
		response, err := provider.ParseResponse(respBody)
		require.NoError(t, err)

		// Validate the response is valid JSON conforming to the schema
		var personData map[string]any
		err = json.Unmarshal([]byte(response.AsText()), &personData)
		require.NoError(t, err, "Response should be valid JSON")

		name, ok := personData["name"].(string)
		require.True(t, ok, "Should have a name field of type string")
		require.Equal(t, "Alex", name)

		age, ok := personData["age"].(float64) // JSON numbers are parsed as float64
		require.True(t, ok, "Should have an age field of type number")
		require.Equal(t, float64(25), age)
	})

	t.Run("Message History with Reasoning", func(t *testing.T) {
		// Create provider with Claude model
		provider := NewOpenRouterProvider(apiKey, "anthropic/claude-3-haiku", nil)
		provider.SetLogger(logger)

		// Create a conversation history
		messages := []types.MemoryMessage{
			{Role: "system", Content: "You are a math tutor that explains step by step."},
			{Role: "user", Content: "What is 17 × 6?"},
		}

		// Prepare the request with message history and reasoning tokens
		requestBody, err := provider.PrepareRequestWithMessages(messages, map[string]any{
			"temperature": 0.0,
			"max_tokens":  200,
			"transforms":  []string{"reasoning"},
		})
		require.NoError(t, err)

		// Make the actual API call
		respBody, err := makeAPICall(t, provider.Endpoint(), provider.Headers(), requestBody)
		require.NoError(t, err)
		t.Logf("Response body: %s", string(respBody))

		// Parse the response
		response, err := provider.ParseResponse(respBody)
		require.NoError(t, err)

		// Validate the response shows reasoning steps and correct answer
		require.Contains(t, response, "17")
		require.Contains(t, response, "6")
		require.Contains(t, response, "102") // Result of 17 × 6
		t.Logf("Response with reasoning: %s", response.AsText())
	})

	t.Run("Tool Calling", func(t *testing.T) {
		// Use a model that supports tool calling
		provider := NewOpenRouterProvider(apiKey, "openai/gpt-4o", nil)
		provider.SetLogger(logger)

		// Define the tools
		tools := []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Get the current weather in a given location",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		}

		// Create a prompt that should trigger tool calling
		prompt := "What's the weather in Tokyo today?"

		// Prepare the request with tools
		requestBody, err := provider.PrepareRequest(prompt, map[string]any{
			"temperature": 0.0,
			"max_tokens":  100,
			"tools":       tools,
			"tool_choice": "auto",
		})
		require.NoError(t, err)

		// Make the actual API call
		respBody, err := makeAPICall(t, provider.Endpoint(), provider.Headers(), requestBody)
		require.NoError(t, err)
		t.Logf("Response body: %s", string(respBody))

		// Process function calls
		toolCallResp, err := provider.HandleFunctionCalls(respBody)
		require.NoError(t, err)
		require.NotNil(t, toolCallResp, "Should detect a tool call")

		// Verify that we get a tool call for weather in Tokyo
		var parsed map[string]any
		err = json.Unmarshal(toolCallResp, &parsed)
		require.NoError(t, err)

		// Extract the tool call details
		choices, ok := parsed["choices"].([]any)
		require.True(t, ok)
		require.NotEmpty(t, choices)

		message, ok := choices[0].(map[string]any)["message"].(map[string]any)
		require.True(t, ok)

		toolCalls, ok := message["tool_calls"].([]any)
		require.True(t, ok)
		require.NotEmpty(t, toolCalls)

		// Verify it's calling the weather function
		toolCall := toolCalls[0].(map[string]any)
		require.Equal(t, "function", toolCall["type"])

		functionCall := toolCall["function"].(map[string]any)
		require.Equal(t, "get_weather", functionCall["name"])

		// Check the location is Tokyo
		args := functionCall["arguments"].(string)
		var argsMap map[string]any
		err = json.Unmarshal([]byte(args), &argsMap)
		require.NoError(t, err)

		location, ok := argsMap["location"].(string)
		require.True(t, ok)
		require.Contains(t, location, "Tokyo")
	})
}

// makeAPICall makes an actual HTTP request to the OpenRouter API
func makeAPICall(t *testing.T, endpoint string, headers map[string]string, requestBody []byte) ([]byte, error) {
	t.Helper()

	// Create an HTTP client with a reasonable timeout
	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Create the request
	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	// Make the request
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close() // Ignore close error in test
	}()

	// Read the response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	return body, nil
}
