package providers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
)

func TestOpenRouterProvider(t *testing.T) {
	apiKey := "test_api_key"
	model := "anthropic/claude-3-5-sonnet"
	provider := NewOpenRouterProvider(apiKey, model, nil).(*OpenRouterProvider)

	t.Run("Name returns openrouter", func(t *testing.T) {
		assert.Equal(t, "openrouter", provider.Name())
	})

	t.Run("Endpoint returns correct URL", func(t *testing.T) {
		assert.Equal(t, "https://openrouter.ai/api/v1/chat/completions", provider.Endpoint())
	})

	t.Run("CompletionsEndpoint returns correct URL", func(t *testing.T) {
		assert.Equal(t, "https://openrouter.ai/api/v1/completions", provider.CompletionsEndpoint())
	})

	t.Run("GenerationEndpoint returns correct URL", func(t *testing.T) {
		assert.Equal(t, "https://openrouter.ai/api/v1/generation?id=test123", provider.GenerationEndpoint("test123"))
	})

	t.Run("Headers include correct authentication", func(t *testing.T) {
		headers := provider.Headers()
		assert.Equal(t, "Bearer test_api_key", headers["Authorization"])
		assert.Equal(t, "application/json", headers["Content-Type"])
		assert.Equal(t, "GoLLM Integration", headers["X-Title"])
	})

	t.Run("SupportsJSONSchema returns true", func(t *testing.T) {
		assert.True(t, provider.SupportsJSONSchema())
	})

	t.Run("SupportsStreaming returns true", func(t *testing.T) {
		assert.True(t, provider.SupportsStreaming())
	})

	t.Run("SetDefaultOptions sets correct options", func(t *testing.T) {
		// Create a config
		cfg := &config.Config{
			Temperature: 0.8,
			MaxTokens:   500,
		}

		// Set seed
		seed := 42
		cfg.Seed = &seed

		// Apply defaults
		provider.SetDefaultOptions(cfg)

		// Check options were set correctly
		assert.Equal(t, float64(0.8), provider.options["temperature"])
		assert.Equal(t, 500, provider.options["max_tokens"])
		assert.Equal(t, 42, provider.options["seed"])
	})

	t.Run("PrepareRequest formats chat completion request correctly", func(t *testing.T) {
		// Reset options
		provider.options = make(map[string]interface{})

		// Test with a basic prompt
		prompt := "Hello, world!"
		options := map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  100,
		}

		body, err := provider.PrepareRequest(prompt, options)
		assert.NoError(t, err)

		// Parse the request to verify structure
		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Verify model
		assert.Equal(t, model, req["model"])

		// Verify options
		assert.Equal(t, float64(0.7), req["temperature"])
		assert.Equal(t, float64(100), req["max_tokens"])

		// Verify messages format
		messages, ok := req["messages"].([]interface{})
		assert.True(t, ok)
		assert.Equal(t, 1, len(messages))

		userMsg := messages[0].(map[string]interface{})
		assert.Equal(t, "user", userMsg["role"])
		assert.Equal(t, "Hello, world!", userMsg["content"])
	})

	t.Run("PrepareRequest handles fallback models", func(t *testing.T) {
		options := map[string]interface{}{
			"fallback_models": []string{"openai/gpt-4o", "mistral/mistral-large"},
		}

		body, err := provider.PrepareRequest("test prompt", options)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check fallback models were set correctly
		models, ok := req["models"].([]interface{})
		assert.True(t, ok)
		assert.Equal(t, 3, len(models))
		assert.Equal(t, model, models[0])
		assert.Equal(t, "openai/gpt-4o", models[1])
		assert.Equal(t, "mistral/mistral-large", models[2])

		// Check fallback_models was removed
		_, exists := req["fallback_models"]
		assert.False(t, exists)
	})

	t.Run("PrepareRequest handles auto routing", func(t *testing.T) {
		options := map[string]interface{}{
			"auto_route": true,
		}

		body, err := provider.PrepareRequest("test prompt", options)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check model was set to auto router
		assert.Equal(t, "openrouter/auto", req["model"])

		// Check auto_route was removed
		_, exists := req["auto_route"]
		assert.False(t, exists)
	})

	t.Run("PrepareRequest handles provider preferences", func(t *testing.T) {
		providerPrefs := map[string]interface{}{
			"openai": map[string]interface{}{
				"weight": 2.0,
			},
		}

		options := map[string]interface{}{
			"provider_preferences": providerPrefs,
		}

		body, err := provider.PrepareRequest("test prompt", options)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check provider preferences were set correctly
		provider, ok := req["provider"].(map[string]interface{})
		assert.True(t, ok)
		assert.Equal(t, providerPrefs, provider)

		// Check provider_preferences was removed
		_, exists := req["provider_preferences"]
		assert.False(t, exists)
	})

	t.Run("PrepareRequest handles tools", func(t *testing.T) {
		tools := []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather",
					"parameters": map[string]interface{}{
						"type":       "object",
						"properties": map[string]interface{}{},
					},
				},
			},
		}

		options := map[string]interface{}{
			"tools":       tools,
			"tool_choice": "auto",
		}

		body, err := provider.PrepareRequest("test prompt", options)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check tools were set correctly
		reqTools, ok := req["tools"].([]interface{})
		assert.True(t, ok)
		assert.Equal(t, 1, len(reqTools))

		// Check tool_choice was set correctly
		assert.Equal(t, "auto", req["tool_choice"])
	})

	t.Run("PrepareRequestWithSchema includes JSON schema", func(t *testing.T) {
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "string",
				},
			},
		}

		body, err := provider.PrepareRequestWithSchema("test prompt", nil, schema)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check response format was set correctly
		responseFormat, ok := req["response_format"].(map[string]interface{})
		assert.True(t, ok)
		assert.Equal(t, "json_object", responseFormat["type"])
		assert.Equal(t, schema, responseFormat["schema"])
	})

	t.Run("PrepareRequestWithMessages formats messages correctly", func(t *testing.T) {
		messages := []types.MemoryMessage{
			{Role: "system", Content: "You are a helpful assistant"},
			{Role: "user", Content: "Hello, world!"},
			{Role: "assistant", Content: "How can I help you?"},
			{Role: "user", Content: "Tell me a joke"},
		}

		body, err := provider.PrepareRequestWithMessages(messages, nil)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check messages were formatted correctly
		reqMessages, ok := req["messages"].([]interface{})
		assert.True(t, ok)
		assert.Equal(t, 4, len(reqMessages))

		// Check roles are preserved
		assert.Equal(t, "system", reqMessages[0].(map[string]interface{})["role"])
		assert.Equal(t, "user", reqMessages[1].(map[string]interface{})["role"])
		assert.Equal(t, "assistant", reqMessages[2].(map[string]interface{})["role"])
		assert.Equal(t, "user", reqMessages[3].(map[string]interface{})["role"])

		// Check content is preserved
		assert.Equal(t, "You are a helpful assistant", reqMessages[0].(map[string]interface{})["content"])
		assert.Equal(t, "Hello, world!", reqMessages[1].(map[string]interface{})["content"])
		assert.Equal(t, "How can I help you?", reqMessages[2].(map[string]interface{})["content"])
		assert.Equal(t, "Tell me a joke", reqMessages[3].(map[string]interface{})["content"])
	})

	t.Run("PrepareCompletionRequest formats prompt correctly", func(t *testing.T) {
		prompt := "Write a poem about nature"
		options := map[string]interface{}{
			"temperature": 0.8,
			"max_tokens":  200,
		}

		body, err := provider.PrepareCompletionRequest(prompt, options)
		assert.NoError(t, err)

		var req map[string]interface{}
		err = json.Unmarshal(body, &req)
		assert.NoError(t, err)

		// Check prompt was set correctly
		assert.Equal(t, prompt, req["prompt"])

		// Check options were set correctly
		assert.Equal(t, float64(0.8), req["temperature"])
		assert.Equal(t, float64(200), req["max_tokens"])
	})

	t.Run("ParseResponse handles chat completion response", func(t *testing.T) {
		responseBody := `{
			"id": "gen-123",
			"choices": [
				{
					"message": {
						"content": "This is a test response",
						"role": "assistant"
					},
					"finish_reason": "stop",
					"native_finish_reason": "stop"
				}
			],
			"model": "anthropic/claude-3-5-sonnet"
		}`

		response, err := provider.ParseResponse([]byte(responseBody))
		assert.NoError(t, err)
		assert.Equal(t, "This is a test response", response)
	})

	t.Run("ParseResponse handles text completion response", func(t *testing.T) {
		responseBody := `{
			"choices": [
				{
					"text": "This is a test text completion"
				}
			]
		}`

		response, err := provider.ParseResponse([]byte(responseBody))
		assert.NoError(t, err)
		assert.Equal(t, "This is a test text completion", response)
	})

	t.Run("ParseStreamResponse handles streaming response", func(t *testing.T) {
		streamChunk := `{
			"id": "gen-123",
			"choices": [
				{
					"delta": {
						"content": "Hello",
						"role": "assistant"
					},
					"finish_reason": null
				}
			],
			"model": "anthropic/claude-3-5-sonnet"
		}`

		content, err := provider.ParseStreamResponse([]byte(streamChunk))
		assert.NoError(t, err)
		assert.Equal(t, "Hello", content)
	})

	t.Run("ParseStreamResponse handles empty delta", func(t *testing.T) {
		streamChunk := `{
			"id": "gen-123",
			"choices": [
				{
					"delta": {
						"role": "assistant"
					},
					"finish_reason": null
				}
			],
			"model": "anthropic/claude-3-5-sonnet"
		}`

		content, err := provider.ParseStreamResponse([]byte(streamChunk))
		assert.NoError(t, err)
		assert.Equal(t, "", content)
	})

	t.Run("HandleFunctionCalls identifies and returns function calls", func(t *testing.T) {
		responseBody := `{
			"id": "gen-123",
			"choices": [
				{
					"message": {
						"content": null,
						"role": "assistant",
						"tool_calls": [
							{
								"id": "call_123",
								"type": "function",
								"function": {
									"name": "get_weather",
									"arguments": "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}"
								}
							}
						]
					},
					"finish_reason": "tool_calls"
				}
			],
			"model": "openai/gpt-4o"
		}`

		result, err := provider.HandleFunctionCalls([]byte(responseBody))
		assert.NoError(t, err)
		assert.NotNil(t, result)

		// Check that the original response was returned
		assert.True(t, strings.Contains(string(result), "get_weather"))
		assert.True(t, strings.Contains(string(result), "San Francisco"))
	})

	t.Run("SetExtraHeaders adds custom headers", func(t *testing.T) {
		extraHeaders := map[string]string{
			"X-Custom-Header": "CustomValue",
		}

		provider.SetExtraHeaders(extraHeaders)
		headers := provider.Headers()

		assert.Equal(t, "CustomValue", headers["X-Custom-Header"])
	})

	t.Run("Reasoning tokens are enabled correctly", func(t *testing.T) {
		// Reset options
		provider.options = make(map[string]interface{})

		// Enable reasoning
		provider.options["enable_reasoning"] = true

		// Set defaults to trigger the transforms setup
		provider.SetDefaultOptions(&config.Config{})

		// Check transforms includes reasoning
		transforms, ok := provider.options["transforms"].([]string)
		assert.True(t, ok)
		assert.Contains(t, transforms, "reasoning")
	})
}
