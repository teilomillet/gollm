package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewVLLMProvider(t *testing.T) {
	// apiKey parameter is ignored for vLLM (no auth required)
	provider := NewVLLMProvider("ignored-api-key", "Qwen/Qwen2-7B-Instruct", nil)

	assert.Equal(t, "vllm", provider.Name())
	assert.True(t, provider.SupportsJSONSchema())
	assert.True(t, provider.SupportsStreaming())
}

func TestVLLMEndpoint(t *testing.T) {
	testCases := []struct {
		name             string
		endpoint         string
		expectedEndpoint string
	}{
		{
			name:             "default endpoint",
			endpoint:         "http://localhost:8000",
			expectedEndpoint: "http://localhost:8000/v1/chat/completions",
		},
		{
			name:             "endpoint with trailing slash",
			endpoint:         "http://localhost:8000/",
			expectedEndpoint: "http://localhost:8000/v1/chat/completions",
		},
		{
			name:             "endpoint already has /v1",
			endpoint:         "http://localhost:8000/v1",
			expectedEndpoint: "http://localhost:8000/v1/chat/completions",
		},
		{
			name:             "endpoint with /v1 and trailing slash",
			endpoint:         "http://localhost:8000/v1/",
			expectedEndpoint: "http://localhost:8000/v1/chat/completions",
		},
		{
			name:             "full path already provided",
			endpoint:         "http://localhost:8000/v1/chat/completions",
			expectedEndpoint: "http://localhost:8000/v1/chat/completions",
		},
		{
			name:             "custom host and port",
			endpoint:         "http://192.168.1.100:9000",
			expectedEndpoint: "http://192.168.1.100:9000/v1/chat/completions",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewVLLMProvider("", "test-model", nil).(*VLLMProvider)
			provider.SetEndpoint(tc.endpoint)

			assert.Equal(t, tc.expectedEndpoint, provider.Endpoint())
		})
	}
}

func TestVLLMDefaultEndpoint(t *testing.T) {
	provider := NewVLLMProvider("", "test-model", nil)

	// Default endpoint should be localhost:8000
	assert.Equal(t, "http://localhost:8000/v1/chat/completions", provider.Endpoint())
}

func TestVLLMHeaders(t *testing.T) {
	provider := NewVLLMProvider("", "test-model", nil)

	headers := provider.Headers()

	// vLLM doesn't require Authorization header
	assert.Equal(t, "application/json", headers["Content-Type"])
	_, hasAuth := headers["Authorization"]
	assert.False(t, hasAuth, "vLLM should not have Authorization header")
}

func TestVLLMHeadersWithExtraHeaders(t *testing.T) {
	extraHeaders := map[string]string{
		"X-Custom-Header": "custom-value",
	}
	provider := NewVLLMProvider("", "test-model", extraHeaders)

	headers := provider.Headers()

	assert.Equal(t, "application/json", headers["Content-Type"])
	assert.Equal(t, "custom-value", headers["X-Custom-Header"])
}

func TestVLLMPrepareRequest(t *testing.T) {
	provider := NewVLLMProvider("", "test-model", nil).(*VLLMProvider)

	options := map[string]interface{}{
		"temperature":   0.7,
		"system_prompt": "You are a helpful assistant.",
	}

	requestBytes, err := provider.PrepareRequest("Hello", options)
	require.NoError(t, err)

	var request map[string]interface{}
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	assert.Equal(t, "test-model", request["model"])
	assert.Equal(t, 0.7, request["temperature"])

	messages, ok := request["messages"].([]interface{})
	require.True(t, ok)
	assert.Len(t, messages, 2) // system + user

	// First message should be system
	systemMsg := messages[0].(map[string]interface{})
	assert.Equal(t, "system", systemMsg["role"])
	assert.Equal(t, "You are a helpful assistant.", systemMsg["content"])

	// Second message should be user
	userMsg := messages[1].(map[string]interface{})
	assert.Equal(t, "user", userMsg["role"])
	assert.Equal(t, "Hello", userMsg["content"])
}

func TestVLLMPrepareRequestWithSchema(t *testing.T) {
	provider := NewVLLMProvider("", "test-model", nil).(*VLLMProvider)

	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
		},
	}

	requestBytes, err := provider.PrepareRequestWithSchema("Hello", nil, schema)
	require.NoError(t, err)

	var request map[string]interface{}
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	// Check response_format includes schema
	responseFormat, ok := request["response_format"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "json_schema", responseFormat["type"])

	jsonSchema, ok := responseFormat["json_schema"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "structured_response", jsonSchema["name"])
	assert.Equal(t, true, jsonSchema["strict"])
	assert.NotNil(t, jsonSchema["schema"])
}

func TestVLLMPrepareStreamRequest(t *testing.T) {
	provider := NewVLLMProvider("", "test-model", nil).(*VLLMProvider)

	options := map[string]interface{}{
		"temperature":   0.7,
		"system_prompt": "You are helpful.",
	}

	requestBytes, err := provider.PrepareStreamRequest("Hello", options)
	require.NoError(t, err)

	var request map[string]interface{}
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	assert.Equal(t, true, request["stream"])
	assert.Equal(t, 0.7, request["temperature"])

	messages, ok := request["messages"].([]interface{})
	require.True(t, ok)
	assert.Len(t, messages, 2) // system + user

	// First message should be system
	systemMsg := messages[0].(map[string]interface{})
	assert.Equal(t, "system", systemMsg["role"])
}

func TestVLLMPrepareStreamRequestWithoutSystemPrompt(t *testing.T) {
	provider := NewVLLMProvider("", "test-model", nil).(*VLLMProvider)

	requestBytes, err := provider.PrepareStreamRequest("Hello", nil)
	require.NoError(t, err)

	var request map[string]interface{}
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	messages, ok := request["messages"].([]interface{})
	require.True(t, ok)
	assert.Len(t, messages, 1) // only user message

	userMsg := messages[0].(map[string]interface{})
	assert.Equal(t, "user", userMsg["role"])
}
