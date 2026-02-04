package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewCohereProvider(t *testing.T) {
	provider := NewCohereProvider("test-api-key", "command-r-plus", nil)

	assert.Equal(t, "cohere", provider.Name())
	assert.Equal(t, "https://api.cohere.com/v2/chat", provider.Endpoint())
}

func TestNewCohereProviderWithURL(t *testing.T) {
	testCases := []struct {
		name             string
		baseURL          string
		expectedEndpoint string
	}{
		{
			name:             "standard base URL",
			baseURL:          "https://api.cohere.com",
			expectedEndpoint: "https://api.cohere.com/v2/chat",
		},
		{
			name:             "base URL with trailing slash",
			baseURL:          "https://api.cohere.com/",
			expectedEndpoint: "https://api.cohere.com/v2/chat",
		},
		{
			name:             "custom proxy URL",
			baseURL:          "https://my-proxy.example.com",
			expectedEndpoint: "https://my-proxy.example.com/v2/chat",
		},
		{
			name:             "custom proxy URL with trailing slash",
			baseURL:          "https://my-proxy.example.com/",
			expectedEndpoint: "https://my-proxy.example.com/v2/chat",
		},
		{
			name:             "localhost URL",
			baseURL:          "http://localhost:8080",
			expectedEndpoint: "http://localhost:8080/v2/chat",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewCohereProviderWithURL("test-api-key", "command-r-plus", tc.baseURL, nil)

			assert.Equal(t, "cohere", provider.Name())
			assert.Equal(t, tc.expectedEndpoint, provider.Endpoint())
		})
	}
}

func TestCohereHeaders(t *testing.T) {
	provider := NewCohereProvider("test-api-key", "command-r-plus", nil)

	headers := provider.Headers()

	assert.Equal(t, "application/json", headers["Content-Type"])
	assert.Equal(t, "Bearer test-api-key", headers["Authorization"])
}

func TestCohereHeadersWithExtraHeaders(t *testing.T) {
	extraHeaders := map[string]string{
		"X-Custom-Header": "custom-value",
	}
	provider := NewCohereProvider("test-api-key", "command-r-plus", extraHeaders)

	headers := provider.Headers()

	assert.Equal(t, "application/json", headers["Content-Type"])
	assert.Equal(t, "Bearer test-api-key", headers["Authorization"])
	assert.Equal(t, "custom-value", headers["X-Custom-Header"])
}

func TestCoherePrepareRequestFiltersUnsupportedParams(t *testing.T) {
	provider := NewCohereProvider("test-api-key", "command-r-plus", nil).(*CohereProvider)

	// Include both supported and unsupported parameters
	options := map[string]any{
		"temperature":   0.7,
		"max_tokens":    100,
		"top_p":         0.9,  // unsupported by Cohere v2
		"unsupported":   true, // unsupported
		"system_prompt": "test",
	}

	requestBytes, err := provider.PrepareRequest("Hello", options)
	require.NoError(t, err)

	var request map[string]any
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	// Supported params should be present
	assert.Equal(t, 0.7, request["temperature"])
	assert.Equal(t, float64(100), request["max_tokens"])

	// Unsupported params should be filtered out
	_, hasTopP := request["top_p"]
	assert.False(t, hasTopP, "top_p should be filtered out")

	_, hasUnsupported := request["unsupported"]
	assert.False(t, hasUnsupported, "unsupported should be filtered out")
}

func TestCoherePrepareRequestWithSchemaFiltersUnsupportedParams(t *testing.T) {
	provider := NewCohereProvider("test-api-key", "command-r-plus", nil).(*CohereProvider)

	options := map[string]any{
		"temperature": 0.7,
		"top_p":       0.9, // unsupported
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	requestBytes, err := provider.PrepareRequestWithSchema("Hello", options, schema)
	require.NoError(t, err)

	var request map[string]any
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	// Supported params should be present
	assert.Equal(t, 0.7, request["temperature"])

	// Unsupported params should be filtered out
	_, hasTopP := request["top_p"]
	assert.False(t, hasTopP, "top_p should be filtered out")

	// Schema should be included
	responseFormat, ok := request["response_format"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "json_object", responseFormat["type"])
}

func TestCoherePrepareStreamRequestIncludesStream(t *testing.T) {
	provider := NewCohereProvider("test-api-key", "command-r-plus", nil).(*CohereProvider)

	options := map[string]interface{}{
		"temperature": 0.7,
	}

	requestBytes, err := provider.PrepareStreamRequest("Hello", options)
	require.NoError(t, err)

	var request map[string]any
	err = json.Unmarshal(requestBytes, &request)
	require.NoError(t, err)

	assert.Equal(t, true, request["stream"])
	assert.Equal(t, 0.7, request["temperature"])
}
