package providers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm/config"
)

func TestMockProvider(t *testing.T) {
	// Create a new mock provider
	provider := NewMockProvider("http://mock.api", "mock-model", nil)
	mockProvider := provider.(*MockProvider)

	// Test basic provider properties
	assert.Equal(t, "mock", provider.Name())
	assert.Equal(t, "http://mock.api", provider.Endpoint())
	assert.True(t, provider.SupportsJSONSchema())

	// Test mock response
	expectedResponse := "custom mock response"
	mockProvider.SetMockResponse(expectedResponse)

	response, err := provider.ParseResponse([]byte("{}"))
	assert.NoError(t, err)
	assert.Equal(t, expectedResponse, response)

	// Test error handling
	mockProvider.SetMockError(true, "mock error")
	_, err = provider.ParseResponse([]byte("{}"))
	assert.Error(t, err)
	assert.Equal(t, "mock error", err.Error())

	// Test request preparation
	mockProvider.SetMockError(false, "")
	reqBody, err := provider.PrepareRequest("test prompt", map[string]interface{}{
		"temperature": 0.7,
	})
	assert.NoError(t, err)
	assert.Contains(t, string(reqBody), "test prompt")
	assert.Contains(t, string(reqBody), "temperature")

	// Test default options
	cfg := &config.Config{
		Temperature: 0.8,
		MaxTokens:   100,
	}
	provider.SetDefaultOptions(cfg)
	assert.Equal(t, 0.8, mockProvider.options["temperature"])
	assert.Equal(t, 100, mockProvider.options["max_tokens"])
}

func TestMockProviderResponses(t *testing.T) {
	provider := NewMockProvider("http://mock.api", "mock-model", nil)
	mockProvider := provider.(*MockProvider)

	// Test sequential responses
	responses := []string{
		"First response",
		"Second response",
		"Third response",
	}

	// Test without looping
	mockProvider.SetResponses(responses, false)

	// Check each response in sequence
	for _, expected := range responses {
		response, err := provider.ParseResponse([]byte("{}"))
		assert.NoError(t, err)
		assert.Equal(t, expected, response)
	}

	// Verify exhaustion error
	_, err := provider.ParseResponse([]byte("{}"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exhausted")

	// Test with looping enabled
	mockProvider.SetResponses(responses, true)

	// Check that responses loop
	for i := 0; i < len(responses)*2; i++ {
		response, err := provider.ParseResponse([]byte("{}"))
		assert.NoError(t, err)
		assert.Equal(t, responses[i%len(responses)], response)
	}
}

func TestMockProviderResponsePatterns(t *testing.T) {
	provider := NewMockProvider("http://mock.api", "mock-model", nil)
	mockProvider := provider.(*MockProvider)

	// Test different response patterns
	testCases := []struct {
		name      string
		responses []string
		loop      bool
		calls     int
		expected  []string
		errorAt   int // -1 for no error expected
	}{
		{
			name:      "Single response",
			responses: []string{"one"},
			loop:      true,
			calls:     3,
			expected:  []string{"one", "one", "one"},
			errorAt:   -1,
		},
		{
			name:      "Multiple responses no loop",
			responses: []string{"first", "second"},
			loop:      false,
			calls:     3,
			expected:  []string{"first", "second"},
			errorAt:   2,
		},
		{
			name:      "Empty response list",
			responses: []string{},
			loop:      false,
			calls:     2,
			expected:  []string{"This is a mock response", "This is a mock response"},
			errorAt:   -1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockProvider.SetResponses(tc.responses, tc.loop)

			for i := 0; i < tc.calls; i++ {
				response, err := provider.ParseResponse([]byte("{}"))

				if tc.errorAt == i {
					assert.Error(t, err)
					return
				}

				assert.NoError(t, err)
				if i < len(tc.expected) {
					assert.Equal(t, tc.expected[i], response)
				}
			}
		})
	}
}
