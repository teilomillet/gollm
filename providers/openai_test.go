package providers

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestNeedsMaxCompletionTokens verifies that the needsMaxCompletionTokens function
// correctly identifies models that require max_completion_tokens instead of max_tokens
func TestNeedsMaxCompletionTokens(t *testing.T) {
	testCases := []struct {
		modelName      string
		expectedResult bool
		description    string
	}{
		{"gpt-4", false, "Standard GPT-4 model should not use max_completion_tokens"},
		{"gpt-4-turbo", false, "GPT-4 Turbo model should not use max_completion_tokens"},
		{"gpt-3.5-turbo", false, "GPT-3.5 Turbo model should not use max_completion_tokens"},
		{"o1-preview", true, "o1-preview model should use max_completion_tokens"},
		{"o-preview", true, "o-preview model should use max_completion_tokens"},
		{"gpt-4o", true, "GPT-4o model should use max_completion_tokens"},
		{"gpt-4o-mini", true, "GPT-4o mini model should use max_completion_tokens"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// Create a provider with the test model
			provider := NewOpenAIProvider("fake-api-key", tc.modelName, nil)

			// Test the needsMaxCompletionTokens function
			result := provider.needsMaxCompletionTokens()
			assert.Equal(
				t,
				tc.expectedResult,
				result,
				"needsMaxCompletionTokens returned unexpected result for model %s",
				tc.modelName,
			)
		})
	}
}
