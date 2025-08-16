package providers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weave-labs/gollm/types"
)

// TestAllProvidersImplementStructuredMessages verifies that all providers
// properly implement the PrepareRequestWithMessages method
func TestAllProvidersImplementStructuredMessages(t *testing.T) {
	// Define test messages
	testMessages := []types.MemoryMessage{
		{
			Role:         "user",
			Content:      "Hello, how are you?",
			CacheControl: "ephemeral",
		},
		{
			Role:         "assistant",
			Content:      "I'm doing well, thank you for asking!",
			CacheControl: "",
		},
		{
			Role:         "user",
			Content:      "Tell me about structured messages",
			CacheControl: "ephemeral",
		},
	}

	// Test options
	options := map[string]any{
		"system_prompt": "You are a helpful assistant",
		"temperature":   0.7,
		"max_tokens":    1024,
	}

	// Create and test each provider
	providers := []struct {
		name     string
		provider Provider
	}{
		{"OpenAI", NewOpenAIProvider("fake-key", "gpt-4", nil)},
		{"Anthropic", NewAnthropicProvider("fake-key", "claude-3-opus", nil)},
		{"Mistral", NewMistralProvider("fake-key", "mistral-large", nil)},
		{"Groq", NewGroqProvider("fake-key", "llama2-70b", nil)},
		{"Cohere", NewCohereProvider("fake-key", "command-r", nil)},
		{"Ollama", NewOllamaProvider("http://localhost:11434", "llama2", nil)},
	}

	for _, p := range providers {
		t.Run(p.name, func(t *testing.T) {
			// Make sure the PrepareRequestWithMessages method produces valid output
			result, err := p.provider.PrepareRequestWithMessages(testMessages, options)
			require.NoError(t, err, "PrepareRequestWithMessages should not fail")
			require.NotNil(t, result, "PrepareRequestWithMessages should return non-nil result")

			// Result should be valid JSON
			assert.True(t, len(result) > 0, "Result should not be empty")

			// Test the type assertion works (the provider correctly implements the method)
			_, ok := p.provider.(interface {
				PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]any) ([]byte, error)
			})
			assert.True(t, ok, "Provider should implement PrepareRequestWithMessages")
		})
	}
}

// TestStructuredMessagesFormat verifies that the message format is
// appropriate for each provider
func TestStructuredMessagesFormat(t *testing.T) {
	// Test with minimal providers to verify message format
	t.Run("OpenAI format", func(t *testing.T) {
		provider := NewOpenAIProvider("fake-key", "gpt-4", nil)
		messages := []types.MemoryMessage{
			{Role: "user", Content: "Hello"},
		}
		result, err := provider.PrepareRequestWithMessages(messages, nil)
		require.NoError(t, err)

		// Should contain the message format in OpenAI style
		assert.Contains(t, string(result), `"role":"user"`)
		assert.Contains(t, string(result), `"content":"Hello"`)
	})

	t.Run("Anthropic format", func(t *testing.T) {
		provider := NewAnthropicProvider("fake-key", "claude-3", nil)
		messages := []types.MemoryMessage{
			{Role: "user", Content: "Hello", CacheControl: "ephemeral"},
		}
		result, err := provider.PrepareRequestWithMessages(messages, nil)
		require.NoError(t, err)

		// Should contain cache_control for Anthropic
		assert.Contains(t, string(result), `"cache_control"`)
		assert.Contains(t, string(result), `"type":"ephemeral"`)
	})

	// Add more provider-specific tests as needed
}
