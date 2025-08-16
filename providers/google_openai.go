// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"github.com/weave-labs/gollm/config"
)

// GoogleProvider implements the Provider interface for Google's Gemini API through the
// OpenAI-compatible endpoint. Accordingly, it inherits from OpenAIProvider
type GoogleProvider struct {
	OpenAIProvider
}

// NewGoogleProvider creates a new Google provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Gemini API key for authentication
//   - model: The model to use (e.g., "gemini-2.0-flash")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Google Provider instance
func NewGoogleProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &GoogleProvider{
		OpenAIProvider: *NewOpenAIProvider(apiKey, model, extraHeaders).(*OpenAIProvider),
	}

	return provider
}

// Name returns "google-openai" as the provider identifier.
func (p *GoogleProvider) Name() string {
	return "google-openai"
}

// Endpoint returns the Gemini API endpoint URL for generating content.
func (p *GoogleProvider) Endpoint() string {
	return "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *GoogleProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens)
}
