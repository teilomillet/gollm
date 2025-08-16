// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"github.com/weave-labs/gollm/config"
)

// DeepSeekProvider implements the Provider interface for DeepSeek's API.
// It inherits from OpenAIProvider since DeepSeek uses an OpenAI-compatible API.
type DeepSeekProvider struct {
	OpenAIProvider
}

// NewDeepSeekProvider creates a new DeepSeek provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: DeepSeek API key for authentication
//   - model: The model to use (e.g., "deepseek-chat")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured DeepSeek Provider instance
func NewDeepSeekProvider(apiKey, model string, extraHeaders map[string]string) *DeepSeekProvider {
	// Create a base OpenAI provider directly
	baseProvider := &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
	}
	provider := &DeepSeekProvider{
		OpenAIProvider: *baseProvider,
	}
	// Override the endpoint
	return provider
}

// Name returns "deepseek" as the provider identifier.
// This is used to identify the provider in the system.
func (p *DeepSeekProvider) Name() string {
	return "deepseek"
}

// Endpoint returns the DeepSeek API endpoint URL.
// This is the URL used to make requests to the DeepSeek API.
func (p *DeepSeekProvider) Endpoint() string {
	return "https://api.deepseek.com/chat/completions"
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes setting options like temperature and max tokens based on the provided config.
//
// Parameters:
//   - config: The global configuration containing options to set
func (p *DeepSeekProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
	p.logger.Debug("Default options set", "temperature", cfg.Temperature, "max_tokens", cfg.MaxTokens)
}
