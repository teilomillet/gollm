// Package providers implement various Language Learning Model (LLM) provider interfaces
// and their concrete implementations. It supports multiple providers including OpenAI,
// Anthropic, Groq, Ollama, and Mistral, providing a unified interface for interacting
// with different LLM services.
package providers

import (
	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// Provider defines the complete interface that all LLM providers must implement.
//
//nolint:interfacebloat // This will be refactored in the future to a capabilities model.
type Provider interface {
	// Core identification and configuration
	Name() string
	Endpoint() string
	Headers() map[string]string
	SetExtraHeaders(extraHeaders map[string]string)
	SetDefaultOptions(cfg *config.Config)
	SetOption(key string, value any)
	SetLogger(logger logging.Logger)

	// Request preparation - unified interface
	PrepareRequest(req *Request, options map[string]any) ([]byte, error)
	PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error)

	// Response handling
	ParseResponse(body []byte) (*Response, error)
	ParseStreamResponse(chunk []byte) (*Response, error)

	// Capability checks
	SupportsStreaming() bool
	SupportsStructuredResponse() bool
	SupportsFunctionCalling() bool
}

// ProviderConfig holds the configuration for a provider
type ProviderConfig struct {
	// Name is the provider identifier
	Name string

	// Endpoint is the API endpoint URL
	Endpoint string

	// AuthHeader is the header key used for authentication
	AuthHeader string

	// AuthPrefix is the prefix to use before the API key (e.g., "Bearer ")
	AuthPrefix string

	// RequiredHeaders are additional headers always needed
	RequiredHeaders map[string]string

	// EndpointParams are URL parameters to add to the endpoint
	EndpointParams map[string]string

	// ResponseFormat defines how to parse the response
	// If empty, uses the default parser for the provider type
	ResponseFormat string

	// SupportsStructuredResponse indicates if JSON schema validation is supported
	SupportsStructuredResponse bool

	// SupportsStreaming indicates if streaming is supported
	SupportsStreaming bool

	// SupportsFunctionCalling indicates if function calling is supported
	SupportsFunctionCalling bool
}

// ProviderConstructor defines a function type for creating new provider instances.
// Each provider implementation must provide a constructor function of this type.
type ProviderConstructor func(apiKey, model string, extraHeaders map[string]string) Provider
