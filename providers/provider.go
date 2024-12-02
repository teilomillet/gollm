// Package providers implements various Language Learning Model (LLM) provider interfaces
// and their concrete implementations. It supports multiple providers including OpenAI,
// Anthropic, Groq, Ollama, and Mistral, providing a unified interface for interacting
// with different LLM services.
package providers

import (
	"fmt"
	"sync"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// Provider defines the interface that all LLM providers must implement.
// This interface abstracts the common operations needed to interact with
// different LLM services, allowing for a unified approach to LLM integration.
type Provider interface {
	// Name returns the provider's identifier (e.g., "openai", "anthropic").
	Name() string

	// Endpoint returns the API endpoint URL for the provider.
	Endpoint() string

	// Headers returns the HTTP headers required for API requests.
	// This typically includes authentication and content-type headers.
	Headers() map[string]string

	// PrepareRequest creates the request body for an API call.
	// It takes a prompt string and additional options, returning the serialized request body.
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)

	// PrepareRequestWithSchema creates a request body that includes JSON schema validation.
	// This is used for providers that support structured output validation.
	PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error)

	// ParseResponse extracts the generated text from the API response.
	// It handles provider-specific response formats and error cases.
	ParseResponse(body []byte) (string, error)

	// SetExtraHeaders configures additional HTTP headers for API requests.
	// This is useful for provider-specific features or authentication methods.
	SetExtraHeaders(extraHeaders map[string]string)

	// HandleFunctionCalls processes function calling capabilities.
	// This is particularly relevant for providers that support function/tool calling.
	HandleFunctionCalls(body []byte) ([]byte, error)

	// SupportsJSONSchema indicates whether the provider supports native JSON schema validation.
	SupportsJSONSchema() bool

	// SetDefaultOptions configures provider-specific defaults from the global configuration.
	SetDefaultOptions(config *config.Config)

	// SetOption sets a specific option for the provider (e.g., temperature, max_tokens).
	SetOption(key string, value interface{})

	// SetLogger configures the logger for the provider instance.
	SetLogger(logger utils.Logger)
}

// ProviderConstructor defines a function type for creating new provider instances.
// Each provider implementation must provide a constructor function of this type.
type ProviderConstructor func(apiKey, model string, extraHeaders map[string]string) Provider

// ProviderRegistry manages the registration and retrieval of LLM providers.
// It provides thread-safe access to provider constructors and supports
// dynamic provider registration.
type ProviderRegistry struct {
	providers map[string]ProviderConstructor
	mutex     sync.RWMutex
}

// NewProviderRegistry creates a new provider registry with the specified providers.
// If no providers are specified, all known providers are registered by default.
//
// Supported providers:
//   - "openai": OpenAI's GPT models
//   - "anthropic": Anthropic's Claude models
//   - "groq": Groq's LLM services
//   - "ollama": Local LLM deployment
//   - "mistral": Mistral AI's models
//   - "mock": Mock provider for testing
//
// Example usage:
//
//	// Register all providers
//	registry := NewProviderRegistry()
//
//	// Register specific providers
//	registry := NewProviderRegistry("openai", "anthropic")
func NewProviderRegistry(providerNames ...string) *ProviderRegistry {
	registry := &ProviderRegistry{
		providers: make(map[string]ProviderConstructor),
	}

	// Register all known providers
	knownProviders := map[string]ProviderConstructor{
		"openai":    NewOpenAIProvider,
		"anthropic": NewAnthropicProvider,
		"groq":      NewGroqProvider,
		"ollama":    NewOllamaProvider,
		"mistral":   NewMistralProvider,
		"mock":      NewMockProvider,
		// Add other providers here as they are implemented
	}

	if len(providerNames) == 0 {
		// If no specific providers are requested, register all known providers
		for name, constructor := range knownProviders {
			registry.Register(name, constructor)
		}
	} else {
		// Register only the requested providers
		for _, name := range providerNames {
			if constructor, ok := knownProviders[name]; ok {
				registry.Register(name, constructor)
			} else {
				// You might want to handle unknown provider names differently
				fmt.Printf("Warning: Unknown provider '%s' requested\n", name)
			}
		}
	}

	return registry
}

// Register adds a new provider constructor to the registry.
// This method is thread-safe and can be used to dynamically add new providers.
//
// Parameters:
//   - name: The identifier for the provider (e.g., "openai")
//   - constructor: A function that creates new instances of the provider
func (pr *ProviderRegistry) Register(name string, constructor ProviderConstructor) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()
	pr.providers[name] = constructor
}

// Get retrieves a provider instance by name.
// It creates a new provider instance using the registered constructor.
//
// Parameters:
//   - name: The provider identifier
//   - apiKey: The API key for authentication
//   - model: The specific model to use
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Provider instance
//   - An error if the provider is not found
//
// Example:
//
//	provider, err := registry.Get("openai", "sk-...", "gpt-4", nil)
func (pr *ProviderRegistry) Get(name, apiKey, model string, extraHeaders map[string]string) (Provider, error) {
	pr.mutex.RLock()
	constructor, exists := pr.providers[name]
	pr.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	return constructor(apiKey, model, extraHeaders), nil
}
