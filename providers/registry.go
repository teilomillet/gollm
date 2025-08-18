package providers

import (
	"fmt"
	"sync"
)

// ProviderRegistry manages the registration and retrieval of LLM providers.
// It provides thread-safe access to provider constructors and supports
// dynamic provider registration.
type ProviderRegistry struct {
	providers map[string]ProviderConstructor
	configs   map[string]ProviderConfig
	mutex     sync.RWMutex
}

// NewProviderRegistry creates a new provider registry with the specified providers.
// If no providers are specified, all known providers are registered by default.
func NewProviderRegistry(providerNames ...string) *ProviderRegistry {
	registry := &ProviderRegistry{
		providers: make(map[string]ProviderConstructor),
		configs:   make(map[string]ProviderConfig),
	}

	// Get known providers and configs
	knownProviders := getKnownProviders()
	standardConfigs := getStandardConfigs()

	// Store standard configs
	for name, config := range standardConfigs {
		registry.configs[name] = config
	}

	if len(providerNames) == 0 {
		// If no specific providers are requested, register all known providers
		for name, constructor := range knownProviders {
			registry.providers[name] = constructor
		}
	} else {
		// Otherwise, register only the requested providers
		for _, name := range providerNames {
			if constructor, ok := knownProviders[name]; ok {
				registry.providers[name] = constructor
			}
		}
	}

	return registry
}

// getKnownProviders returns all known provider constructors
func getKnownProviders() map[string]ProviderConstructor {
	return map[string]ProviderConstructor{
		"openai": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewOpenAIProvider(apiKey, model, extraHeaders)
		},
		"anthropic": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewAnthropicProvider(apiKey, model, extraHeaders)
		},
		"groq": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewGroqProvider(apiKey, model, extraHeaders)
		},
		"ollama": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewOllamaProvider(apiKey, model, extraHeaders)
		},
		"mistral": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewMistralProvider(apiKey, model, extraHeaders)
		},
		"cohere": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewCohereProvider(apiKey, model, extraHeaders)
		},
		"deepseek": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewDeepSeekProvider(apiKey, model, extraHeaders)
		},
		"google": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewGeminiProvider(apiKey, model, extraHeaders)
		},
		"gemini": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewGeminiProvider(apiKey, model, extraHeaders)
		},
		"openrouter": func(apiKey, model string, extraHeaders map[string]string) Provider {
			return NewOpenRouterProvider(apiKey, model, extraHeaders)
		},
	}
}

// getStandardConfigs returns standard provider configurations
func getStandardConfigs() map[string]ProviderConfig {
	configs := map[string]ProviderConfig{
		"openai": {
			Name:                       "openai",
			Endpoint:                   "https://api.openai.com/v1/chat/completions",
			AuthHeader:                 "Authorization",
			AuthPrefix:                 "Bearer ",
			RequiredHeaders:            map[string]string{"Content-Type": "application/json"},
			SupportsStructuredResponse: true,
			SupportsStreaming:          true,
		},
		"azure-openai": {
			Name: "azure-openai",
			// The user should configure an endpoint
			AuthHeader:                 "api-key",
			AuthPrefix:                 "",
			RequiredHeaders:            map[string]string{"Content-Type": "application/json"},
			SupportsStructuredResponse: true,
			SupportsStreaming:          true,
		},
		"anthropic": {
			Name:       "anthropic",
			Endpoint:   "https://api.anthropic.com/v1/messages",
			AuthHeader: "x-api-key",
			AuthPrefix: "",
			RequiredHeaders: map[string]string{
				"Content-Type":      "application/json",
				"anthropic-version": "2023-06-01",
			},
			SupportsStructuredResponse: true,
			SupportsStreaming:          true,
		},
		"groq": {
			Name:                       "groq",
			Endpoint:                   "https://api.groq.com/openai/v1/chat/completions",
			AuthHeader:                 "Authorization",
			AuthPrefix:                 "Bearer ",
			RequiredHeaders:            map[string]string{"Content-Type": "application/json"},
			SupportsStructuredResponse: true,
			SupportsStreaming:          true,
		},
		"google": {
			Name:                       "google",
			Endpoint:                   "https://generativelanguage.googleapis.com/v1beta/",
			AuthHeader:                 "Authorization",
			AuthPrefix:                 "Bearer ",
			RequiredHeaders:            map[string]string{"Content-Type": "application/json"},
			SupportsStructuredResponse: true,
			SupportsStreaming:          true,
		},
		"ollama": {
			Name:                       "ollama",
			Endpoint:                   "http://localhost:11434/api/generate",
			AuthHeader:                 "", // Ollama doesn't require authentication
			AuthPrefix:                 "",
			RequiredHeaders:            map[string]string{"Content-Type": "application/json"},
			SupportsStructuredResponse: false,
			SupportsStreaming:          true,
		},
		"deepseek": {
			Name:                       "deepseek",
			Endpoint:                   "https://api.deepseek.com/chat/completions",
			AuthHeader:                 "Authorization",
			AuthPrefix:                 "Bearer ",
			RequiredHeaders:            map[string]string{"Content-Type": "application/json"},
			SupportsStructuredResponse: true,
			SupportsStreaming:          true,
		},
	}

	return configs
}

// GetProviderConfig returns the configuration for a named provider
// Returns the config and a boolean indicating whether the provider was found
func (r *ProviderRegistry) GetProviderConfig(name string) (ProviderConfig, bool) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	cfg, exists := r.configs[name]
	return cfg, exists
}

// RegisterProviderConfig registers a new provider configuration
func (r *ProviderRegistry) RegisterProviderConfig(name string, cfg ProviderConfig) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	r.configs[name] = cfg
}

// IsKnownProvider returns true if the default registry recognizes the given provider name.
// A provider is considered known if it has a ProviderConfig registered in the registry.
// Note: This doesn't guarantee that a constructor is registered (i.e., it may not be instantiable).
func IsKnownProvider(name string) bool {
	_, ok := NewProviderRegistry().GetProviderConfig(name)
	return ok
}

// Register adds a new provider constructor to the registry.
// This method is thread-safe and can be used to dynamically add new providers.
func (r *ProviderRegistry) Register(name string, constructor ProviderConstructor) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	r.providers[name] = constructor
}

// Get retrieves a provider instance by name.
// It creates a new provider instance using the registered constructor.
func (r *ProviderRegistry) Get(name, apiKey, model string, extraHeaders map[string]string) (Provider, error) {
	r.mutex.RLock()
	constructor, exists := r.providers[name]
	r.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	return constructor(apiKey, model, extraHeaders), nil
}
