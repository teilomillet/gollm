package providers

import (
	"fmt"
	"sync"

	"github.com/teilomillet/gollm/config"
)

type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
	SetExtraHeaders(extraHeaders map[string]string)
	HandleFunctionCalls(body []byte) ([]byte, error)
	SupportsJSONSchema() bool
	SetDefaultOptions(config *config.Config)
	SetOption(key string, value interface{})
}

type ProviderConstructor func(apiKey, model string, extraHeaders map[string]string) Provider

type ProviderRegistry struct {
	providers map[string]ProviderConstructor
	mutex     sync.RWMutex
}

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

func (pr *ProviderRegistry) Register(name string, constructor ProviderConstructor) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()
	pr.providers[name] = constructor
}

func (pr *ProviderRegistry) Get(name, apiKey, model string, extraHeaders map[string]string) (Provider, error) {
	pr.mutex.RLock()
	constructor, exists := pr.providers[name]
	pr.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	return constructor(apiKey, model, extraHeaders), nil
}
