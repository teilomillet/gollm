package llm

import (
	"fmt"
	"sync"
)

type ProviderConstructor func(apiKey, model string) Provider

var (
	providerRegistry = make(map[string]ProviderConstructor)
	registryMutex    = &sync.RWMutex{}
)

// RegisterProvider adds a new provider to the registry
func RegisterProvider(name string, constructor ProviderConstructor) {
	registryMutex.Lock()
	defer registryMutex.Unlock()
	providerRegistry[name] = constructor
}

// GetProvider returns a provider instance based on the name
func GetProvider(name, apiKey, model string) (Provider, error) {
	registryMutex.RLock()
	constructor, exists := providerRegistry[name]
	registryMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	return constructor(apiKey, model), nil
}
