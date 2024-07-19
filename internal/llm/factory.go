package llm

import (
	"fmt"
	"os"
)

// getProvider returns the appropriate Provider based on the provider name and model
func getProvider(name, model string) (Provider, error) {
	envVar := fmt.Sprintf("%s_API_KEY", name)
	apiKey := os.Getenv(envVar)
	if apiKey == "" {
		return nil, fmt.Errorf("%s not set in environment", envVar)
	}

	return GetProvider(name, apiKey, model)
}
