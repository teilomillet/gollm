package providers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProviderRegistration verifies that all expected providers are registered
// and accessible from the default registry.
//
// This test addresses GitHub issues #42 and #38 where providers like
// openrouter and azure-openai were returning "unknown provider" errors.
func TestProviderRegistration(t *testing.T) {
	// These providers should all be available from the default registry
	expectedProviders := []string{
		"openai",
		"anthropic",
		"groq",
		"ollama",
		"mistral",
		"cohere",
		"deepseek",
		"google-openai",
		"openrouter",    // Registered via init() - was failing in issue #42
		"azure-openai",  // Has config but needs constructor - was failing in issue #38
		"aliyun",        // Alibaba Cloud DashScope - issue #52
	}

	registry := GetDefaultRegistry()

	for _, providerName := range expectedProviders {
		t.Run(providerName, func(t *testing.T) {
			provider, err := registry.Get(providerName, "test-api-key", "test-model", nil)
			require.NoError(t, err, "Provider %q should be registered in default registry", providerName)
			assert.NotNil(t, provider, "Provider %q should return a valid instance", providerName)
			assert.Equal(t, providerName, provider.Name(), "Provider name should match")
		})
	}
}

// TestNewProviderRegistryContainsKnownProviders verifies that NewProviderRegistry
// includes all providers in knownProviders map.
func TestNewProviderRegistryContainsKnownProviders(t *testing.T) {
	// These are the providers defined in knownProviders map
	knownProviders := []string{
		"openai",
		"anthropic",
		"groq",
		"ollama",
		"mistral",
		"cohere",
		"deepseek",
		"google-openai",
		"azure-openai",
		"aliyun",
	}

	registry := NewProviderRegistry()

	for _, providerName := range knownProviders {
		t.Run(providerName, func(t *testing.T) {
			provider, err := registry.Get(providerName, "test-api-key", "test-model", nil)
			require.NoError(t, err, "Provider %q should be in knownProviders", providerName)
			assert.NotNil(t, provider)
		})
	}
}

// TestOpenRouterRegisteredViaInit specifically tests that OpenRouter,
// which registers itself via init(), is available in the default registry.
// This is the core test for GitHub issue #42.
func TestOpenRouterRegisteredViaInit(t *testing.T) {
	registry := GetDefaultRegistry()

	provider, err := registry.Get("openrouter", "test-api-key", "openai/gpt-4", nil)
	require.NoError(t, err, "OpenRouter should be registered via init() in default registry")
	assert.NotNil(t, provider)
	assert.Equal(t, "openrouter", provider.Name())
}

// TestAzureOpenAIAvailable tests that azure-openai provider is available.
// This is the core test for GitHub issue #38.
func TestAzureOpenAIAvailable(t *testing.T) {
	registry := GetDefaultRegistry()

	provider, err := registry.Get("azure-openai", "test-api-key", "gpt-4", nil)
	require.NoError(t, err, "Azure OpenAI should be available in default registry")
	assert.NotNil(t, provider)
	assert.Equal(t, "azure-openai", provider.Name())
}

// TestProviderConfigsExist verifies that provider configs are registered
// for providers that need them.
func TestProviderConfigsExist(t *testing.T) {
	providersWithConfigs := []string{
		"openai",
		"azure-openai",
		"anthropic",
		"groq",
		"ollama",
		"deepseek",
		"google-openai",
		"aliyun",
	}

	registry := GetDefaultRegistry()

	for _, providerName := range providersWithConfigs {
		t.Run(providerName, func(t *testing.T) {
			config, exists := registry.GetProviderConfig(providerName)
			assert.True(t, exists, "Config for %q should exist", providerName)
			assert.Equal(t, providerName, config.Name, "Config name should match provider name")
		})
	}
}
