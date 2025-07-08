package providers_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/guiperry/gollm_cerebras"
	"github.com/guiperry/gollm_cerebras/config"
	"github.com/guiperry/gollm_cerebras/providers"
)

// TestAzureOpenAIIntegration tests the Azure OpenAI provider integration
// This test is skipped by default. To run it, set the following environment variables:
// - AZURE_OPENAI_API_KEY
// - AZURE_OPENAI_RESOURCE_NAME
// - AZURE_OPENAI_DEPLOYMENT_NAME
// - RUN_INTEGRATION_TESTS=true
func TestAzureOpenAIIntegration(t *testing.T) {
	// Check if integration tests should run
	if os.Getenv("RUN_INTEGRATION_TESTS") != "true" {
		t.Skip("Skipping Azure OpenAI integration test. Set RUN_INTEGRATION_TESTS=true to run")
	}

	// Check required env vars
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("AZURE_OPENAI_API_KEY environment variable is not set")
	}

	resourceName := os.Getenv("AZURE_OPENAI_RESOURCE_NAME")
	if resourceName == "" {
		t.Skip("AZURE_OPENAI_RESOURCE_NAME environment variable is not set")
	}

	deploymentName := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
	if deploymentName == "" {
		t.Skip("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set")
	}

	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	if apiVersion == "" {
		apiVersion = "2023-05-15" // Default
	}

	t.Run("Azure OpenAI with GenericProvider", func(t *testing.T) {
		// Create endpoint URL
		endpoint := fmt.Sprintf(
			"https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s",
			resourceName, deploymentName, apiVersion,
		)

		// Register the provider if not already registered
		providerName := "test-azure-openai"
		azureConfig := providers.ProviderConfig{
			Name:              providerName,
			Type:              providers.TypeOpenAI,
			Endpoint:          endpoint,
			AuthHeader:        "api-key",
			AuthPrefix:        "",
			RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
			SupportsSchema:    true,
			SupportsStreaming: true,
		}
		providers.RegisterGenericProvider(providerName, azureConfig)

		// Create LLM
		llm, err := gollm.NewLLM(
			config.SetProvider(providerName),
			config.SetAPIKey(apiKey),
			config.SetModel(deploymentName),
			config.SetMaxTokens(50),
			config.SetTemperature(0.7),
		)
		require.NoError(t, err)

		// Test simple prompt
		ctx := context.Background()
		prompt := gollm.NewPrompt("What is 2+2? Answer with just the number.")

		response, err := llm.Generate(ctx, prompt)
		require.NoError(t, err)
		assert.Contains(t, response, "4")
	})

	t.Run("Azure OpenAI with ExtraHeaders", func(t *testing.T) {
		// Create endpoint URL
		endpoint := fmt.Sprintf(
			"https://%s.openai.azure.com/openai/deployments/%s/chat/completions?api-version=%s",
			resourceName, deploymentName, apiVersion,
		)

		// Create LLM with built-in azure-openai provider
		llm, err := gollm.NewLLM(
			config.SetProvider("azure-openai"),
			config.SetAPIKey(apiKey),
			config.SetModel(deploymentName),
			config.SetExtraHeaders(map[string]string{
				"azure_endpoint": endpoint,
			}),
			config.SetMaxTokens(50),
			config.SetTemperature(0.7),
		)
		require.NoError(t, err)

		// Test simple prompt
		ctx := context.Background()
		prompt := gollm.NewPrompt("What is the capital of France? Answer with just the city name.")

		response, err := llm.Generate(ctx, prompt)
		require.NoError(t, err)
		assert.Contains(t, response, "Paris")
	})
}
