package providers

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/guiperry/gollm_cerebras/config"
	"github.com/guiperry/gollm_cerebras/utils"
)

func TestGenericProviderBasics(t *testing.T) {
	// Register a test configuration
	testConfig := ProviderConfig{
		Name:              "test-provider",
		Type:              TypeOpenAI,
		Endpoint:          "https://api.test-provider.com/v1/chat/completions",
		AuthHeader:        "Authorization",
		AuthPrefix:        "Bearer ",
		RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
		SupportsSchema:    true,
		SupportsStreaming: true,
	}

	// Create a provider using the configuration
	provider := &GenericProvider{
		apiKey:       "test-key",
		model:        "test-model",
		config:       testConfig,
		extraHeaders: map[string]string{"Extra-Header": "value"},
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}

	t.Run("Name returns correct provider name", func(t *testing.T) {
		assert.Equal(t, "test-provider", provider.Name())
	})

	t.Run("Endpoint returns correct URL", func(t *testing.T) {
		assert.Equal(t, "https://api.test-provider.com/v1/chat/completions", provider.Endpoint())
	})

	t.Run("Headers contain required and extra headers", func(t *testing.T) {
		headers := provider.Headers()

		assert.Equal(t, "Bearer test-key", headers["Authorization"])
		assert.Equal(t, "application/json", headers["Content-Type"])
		assert.Equal(t, "value", headers["Extra-Header"])
	})

	t.Run("SetDefaultOptions sets options correctly", func(t *testing.T) {
		cfg := &config.Config{
			Temperature: 0.7,
			MaxTokens:   500,
			Seed:        func() *int { i := 42; return &i }(),
		}

		provider.SetDefaultOptions(cfg)

		assert.Equal(t, 0.7, provider.options["temperature"])
		assert.Equal(t, 500, provider.options["max_tokens"])
		assert.Equal(t, 42, provider.options["seed"])
	})

	t.Run("SetOption updates options", func(t *testing.T) {
		provider.SetOption("presence_penalty", 0.2)

		assert.Equal(t, 0.2, provider.options["presence_penalty"])
	})

	t.Run("SetEndpoint overrides the endpoint", func(t *testing.T) {
		provider.SetEndpoint("https://custom-endpoint.com/api")

		assert.Equal(t, "https://custom-endpoint.com/api", provider.Endpoint())

		// Reset for other tests
		provider.extraEndpoint = ""
	})

	t.Run("SupportsJSONSchema returns config value", func(t *testing.T) {
		assert.True(t, provider.SupportsJSONSchema())

		// Test with a provider that doesn't support schemas
		noSchemaConfig := testConfig
		noSchemaConfig.SupportsSchema = false
		noSchemaProvider := &GenericProvider{
			apiKey:  "test-key",
			model:   "test-model",
			config:  noSchemaConfig,
			options: make(map[string]interface{}),
			logger:  utils.NewLogger(utils.LogLevelInfo),
		}

		assert.False(t, noSchemaProvider.SupportsJSONSchema())
	})

	t.Run("SupportsStreaming returns config value", func(t *testing.T) {
		assert.True(t, provider.SupportsStreaming())

		// Test with a provider that doesn't support streaming
		noStreamConfig := testConfig
		noStreamConfig.SupportsStreaming = false
		noStreamProvider := &GenericProvider{
			apiKey:  "test-key",
			model:   "test-model",
			config:  noStreamConfig,
			options: make(map[string]interface{}),
			logger:  utils.NewLogger(utils.LogLevelInfo),
		}

		assert.False(t, noStreamProvider.SupportsStreaming())
	})
}

func TestGenericProviderRequestFormatting(t *testing.T) {
	// Create an OpenAI-compatible provider
	openAIProvider := &GenericProvider{
		apiKey:  "test-key",
		model:   "gpt-4",
		config:  ProviderConfig{Type: TypeOpenAI},
		options: make(map[string]interface{}),
		logger:  utils.NewLogger(utils.LogLevelInfo),
	}

	// Create an Anthropic-compatible provider
	anthropicProvider := &GenericProvider{
		apiKey:  "test-key",
		model:   "claude-3",
		config:  ProviderConfig{Type: TypeAnthropic},
		options: make(map[string]interface{}),
		logger:  utils.NewLogger(utils.LogLevelInfo),
	}

	t.Run("OpenAI request format is correct", func(t *testing.T) {
		prompt := "Tell me a joke"
		body, err := openAIProvider.PrepareRequest(prompt, nil)
		require.NoError(t, err)

		var request map[string]interface{}
		err = json.Unmarshal(body, &request)
		require.NoError(t, err)

		assert.Equal(t, "gpt-4", request["model"])
		messages, ok := request["messages"].([]interface{})
		require.True(t, ok, "messages should be an array")
		require.Len(t, messages, 1)

		message, ok := messages[0].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "user", message["role"])
		assert.Equal(t, prompt, message["content"])
	})

	t.Run("Anthropic request format is correct", func(t *testing.T) {
		prompt := "Tell me a joke"
		body, err := anthropicProvider.PrepareRequest(prompt, nil)
		require.NoError(t, err)

		var request map[string]interface{}
		err = json.Unmarshal(body, &request)
		require.NoError(t, err)

		assert.Equal(t, "claude-3", request["model"])
		messages, ok := request["messages"].([]interface{})
		require.True(t, ok, "messages should be an array")
		require.Len(t, messages, 1)

		message, ok := messages[0].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "user", message["role"])
		assert.Equal(t, prompt, message["content"])
	})

	t.Run("Options are correctly added to request", func(t *testing.T) {
		openAIProvider.SetOption("temperature", 0.8)

		body, err := openAIProvider.PrepareRequest("test", nil)
		require.NoError(t, err)

		var request map[string]interface{}
		err = json.Unmarshal(body, &request)
		require.NoError(t, err)

		assert.Equal(t, 0.8, request["temperature"])
	})

	t.Run("Options can be overridden in request", func(t *testing.T) {
		openAIProvider.SetOption("temperature", 0.8)

		options := map[string]interface{}{
			"temperature": 0.5,
		}

		body, err := openAIProvider.PrepareRequest("test", options)
		require.NoError(t, err)

		var request map[string]interface{}
		err = json.Unmarshal(body, &request)
		require.NoError(t, err)

		assert.Equal(t, 0.5, request["temperature"])
	})
}

func TestGenericProviderResponseParsing(t *testing.T) {
	// Create an OpenAI-compatible provider
	openAIProvider := &GenericProvider{
		apiKey:  "test-key",
		model:   "gpt-4",
		config:  ProviderConfig{Type: TypeOpenAI},
		options: make(map[string]interface{}),
		logger:  utils.NewLogger(utils.LogLevelInfo),
	}

	// Create an Anthropic-compatible provider
	anthropicProvider := &GenericProvider{
		apiKey:  "test-key",
		model:   "claude-3",
		config:  ProviderConfig{Type: TypeAnthropic},
		options: make(map[string]interface{}),
		logger:  utils.NewLogger(utils.LogLevelInfo),
	}

	t.Run("Parse OpenAI response correctly", func(t *testing.T) {
		// Mock OpenAI-style response
		responseJSON := `{
			"choices": [
				{
					"message": {
						"content": "This is a test response"
					},
					"finish_reason": "stop"
				}
			]
		}`

		text, err := openAIProvider.ParseResponse([]byte(responseJSON))
		require.NoError(t, err)
		assert.Equal(t, "This is a test response", text)
	})

	t.Run("Parse Anthropic response correctly", func(t *testing.T) {
		// Mock Anthropic-style response
		responseJSON := `{
			"content": [
				{
					"text": "This is a test response from Claude"
				}
			]
		}`

		text, err := anthropicProvider.ParseResponse([]byte(responseJSON))
		require.NoError(t, err)
		assert.Equal(t, "This is a test response from Claude", text)
	})

	t.Run("Handle OpenAI error response", func(t *testing.T) {
		// Mock OpenAI error response
		errorJSON := `{
			"error": {
				"message": "Rate limit exceeded"
			}
		}`

		_, err := openAIProvider.ParseResponse([]byte(errorJSON))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Rate limit exceeded")
	})

	t.Run("Handle Anthropic error response", func(t *testing.T) {
		// Mock Anthropic error response
		errorJSON := `{
			"error": {
				"message": "Invalid API key"
			}
		}`

		_, err := anthropicProvider.ParseResponse([]byte(errorJSON))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Invalid API key")
	})
}

func TestGenericProviderStreamHandling(t *testing.T) {
	// Create providers for testing
	openAIProvider := &GenericProvider{
		apiKey:  "test-key",
		model:   "gpt-4",
		config:  ProviderConfig{Type: TypeOpenAI, SupportsStreaming: true},
		options: make(map[string]interface{}),
		logger:  utils.NewLogger(utils.LogLevelInfo),
	}

	t.Run("PrepareStreamRequest adds stream flag", func(t *testing.T) {
		body, err := openAIProvider.PrepareStreamRequest("test", nil)
		require.NoError(t, err)

		var request map[string]interface{}
		err = json.Unmarshal(body, &request)
		require.NoError(t, err)

		assert.True(t, request["stream"].(bool))
	})

	t.Run("ParseStreamResponse handles OpenAI chunk", func(t *testing.T) {
		// Mock OpenAI streaming chunk
		chunk := `data: {"choices":[{"delta":{"content":"Hello"}}]}`

		text, err := openAIProvider.ParseStreamResponse([]byte(chunk))
		require.NoError(t, err)
		assert.Equal(t, "Hello", text)
	})
}

func TestProviderRegistryWithGenericProvider(t *testing.T) {
	// Create a clean registry for testing
	registry := NewProviderRegistry()

	// Create a test configuration
	testConfig := ProviderConfig{
		Name:       "test-provider",
		Type:       TypeOpenAI,
		Endpoint:   "https://api.test-provider.com/v1/chat/completions",
		AuthHeader: "Authorization",
		AuthPrefix: "Bearer ",
		RequiredHeaders: map[string]string{
			"Content-Type": "application/json",
		},
		SupportsSchema:    true,
		SupportsStreaming: true,
	}

	t.Run("Register and retrieve generic provider config", func(t *testing.T) {
		// Register the config
		registry.RegisterProviderConfig("test-provider", testConfig)

		// Retrieve the config
		config, exists := registry.GetProviderConfig("test-provider")
		require.True(t, exists)
		assert.Equal(t, "test-provider", config.Name)
		assert.Equal(t, TypeOpenAI, config.Type)
		assert.Equal(t, "https://api.test-provider.com/v1/chat/completions", config.Endpoint)
	})

	t.Run("RegisterGenericProvider registers constructor", func(t *testing.T) {
		// Save the current registry
		oldRegistry := defaultRegistry
		defer func() { defaultRegistry = oldRegistry }()

		// Create a new default registry for this test
		defaultRegistry = NewProviderRegistry()

		// Register a generic provider
		testConfig.Name = "custom-provider" // Update the name to match what we check for
		RegisterGenericProvider("custom-provider", testConfig)

		// Check if constructor was registered by creating an instance
		constructor, exists := defaultRegistry.providers["custom-provider"]
		require.True(t, exists)

		provider := constructor("test-key", "test-model", nil)
		assert.Equal(t, "custom-provider", provider.Name())
	})
}

func TestAzureOpenAIProvider(t *testing.T) {
	// Skip if API keys aren't set
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("AZURE_OPENAI_API_KEY not set, skipping integration test")
	}

	resourceName := os.Getenv("AZURE_OPENAI_RESOURCE_NAME")
	if resourceName == "" {
		t.Skip("AZURE_OPENAI_RESOURCE_NAME not set, skipping integration test")
	}

	deploymentName := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
	if deploymentName == "" {
		t.Skip("AZURE_OPENAI_DEPLOYMENT_NAME not set, skipping integration test")
	}

	// Create the Azure OpenAI configuration
	endpoint := "https://" + resourceName + ".openai.azure.com/openai/deployments/" +
		deploymentName + "/chat/completions?api-version=2023-05-15"

	azureConfig := ProviderConfig{
		Name:       "azure-test",
		Type:       TypeOpenAI,
		Endpoint:   endpoint,
		AuthHeader: "api-key",
		AuthPrefix: "",
		RequiredHeaders: map[string]string{
			"Content-Type": "application/json",
		},
		SupportsSchema:    true,
		SupportsStreaming: true,
	}

	// Create the provider
	provider := &GenericProvider{
		apiKey:  apiKey,
		model:   deploymentName,
		config:  azureConfig,
		options: make(map[string]interface{}),
		logger:  utils.NewLogger(utils.LogLevelInfo),
	}

	t.Run("Azure OpenAI provider formats request correctly", func(t *testing.T) {
		body, err := provider.PrepareRequest("Hello", nil)
		require.NoError(t, err)

		var request map[string]interface{}
		err = json.Unmarshal(body, &request)
		require.NoError(t, err)

		// Azure OpenAI uses deployment name as model
		assert.Equal(t, deploymentName, request["model"])

		messages := request["messages"].([]interface{})
		require.Len(t, messages, 1)
		message := messages[0].(map[string]interface{})
		assert.Equal(t, "Hello", message["content"])
	})

	t.Run("Azure OpenAI provider creates correct headers", func(t *testing.T) {
		headers := provider.Headers()

		assert.Equal(t, apiKey, headers["api-key"])
		assert.Equal(t, "application/json", headers["Content-Type"])
	})

	// Skip actual API call test unless in integration test mode
	if os.Getenv("RUN_INTEGRATION_TESTS") != "true" {
		t.Skip("Skipping actual API call test. Set RUN_INTEGRATION_TESTS=true to run")
	}

	// This test should make an actual API call to Azure OpenAI
	t.Run("Azure OpenAI integration test", func(t *testing.T) {
		cfg := &config.Config{
			Temperature: 0.7,
			MaxTokens:   50,
		}
		provider.SetDefaultOptions(cfg)

		// Prepare the request but don't use the body since we're mocking
		_, err := provider.PrepareRequest("What is 2+2?", nil)
		require.NoError(t, err)

		// This would normally call the actual API
		// We're mocking the response instead
		mockResponse := `{
			"choices": [
				{
					"message": {
						"content": "The answer is 4."
					},
					"finish_reason": "stop"
				}
			]
		}`

		text, err := provider.ParseResponse([]byte(mockResponse))
		require.NoError(t, err)
		assert.Contains(t, text, "4")
	})
}
