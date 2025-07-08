//go:build ignore
// +build ignore

// Ignore this file for now, until gollm is updated to use the new provider registry

package gollm_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/guiperry/gollm_cerebras"
	"github.com/guiperry/gollm_cerebras/config"
	"github.com/guiperry/gollm_cerebras/providers"
	"github.com/guiperry/gollm_cerebras/utils"
)

// MockProvider implements the Provider interface for testing
type MockProvider struct {
	mock.Mock
}

func (m *MockProvider) Name() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockProvider) Endpoint() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockProvider) Headers() map[string]string {
	args := m.Called()
	return args.Get(0).(map[string]string)
}

func (m *MockProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	args := m.Called(prompt, options)
	return args.Get(0).([]byte), args.Error(1)
}

func (m *MockProvider) ParseResponse(body []byte) (string, error) {
	args := m.Called(body)
	return args.String(0), args.Error(1)
}

func (m *MockProvider) SupportsJSONSchema() bool {
	args := m.Called()
	return args.Bool(0)
}

func (m *MockProvider) SetOption(key string, value interface{}) {
	m.Called(key, value)
}

func (m *MockProvider) SetDefaultOptions(config *config.Config) {
	m.Called(config)
}

func (m *MockProvider) SetLogger(logger utils.Logger) {
	m.Called(logger)
}

type mockProviderFactory struct {
	mockProvider *MockProvider
}

func (f *mockProviderFactory) Create(apiKey, model string, extraHeaders map[string]string) (providers.Provider, error) {
	return f.mockProvider, nil
}

func TestStructuredOutput(t *testing.T) {
	ctx := context.Background()

	// Create and configure mock provider
	mockProvider := new(MockProvider)
	mockProvider.On("Name").Return("mock")
	mockProvider.On("SupportsJSONSchema").Return(true)
	mockProvider.On("SetOption", mock.Anything, mock.Anything).Return()
	mockProvider.On("SetDefaultOptions", mock.Anything).Return()
	mockProvider.On("SetLogger", mock.Anything).Return()
	mockProvider.On("Headers").Return(map[string]string{"Authorization": "Bearer test-key"})
	mockProvider.On("Endpoint").Return("https://api.mock.com/v1/completions")

	// Mock JSON response
	expectedJSON := `[{"benefit": "Cardiovascular health", "description": "Improves heart function"}]`
	mockProvider.On("PrepareRequest", mock.Anything, mock.Anything).
		Return([]byte(`{"messages": [{"role": "user", "content": "test"}]}`), nil)
	mockProvider.On("ParseResponse", mock.Anything).Return(expectedJSON, nil)

	// Register the mock provider
	registry := providers.NewProviderRegistry()
	registry.Register("mock", &mockProviderFactory{mockProvider: mockProvider})

	llm, err := gollm.NewLLM(
		gollm.SetProvider("mock"),
		gollm.SetModel("mock-model"),
		gollm.SetAPIKey("test-key"),
		gollm.WithProviderRegistry(registry),
	)
	assert.NoError(t, err)

	prompt := gollm.NewPrompt("List the top 3 benefits of exercise",
		gollm.WithOutput("JSON array of benefits"),
		gollm.WithSystemPrompt("You are a JSON-only assistant.", gollm.CacheTypeEphemeral),
	)

	response, err := llm.Generate(ctx, prompt)
	assert.NoError(t, err)
	assert.Equal(t, expectedJSON, response)

	// Verify that all expected mock calls were made
	mockProvider.AssertExpectations(t)
}

func TestJSONSchemaValidation(t *testing.T) {
	ctx := context.Background()

	// Create and configure mock provider
	mockProvider := new(MockProvider)
	mockProvider.On("Name").Return("mock")
	mockProvider.On("SupportsJSONSchema").Return(true)
	mockProvider.On("SetOption", mock.Anything, mock.Anything).Return()
	mockProvider.On("SetDefaultOptions", mock.Anything).Return()
	mockProvider.On("SetLogger", mock.Anything).Return()

	// Mock JSON response with schema
	expectedJSON := `{"name": "John Doe", "age": 25, "interests": ["coding", "reading"]}`
	mockProvider.On("PrepareRequest", mock.Anything, mock.Anything).
		Return([]byte(`{"messages": [{"role": "user", "content": "test"}]}`), nil)
	mockProvider.On("ParseResponse", mock.Anything).Return(expectedJSON, nil)

	llm, err := gollm.NewLLM(
		gollm.SetProvider("mock"),
		gollm.SetModel("mock-model"),
		gollm.SetAPIKey("test-key"),
	)
	assert.NoError(t, err)

	schema := `{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 18},
            "interests": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age", "interests"]
    }`

	prompt := gollm.NewPrompt("Generate a user profile",
		gollm.WithOutput(schema),
		gollm.WithSystemPrompt("You are a JSON-only assistant.", gollm.CacheTypeEphemeral),
	)

	response, err := llm.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
	assert.NoError(t, err)
	assert.Equal(t, expectedJSON, response)

	// Verify that all expected mock calls were made
	mockProvider.AssertExpectations(t)
}
