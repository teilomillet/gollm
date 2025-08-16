package llm

import (
	"context"
	"errors"
	"testing"

	"github.com/weave-labs/gollm/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// MockProvider implements a simple mock provider for testing
type MockProvider struct {
	messages   []models.MemoryMessage
	flattened  string
	structured bool
	logger     logging.Logger
	options    map[string]any
}

func NewMockProvider() *MockProvider {
	return &MockProvider{
		logger:  logging.NewLogger(logging.LogLevelDebug),
		options: make(map[string]any),
	}
}

// Implement required Provider interface methods
func (p *MockProvider) Name() string               { return "mock" }
func (p *MockProvider) Endpoint() string           { return "mock://endpoint" }
func (p *MockProvider) Headers() map[string]string { return map[string]string{} }
func (p *MockProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
	p.flattened = prompt
	p.structured = false
	return []byte(`{"prompt":"` + prompt + `"}`), nil
}

func (p *MockProvider) PrepareRequestWithMessages(
	messages []models.MemoryMessage,
	options map[string]any,
) ([]byte, error) {
	p.messages = messages
	p.structured = true
	return []byte(`{"messages":"structured"}`), nil
}
func (p *MockProvider) PrepareRequestWithSchema(prompt string, options map[string]any, schema any) ([]byte, error) {
	return []byte(`{}`), nil
}
func (p *MockProvider) SetExtraHeaders(extraHeaders map[string]string)  {}
func (p *MockProvider) HandleFunctionCalls(body []byte) ([]byte, error) { return nil, nil }
func (p *MockProvider) SupportsJSONSchema() bool                        { return false }
func (p *MockProvider) SetDefaultOptions(cfg *config.Config)            {}
func (p *MockProvider) SetOption(key string, value any)                 {}
func (p *MockProvider) SetLogger(logger logging.Logger)                 { p.logger = logger }
func (p *MockProvider) SupportsStreaming() bool                         { return false }
func (p *MockProvider) PrepareStreamRequest(prompt string, options map[string]any) ([]byte, error) {
	return []byte(`{}`), nil
}
func (p *MockProvider) ParseStreamResponse(chunk []byte) (string, error) { return "", nil }
func (p *MockProvider) SupportsStructuredResponse() bool                 { return false }
func (p *MockProvider) ParseResponse(body []byte) (string, error) {
	return "mock response", nil
}

// MockProviderRegistry for testing
type MockProviderRegistry struct {
	provider *MockProvider
}

func (r *MockProviderRegistry) Get(name, apiKey, model string, extraHeaders map[string]string) (any, error) {
	return r.provider, nil
}

// MockLLM is a custom implementation for testing
type MockLLM struct {
	provider *MockProvider
	logger   logging.Logger
}

func NewMockLLM(provider *MockProvider, logger logging.Logger) *MockLLM {
	return &MockLLM{
		provider: provider,
		logger:   logger,
	}
}

// Implement required LLM interface methods for testing
func (l *MockLLM) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	genConfig := &GenerateConfig{}
	for _, opt := range opts {
		opt(genConfig)
	}

	// Get the structured messages that were set with SetOption
	structuredMsgs, ok := l.provider.options["structured_messages"]
	if ok {
		if messages, ok := structuredMsgs.([]models.MemoryMessage); ok {
			if _, err := l.provider.PrepareRequestWithMessages(messages, nil); err != nil {
				return "", err
			}
			return "mock response", nil
		}
	}

	// Use flattened message
	if _, err := l.provider.PrepareRequest(prompt.String(), nil); err != nil {
		return "", err
	}
	return "mock response", nil
}

func (l *MockLLM) GenerateWithSchema(
	ctx context.Context,
	prompt *Prompt,
	schema any,
	opts ...GenerateOption,
) (string, error) {
	return "mock schema response", nil
}
func (l *MockLLM) GenerateStream(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (TokenStream, error) {
	return nil, errors.New("streaming not supported")
}
func (l *MockLLM) SupportsStreaming() bool { return false }
func (l *MockLLM) SetOption(key string, value any) {
	if key == "structured_messages" {
		if messages, ok := value.([]models.MemoryMessage); ok {
			l.provider.options = map[string]any{
				"structured_messages": messages,
			}
		}
	}
}
func (l *MockLLM) SetLogLevel(level logging.LogLevel) {}
func (l *MockLLM) SetEndpoint(endpoint string)        {}
func (l *MockLLM) NewPrompt(input string) *Prompt     { return &Prompt{Input: input} }
func (l *MockLLM) GetLogger() logging.Logger          { return l.logger }
func (l *MockLLM) SupportsJSONSchema() bool           { return false }

// TestStructuredMessageStorage tests that structured messages are properly stored
func TestStructuredMessageStorage(t *testing.T) {
	// Create logger
	logger := logging.NewLogger(logging.LogLevelDebug)

	// Create memory manager
	memory, err := NewMemory(1000, "gpt-4", logger)
	require.NoError(t, err)

	// Add regular message
	memory.Add("user", "Hello, how are you?")

	// Add structured message with cache control
	structuredMsg := models.MemoryMessage{
		Role:         "user",
		Content:      "This has cache control",
		CacheControl: "ephemeral",
	}
	memory.AddStructured(structuredMsg)

	// Get all messages
	allMessages := memory.GetMessages()

	// Verify message count
	assert.Len(t, allMessages, 2)

	// Verify regular message
	assert.Equal(t, "user", allMessages[0].Role)
	assert.Equal(t, "Hello, how are you?", allMessages[0].Content)
	assert.Empty(t, allMessages[0].CacheControl)

	// Verify structured message with cache control
	assert.Equal(t, "user", allMessages[1].Role)
	assert.Equal(t, "This has cache control", allMessages[1].Content)
	assert.Equal(t, "ephemeral", allMessages[1].CacheControl)

	// Test GetPrompt returns flattened text
	flattened := memory.GetPrompt()
	assert.Contains(t, flattened, "user: Hello, how are you?")
	assert.Contains(t, flattened, "user: This has cache control")
}

// TestAddStructuredMessage tests the helper method on LLMWithMemory
func TestAddStructuredMessage(t *testing.T) {
	// Create logger
	logger := logging.NewLogger(logging.LogLevelDebug)

	// Create memory manager
	memory, err := NewMemory(1000, "gpt-4", logger)
	require.NoError(t, err)

	// Create LLMWithMemory
	memoryLLMory := &LLMWithMemory{
		memory:                memory,
		useStructuredMessages: true,
	}

	// Add message with cache control
	memoryLLMory.AddStructuredMessage("user", "Hello with cache", "ephemeral")

	// Get messages
	messages := memoryLLMory.GetMemory()

	// Verify message was added with correct cache control
	assert.Len(t, messages, 1)
	assert.Equal(t, "user", messages[0].Role)
	assert.Equal(t, "Hello with cache", messages[0].Content)
	assert.Equal(t, "ephemeral", messages[0].CacheControl)
}

// TestCachingBenefit only runs if ANTHROPIC_API_KEY is set and performs a real-world test
// of caching performance with structured messages
// Note: This test is commented out to avoid import cycles with the providers package.
// To run this test, uncomment it and ensure the providers package is available.
/*
func TestCachingBenefit(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping caching test: ANTHROPIC_API_KEY not set")
	}

	// Create config for Anthropic
	cfg := &config.Config{
		Provider:      "anthropic",
		Model:         "claude-3-haiku-20240307",
		MaxTokens:     1024,
		Temperature:   0.7,
		EnableCaching: true,
		APIKeys: map[string]string{
			"anthropic": apiKey,
		},
		ExtraHeaders: map[string]string{
			"anthropic-beta": "prompt-caching-2024-07-31",
		},
	}

	// Create logger
	logger := logging.NewLogger(logging.LogLevelDebug)

	// Create mock registry
	registry := &MockProviderRegistry{
		provider: NewMockProvider(),
	}

	// Create LLM instance
	baseLLM, err := NewLLM(cfg, logger, registry)
	require.NoError(t, err)

	// Create LLM with memory
	memoryLLM, err := NewLLMWithMemory(baseLLM, 4000, cfg.Model)
	require.NoError(t, err)

	// Set up structured messages
	memoryLLM.SetUseStructuredMessages(true)
	memoryLLM.ClearMemory()

	// Add message with cache control
	memoryLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	// Create test prompt
	prompt := memoryLLM.NewPrompt("Give me a 30-word description of machine learning.")
	prompt.SystemPrompt = "You are a helpful assistant that provides concise responses."

	// First run (no cache)
	ctx := context.Background()
	start := time.Now()
	_, err = memoryLLM.Generate(ctx, prompt)
	require.NoError(t, err)
	firstRunDuration := time.Since(start)

	// Second run (should use cache)
	memoryLLM.ClearMemory()
	memoryLLM.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	start = time.Now()
	_, err = memoryLLM.Generate(ctx, prompt)
	require.NoError(t, err)
	secondRunDuration := time.Since(start)

	// Report times - don't assert since network conditions vary
	t.Logf("First run (no cache): %v", firstRunDuration)
	t.Logf("Second run (with cache): %v", secondRunDuration)
	t.Logf("Speedup: %.2fx", float64(firstRunDuration)/float64(secondRunDuration))
}
*/
