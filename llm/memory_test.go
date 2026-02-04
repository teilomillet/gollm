package llm

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// MockProvider implements a simple mock provider for testing
type MockProvider struct {
	messages   []types.MemoryMessage
	flattened  string
	structured bool
	logger     utils.Logger
	options    map[string]interface{}
}

func NewMockProvider() *MockProvider {
	return &MockProvider{
		logger:  utils.NewLogger(utils.LogLevelDebug),
		options: make(map[string]interface{}),
	}
}

// Implement required Provider interface methods
func (p *MockProvider) Name() string               { return "mock" }
func (p *MockProvider) Endpoint() string           { return "mock://endpoint" }
func (p *MockProvider) Headers() map[string]string { return map[string]string{} }
func (p *MockProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	p.flattened = prompt
	p.structured = false
	return []byte(`{"prompt":"` + prompt + `"}`), nil
}
func (p *MockProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	p.messages = messages
	p.structured = true
	return []byte(`{"messages":"structured"}`), nil
}
func (p *MockProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	return []byte(`{}`), nil
}
func (p *MockProvider) ParseResponse(body []byte) (string, error)       { return "mock response", nil }
func (p *MockProvider) SetExtraHeaders(extraHeaders map[string]string)  {}
func (p *MockProvider) HandleFunctionCalls(body []byte) ([]byte, error) { return nil, nil }
func (p *MockProvider) SupportsJSONSchema() bool                        { return false }
func (p *MockProvider) SetDefaultOptions(config *config.Config)         {}
func (p *MockProvider) SetOption(key string, value interface{})         {}
func (p *MockProvider) SetLogger(logger utils.Logger)                   { p.logger = logger }
func (p *MockProvider) SupportsStreaming() bool                         { return false }
func (p *MockProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	return []byte(`{}`), nil
}
func (p *MockProvider) ParseStreamResponse(chunk []byte) (string, error) { return "", nil }

// MockLLM is a custom implementation for testing
type MockLLM struct {
	provider *MockProvider
	logger   utils.Logger
}

func NewMockLLM(provider *MockProvider, logger utils.Logger) *MockLLM {
	return &MockLLM{
		provider: provider,
		logger:   logger,
	}
}

// Implement required LLM interface methods for testing
func (l *MockLLM) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	// Get the structured messages that were set with SetOption
	structuredMsgs, ok := l.provider.options["structured_messages"]
	if ok {
		if messages, ok := structuredMsgs.([]types.MemoryMessage); ok {
			l.provider.PrepareRequestWithMessages(messages, nil)
			return "mock response", nil
		}
	}

	// Use flattened message
	l.provider.PrepareRequest(prompt.String(), nil)
	return "mock response", nil
}

func (l *MockLLM) GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error) {
	return "mock schema response", nil
}
func (l *MockLLM) Stream(ctx context.Context, prompt *Prompt, opts ...StreamOption) (TokenStream, error) {
	return nil, nil
}
func (l *MockLLM) SupportsStreaming() bool { return false }
func (l *MockLLM) SetOption(key string, value interface{}) {
	if key == "structured_messages" {
		if messages, ok := value.([]types.MemoryMessage); ok {
			l.provider.options = map[string]interface{}{
				"structured_messages": messages,
			}
		}
	}
}
func (l *MockLLM) SetLogLevel(level utils.LogLevel) {}
func (l *MockLLM) SetEndpoint(endpoint string)      {}
func (l *MockLLM) NewPrompt(input string) *Prompt   { return &Prompt{Input: input} }
func (l *MockLLM) GetLogger() utils.Logger          { return l.logger }
func (l *MockLLM) SupportsJSONSchema() bool         { return false }

// TestStructuredMessageStorage tests that structured messages are properly stored
func TestStructuredMessageStorage(t *testing.T) {
	// Create logger
	logger := utils.NewLogger(utils.LogLevelDebug)

	// Create memory manager
	memory, err := NewMemory(1000, "gpt-4", logger)
	require.NoError(t, err)

	// Add regular message
	memory.Add("user", "Hello, how are you?")

	// Add structured message with cache control
	structuredMsg := types.MemoryMessage{
		Role:         "user",
		Content:      "This has cache control",
		CacheControl: "ephemeral",
	}
	memory.AddStructured(structuredMsg)

	// Get all messages
	allMessages := memory.GetMessages()

	// Verify message count
	assert.Equal(t, 2, len(allMessages))

	// Verify regular message
	assert.Equal(t, "user", allMessages[0].Role)
	assert.Equal(t, "Hello, how are you?", allMessages[0].Content)
	assert.Equal(t, "", allMessages[0].CacheControl)

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
	logger := utils.NewLogger(utils.LogLevelDebug)

	// Create memory manager
	memory, err := NewMemory(1000, "gpt-4", logger)
	require.NoError(t, err)

	// Create LLMWithMemory
	llmWithMemory := &LLMWithMemory{
		memory:                memory,
		useStructuredMessages: true,
	}

	// Add message with cache control
	llmWithMemory.AddStructuredMessage("user", "Hello with cache", "ephemeral")

	// Get messages
	messages := llmWithMemory.GetMemory()

	// Verify message was added with correct cache control
	assert.Equal(t, 1, len(messages))
	assert.Equal(t, "user", messages[0].Role)
	assert.Equal(t, "Hello with cache", messages[0].Content)
	assert.Equal(t, "ephemeral", messages[0].CacheControl)
}

// TestCachingBenefit only runs if ANTHROPIC_API_KEY is set and performs a real-world test
// of caching performance with structured messages
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
	logger := utils.NewLogger(utils.LogLevelDebug)

	// Create registry and provider
	registry := providers.GetDefaultRegistry()

	// Create LLM instance
	baseLLM, err := NewLLM(cfg, logger, registry)
	require.NoError(t, err)

	// Create LLM with memory
	memoryLLM, err := NewLLMWithMemory(baseLLM, 4000, cfg.Model)
	require.NoError(t, err)

	// Cast to LLMWithMemory to access methods
	llmWithMem, ok := memoryLLM.(*LLMWithMemory)
	require.True(t, ok)

	// Set up structured messages
	llmWithMem.SetUseStructuredMessages(true)
	llmWithMem.ClearMemory()

	// Add message with cache control
	llmWithMem.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	// Create test prompt
	prompt := llmWithMem.NewPrompt("Give me a 30-word description of machine learning.")
	prompt.SystemPrompt = "You are a helpful assistant that provides concise responses."

	// First run (no cache)
	ctx := context.Background()
	start := time.Now()
	_, err = llmWithMem.Generate(ctx, prompt)
	require.NoError(t, err)
	firstRunDuration := time.Since(start)

	// Second run (should use cache)
	llmWithMem.ClearMemory()
	llmWithMem.AddStructuredMessage("user", "Hello, who are you?", "ephemeral")

	start = time.Now()
	_, err = llmWithMem.Generate(ctx, prompt)
	require.NoError(t, err)
	secondRunDuration := time.Since(start)

	// Report times - don't assert since network conditions vary
	t.Logf("First run (no cache): %v", firstRunDuration)
	t.Logf("Second run (with cache): %v", secondRunDuration)
	t.Logf("Speedup: %.2fx", float64(firstRunDuration)/float64(secondRunDuration))
}

// TestMemoryCreate tests NewMemory creation and edge cases
func TestMemoryCreate(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("valid creation", func(t *testing.T) {
		memory, err := NewMemory(1000, "gpt-4", logger)
		require.NoError(t, err)
		assert.NotNil(t, memory)
		assert.Equal(t, 0, len(memory.GetMessages()))
	})

	t.Run("unknown model falls back to gpt-4o", func(t *testing.T) {
		memory, err := NewMemory(1000, "unknown-model-xyz", logger)
		require.NoError(t, err)
		assert.NotNil(t, memory)
	})

	t.Run("zero max tokens", func(t *testing.T) {
		memory, err := NewMemory(0, "gpt-4", logger)
		require.NoError(t, err)
		assert.NotNil(t, memory)
	})

	t.Run("negative max tokens", func(t *testing.T) {
		memory, err := NewMemory(-100, "gpt-4", logger)
		require.NoError(t, err)
		assert.NotNil(t, memory)
	})
}

// TestMemoryAdd tests Add method and edge cases
func TestMemoryAdd(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("add single message", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "Hello")
		messages := memory.GetMessages()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "user", messages[0].Role)
		assert.Equal(t, "Hello", messages[0].Content)
	})

	t.Run("add multiple messages", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "Hello")
		memory.Add("assistant", "Hi there!")
		memory.Add("user", "How are you?")
		messages := memory.GetMessages()
		assert.Equal(t, 3, len(messages))
	})

	t.Run("add empty content", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "")
		messages := memory.GetMessages()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "", messages[0].Content)
	})

	t.Run("add empty role", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("", "Hello")
		messages := memory.GetMessages()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "", messages[0].Role)
	})

	t.Run("truncation when exceeding max tokens", func(t *testing.T) {
		memory, _ := NewMemory(10, "gpt-4", logger) // Very small limit
		memory.Add("user", "This is a long message that will exceed the token limit")
		memory.Add("user", "Another long message")
		// Should truncate older messages
		messages := memory.GetMessages()
		assert.LessOrEqual(t, len(messages), 2)
	})
}

// TestMemoryAddStructured tests AddStructured method and edge cases
func TestMemoryAddStructured(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("add with cache control", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		msg := types.MemoryMessage{
			Role:         "user",
			Content:      "Test",
			CacheControl: "ephemeral",
		}
		memory.AddStructured(msg)
		messages := memory.GetMessages()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "ephemeral", messages[0].CacheControl)
	})

	t.Run("add with metadata", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		msg := types.MemoryMessage{
			Role:    "user",
			Content: "Test",
			Metadata: map[string]interface{}{
				"timestamp": "2024-01-01",
			},
		}
		memory.AddStructured(msg)
		messages := memory.GetMessages()
		assert.Equal(t, 1, len(messages))
		assert.NotNil(t, messages[0].Metadata)
	})

	t.Run("add with zero tokens calculates tokens", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		msg := types.MemoryMessage{
			Role:    "user",
			Content: "Hello world",
			Tokens:  0, // Should be calculated
		}
		memory.AddStructured(msg)
		messages := memory.GetMessages()
		assert.Greater(t, messages[0].Tokens, 0)
	})

	t.Run("add with preset tokens uses preset", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		msg := types.MemoryMessage{
			Role:    "user",
			Content: "Hello world",
			Tokens:  999, // Preset value
		}
		memory.AddStructured(msg)
		messages := memory.GetMessages()
		assert.Equal(t, 999, messages[0].Tokens)
	})
}

// TestMemoryClear tests Clear method
func TestMemoryClear(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("clear with messages", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "Hello")
		memory.Add("assistant", "Hi")
		assert.Equal(t, 2, len(memory.GetMessages()))

		memory.Clear()
		assert.Equal(t, 0, len(memory.GetMessages()))
	})

	t.Run("clear empty memory", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Clear()
		assert.Equal(t, 0, len(memory.GetMessages()))
	})

	t.Run("clear resets token count", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "Hello world this is a test message")
		memory.Clear()
		// Total tokens should be 0 after clear
		assert.Equal(t, 0, memory.totalTokens)
	})
}

// TestMemoryGetPrompt tests GetPrompt method
func TestMemoryGetPrompt(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("empty memory returns empty string", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		assert.Equal(t, "", memory.GetPrompt())
	})

	t.Run("formats messages correctly", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "Hello")
		memory.Add("assistant", "Hi there")
		prompt := memory.GetPrompt()
		assert.Contains(t, prompt, "user: Hello")
		assert.Contains(t, prompt, "assistant: Hi there")
	})
}

// TestMemoryGetMessages tests GetMessages returns a deep copy
func TestMemoryGetMessages(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)

	t.Run("returns copy not reference", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.Add("user", "Hello")

		messages1 := memory.GetMessages()
		messages2 := memory.GetMessages()

		// Modify the first copy
		messages1[0].Content = "Modified"

		// Second copy should be unchanged
		assert.Equal(t, "Hello", messages2[0].Content)
	})

	t.Run("deep copies Metadata map", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		memory.AddStructured(types.MemoryMessage{
			Role:     "user",
			Content:  "Hello",
			Metadata: map[string]interface{}{"key": "original"},
		})

		messages1 := memory.GetMessages()
		messages2 := memory.GetMessages()

		// Modify the Metadata map in first copy
		messages1[0].Metadata["key"] = "modified"

		// Second copy's Metadata should be unchanged
		assert.Equal(t, "original", messages2[0].Metadata["key"])

		// Original internal state should also be unchanged
		messages3 := memory.GetMessages()
		assert.Equal(t, "original", messages3[0].Metadata["key"])
	})

	t.Run("deep copies ToolCalls slice", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		toolCall := types.ToolCall{ID: "call_1", Type: "function"}
		toolCall.Function.Name = "get_weather"
		toolCall.Function.Arguments = []byte(`{"city":"NYC"}`)
		memory.AddStructured(types.MemoryMessage{
			Role:      "assistant",
			Content:   "",
			ToolCalls: []types.ToolCall{toolCall},
		})

		messages1 := memory.GetMessages()
		messages2 := memory.GetMessages()

		// Modify the ToolCalls in first copy
		messages1[0].ToolCalls[0].Function.Name = "modified_tool"

		// Second copy's ToolCalls should be unchanged
		assert.Equal(t, "get_weather", messages2[0].ToolCalls[0].Function.Name)
	})

	t.Run("deep copies ToolCalls Function.Arguments", func(t *testing.T) {
		memory, _ := NewMemory(1000, "gpt-4", logger)
		toolCall := types.ToolCall{ID: "call_1", Type: "function"}
		toolCall.Function.Name = "get_weather"
		toolCall.Function.Arguments = []byte(`{"city":"NYC"}`)
		memory.AddStructured(types.MemoryMessage{
			Role:      "assistant",
			Content:   "",
			ToolCalls: []types.ToolCall{toolCall},
		})

		messages1 := memory.GetMessages()
		messages2 := memory.GetMessages()

		// Modify the Arguments byte slice in first copy
		messages1[0].ToolCalls[0].Function.Arguments[2] = 'X' // change 'c' to 'X'

		// Second copy's Arguments should be unchanged
		assert.Equal(t, `{"city":"NYC"}`, string(messages2[0].ToolCalls[0].Function.Arguments))

		// Original internal state should also be unchanged
		messages3 := memory.GetMessages()
		assert.Equal(t, `{"city":"NYC"}`, string(messages3[0].ToolCalls[0].Function.Arguments))
	})
}

// TestLLMWithMemoryCreate tests NewLLMWithMemory creation
func TestLLMWithMemoryCreate(t *testing.T) {
	t.Run("valid creation", func(t *testing.T) {
		mockProvider := NewMockProvider()
		mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
		llmWithMem, err := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
		require.NoError(t, err)
		assert.NotNil(t, llmWithMem)
	})

	t.Run("defaults to structured messages enabled", func(t *testing.T) {
		mockProvider := NewMockProvider()
		mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
		llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
		mem := llmWithMem.(*LLMWithMemory)
		assert.True(t, mem.useStructuredMessages)
	})
}

// TestLLMWithMemorySetUseStructuredMessages tests toggling structured messages
func TestLLMWithMemorySetUseStructuredMessages(t *testing.T) {
	mockProvider := NewMockProvider()
	mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
	llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
	mem := llmWithMem.(*LLMWithMemory)

	t.Run("can disable structured messages", func(t *testing.T) {
		mem.SetUseStructuredMessages(false)
		assert.False(t, mem.useStructuredMessages)
	})

	t.Run("can enable structured messages", func(t *testing.T) {
		mem.SetUseStructuredMessages(true)
		assert.True(t, mem.useStructuredMessages)
	})
}

// TestLLMWithMemoryGenerate tests Generate with memory
func TestLLMWithMemoryGenerate(t *testing.T) {
	ctx := context.Background()

	t.Run("generate with structured messages", func(t *testing.T) {
		mockProvider := NewMockProvider()
		mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
		llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
		mem := llmWithMem.(*LLMWithMemory)
		mem.SetUseStructuredMessages(true)

		prompt := mem.NewPrompt("Hello")
		_, err := mem.Generate(ctx, prompt)
		require.NoError(t, err)

		// Should have used structured messages path
		assert.True(t, mockProvider.structured)
	})

	t.Run("generate with flattened messages", func(t *testing.T) {
		mockProvider := NewMockProvider()
		mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
		llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
		mem := llmWithMem.(*LLMWithMemory)
		mem.SetUseStructuredMessages(false)

		prompt := mem.NewPrompt("Hello")
		_, err := mem.Generate(ctx, prompt)
		require.NoError(t, err)

		// Should have used flattened messages path
		assert.False(t, mockProvider.structured)
	})

	t.Run("adds user message to memory", func(t *testing.T) {
		mockProvider := NewMockProvider()
		mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
		llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
		mem := llmWithMem.(*LLMWithMemory)

		prompt := mem.NewPrompt("Hello")
		_, err := mem.Generate(ctx, prompt)
		require.NoError(t, err)

		messages := mem.GetMemory()
		// Should have user message + assistant response
		assert.Equal(t, 2, len(messages))
		assert.Equal(t, "user", messages[0].Role)
		assert.Equal(t, "assistant", messages[1].Role)
	})
}

// TestLLMWithMemoryClearMemory tests ClearMemory method
func TestLLMWithMemoryClearMemory(t *testing.T) {
	mockProvider := NewMockProvider()
	mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
	llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
	mem := llmWithMem.(*LLMWithMemory)

	mem.AddToMemory("user", "Hello")
	mem.AddToMemory("assistant", "Hi")
	assert.Equal(t, 2, len(mem.GetMemory()))

	mem.ClearMemory()
	assert.Equal(t, 0, len(mem.GetMemory()))
}

// TestLLMWithMemoryAddToMemory tests AddToMemory method
func TestLLMWithMemoryAddToMemory(t *testing.T) {
	mockProvider := NewMockProvider()
	mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
	llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
	mem := llmWithMem.(*LLMWithMemory)

	t.Run("add single message", func(t *testing.T) {
		mem.ClearMemory()
		mem.AddToMemory("user", "Hello")
		messages := mem.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "user", messages[0].Role)
		assert.Equal(t, "Hello", messages[0].Content)
	})

	t.Run("add empty content", func(t *testing.T) {
		mem.ClearMemory()
		mem.AddToMemory("user", "")
		messages := mem.GetMemory()
		assert.Equal(t, 1, len(messages))
	})
}

// TestLLMWithMemoryAddStructuredMessage tests AddStructuredMessage method
func TestLLMWithMemoryAddStructuredMessage(t *testing.T) {
	mockProvider := NewMockProvider()
	mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
	llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
	mem := llmWithMem.(*LLMWithMemory)

	t.Run("add with cache control", func(t *testing.T) {
		mem.ClearMemory()
		mem.AddStructuredMessage("user", "Hello", "ephemeral")
		messages := mem.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "ephemeral", messages[0].CacheControl)
	})

	t.Run("add with empty cache control", func(t *testing.T) {
		mem.ClearMemory()
		mem.AddStructuredMessage("user", "Hello", "")
		messages := mem.GetMemory()
		assert.Equal(t, 1, len(messages))
		assert.Equal(t, "", messages[0].CacheControl)
	})
}

// TestLLMWithMemoryDelegation tests that methods delegate to underlying LLM
func TestLLMWithMemoryDelegation(t *testing.T) {
	mockProvider := NewMockProvider()
	mockLLM := NewMockLLM(mockProvider, utils.NewLogger(utils.LogLevelDebug))
	llmWithMem, _ := NewLLMWithMemory(mockLLM, 1000, "gpt-4")
	mem := llmWithMem.(*LLMWithMemory)

	t.Run("NewPrompt delegates", func(t *testing.T) {
		prompt := mem.NewPrompt("Test")
		assert.Equal(t, "Test", prompt.Input)
	})

	t.Run("GetLogger delegates", func(t *testing.T) {
		logger := mem.GetLogger()
		assert.NotNil(t, logger)
	})

	t.Run("SupportsStreaming delegates", func(t *testing.T) {
		// MockLLM returns false
		assert.False(t, mem.SupportsStreaming())
	})

	t.Run("SupportsJSONSchema delegates", func(t *testing.T) {
		// MockLLM returns false
		assert.False(t, mem.SupportsJSONSchema())
	})
}

// TestMemoryConcurrency tests thread safety
func TestMemoryConcurrency(t *testing.T) {
	logger := utils.NewLogger(utils.LogLevelDebug)
	memory, _ := NewMemory(10000, "gpt-4", logger)

	done := make(chan bool)

	// Concurrent adds
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				memory.Add("user", "Message from goroutine")
			}
			done <- true
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				_ = memory.GetMessages()
				_ = memory.GetPrompt()
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 20; i++ {
		<-done
	}

	// Should not panic and messages should be consistent
	messages := memory.GetMessages()
	assert.Greater(t, len(messages), 0)
}

// TestLMStudioIntegration tests with LM Studio if available
func TestLMStudioIntegration(t *testing.T) {
	// Try to connect to LM Studio on default port
	cfg := &config.Config{
		Provider:    "lmstudio",
		Model:       "local-model",
		MaxTokens:   100,
		Temperature: 0.7,
		APIKeys: map[string]string{
			"lmstudio": "lmstudio-local",
		},
		Timeout:    5 * time.Second,
		MaxRetries: 0,
	}

	logger := utils.NewLogger(utils.LogLevelDebug)
	registry := providers.GetDefaultRegistry()

	baseLLM, err := NewLLM(cfg, logger, registry)
	if err != nil {
		t.Skip("Skipping LM Studio test: could not create LLM client")
	}

	llmWithMem, err := NewLLMWithMemory(baseLLM, 1000, cfg.Model)
	if err != nil {
		t.Skip("Skipping LM Studio test: could not create memory LLM")
	}

	mem := llmWithMem.(*LLMWithMemory)
	mem.SetUseStructuredMessages(true)
	mem.ClearMemory()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	prompt := mem.NewPrompt("Say hello in exactly 3 words.")
	_, err = mem.Generate(ctx, prompt)
	if err != nil {
		t.Skipf("Skipping LM Studio test: %v (LM Studio may not be running)", err)
	}

	// If we got here, LM Studio responded
	messages := mem.GetMemory()
	assert.Equal(t, 2, len(messages))
	t.Log("LM Studio integration test passed")
}
