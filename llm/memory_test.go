package llm

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/utils"
)

type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	args := m.Called(ctx, prompt, opts)
	return args.String(0), args.Error(1)
}

func (m *MockLLM) GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error) {
	args := m.Called(ctx, prompt, schema, opts)
	return args.String(0), args.Error(1)
}

func (m *MockLLM) SetOption(key string, value interface{}) {
	m.Called(key, value)
}

func (m *MockLLM) SetEndpoint(endpoint string) {
	m.Called(endpoint)
}

func (m *MockLLM) SetLogLevel(level utils.LogLevel) {
	m.Called(level)
}

func (m *MockLLM) NewPrompt(input string) *Prompt {
	args := m.Called(input)
	return args.Get(0).(*Prompt)
}

func (m *MockLLM) GetLogger() utils.Logger {
	args := m.Called()
	return args.Get(0).(utils.Logger)
}

func (m *MockLLM) SupportsJSONSchema() bool {
	args := m.Called()
	return args.Bool(0)
}

func TestMemory(t *testing.T) {
	logger := &utils.MockLogger{}
	logger.On("Debug", "Added message to memory", mock.Anything).Return()
	logger.On("Debug", "Removed message from memory", mock.Anything).Return()
	logger.On("Debug", "Cleared memory", mock.Anything).Return()

	t.Run("NewMemory", func(t *testing.T) {
		memory, err := NewMemory(100, "gpt-4o-mini", logger)
		require.NoError(t, err)
		assert.NotNil(t, memory)
		assert.Equal(t, 100, memory.maxTokens)
	})

	t.Run("Add and GetPrompt", func(t *testing.T) {
		memory, _ := NewMemory(100, "gpt-4o-mini", logger)
		memory.Add("user", "Hello")
		memory.Add("assistant", "Hi there!")

		prompt := memory.GetPrompt()
		assert.Contains(t, prompt, "user: Hello")
		assert.Contains(t, prompt, "assistant: Hi there!")
	})

	t.Run("Truncate", func(t *testing.T) {
		memory, _ := NewMemory(10, "gpt-4o-mini", logger)
		memory.Add("user", "This is a long message that exceeds the token limit")
		memory.Add("assistant", "Short reply")

		messages := memory.GetMessages()
		assert.Len(t, messages, 1)
		assert.Equal(t, "assistant", messages[0].Role)
	})

	t.Run("Clear", func(t *testing.T) {
		memory, _ := NewMemory(100, "gpt-4o-mini", logger)
		memory.Add("user", "Hello")
		memory.Clear()

		assert.Empty(t, memory.GetMessages())
	})

	logger.AssertExpectations(t)
}

func TestLLMWithMemory(t *testing.T) {
	logger := &utils.MockLogger{}
	logger.On("Debug", "Added message to memory", mock.Anything).Return()
	logger.On("Debug", "Cleared memory", mock.Anything).Return()

	mockLLM := new(MockLLM)
	llmWithMemory, err := NewLLMWithMemory(mockLLM, 100, "gpt-4o-mini", logger)
	require.NoError(t, err)

	ctx := context.Background()
	prompt := NewPrompt("Test prompt")

	t.Run("Generate", func(t *testing.T) {
		mockLLM.On("Generate", ctx, mock.Anything, mock.Anything).Return("Test response", nil)

		response, err := llmWithMemory.Generate(ctx, prompt)
		require.NoError(t, err)
		assert.Equal(t, "Test response", response)

		messages := llmWithMemory.GetMemory()
		assert.Len(t, messages, 2)
		assert.Equal(t, "user", messages[0].Role)
		assert.Equal(t, "assistant", messages[1].Role)

		mockLLM.AssertExpectations(t)
	})

	t.Run("GenerateWithSchema", func(t *testing.T) {
		schema := map[string]interface{}{"type": "string"}
		mockLLM.On("GenerateWithSchema", ctx, mock.Anything, schema, mock.Anything).Return("Structured response", nil)

		response, err := llmWithMemory.GenerateWithSchema(ctx, prompt, schema)
		require.NoError(t, err)
		assert.Equal(t, "Structured response", response)

		messages := llmWithMemory.GetMemory()
		assert.Len(t, messages, 4)

		mockLLM.AssertExpectations(t)
	})

	t.Run("ClearMemory", func(t *testing.T) {
		llmWithMemory.ClearMemory()
		assert.Empty(t, llmWithMemory.GetMemory())
	})
}
