// Package llm provides a unified interface for interacting with various Language Learning Model providers.
package llm

import (
	"context"
	"fmt"
	"sync"

	"github.com/mauza/gollm/utils"
	"github.com/pkoukk/tiktoken-go"
)

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
type MemoryMessage struct {
	Role    string // Role of the message sender (e.g., "user", "assistant")
	Content string // The actual message content
	Tokens  int    // Number of tokens in the message
}

// Memory manages conversation history with token-based truncation.
// It provides thread-safe operations for adding, retrieving, and managing messages
// while ensuring the total token count stays within specified limits.
type Memory struct {
	messages    []MemoryMessage    // Ordered list of conversation messages
	mutex       sync.Mutex         // Ensures thread-safe operations
	totalTokens int                // Current total token count
	maxTokens   int                // Maximum allowed tokens
	encoding    *tiktoken.Tiktoken // Token encoder for the model
	logger      utils.Logger       // Logger for debugging and monitoring
}

// NewMemory creates a new Memory instance with the specified token limit and model.
// It initializes the token encoder based on the model and sets up logging.
//
// Parameters:
//   - maxTokens: Maximum number of tokens to keep in memory
//   - model: Name of the LLM model for token encoding
//   - logger: Logger for debugging and monitoring
//
// Returns:
//   - Initialized Memory instance
//   - ErrorTypeProvider if token encoding initialization fails
func NewMemory(maxTokens int, model string, logger utils.Logger) (*Memory, error) {
	encoding, err := tiktoken.EncodingForModel(model)
	if err != nil {
		logger.Warn("Failed to get encoding for model, defaulting to gpt-4o", "model", model, "error", err)
		encoding, err = tiktoken.EncodingForModel("gpt-4o")
		if err != nil {
			return nil, fmt.Errorf("failed to get default encoding: %v", err)
		}
	}

	return &Memory{
		messages:  []MemoryMessage{},
		maxTokens: maxTokens,
		encoding:  encoding,
		logger:    logger,
	}, nil
}

// Add appends a new message to the conversation history.
// It automatically truncates older messages if the token limit is exceeded.
// This operation is thread-safe.
//
// Parameters:
//   - role: Role of the message sender
//   - content: Content of the message
func (m *Memory) Add(role, content string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	tokens := m.encoding.Encode(content, nil, nil)
	message := MemoryMessage{Role: role, Content: content, Tokens: len(tokens)}
	m.messages = append(m.messages, message)
	m.totalTokens += len(tokens)

	m.truncate()
	m.logger.Debug("Added message to memory", "role", role, "tokens", len(tokens), "total_tokens", m.totalTokens)
}

// truncate removes oldest messages until the total token count is within limits.
// This is called automatically by Add when necessary.
func (m *Memory) truncate() {
	for m.totalTokens > m.maxTokens && len(m.messages) > 1 {
		removed := m.messages[0]
		m.messages = m.messages[1:]
		m.totalTokens -= removed.Tokens
		m.logger.Debug("Removed message from memory", "role", removed.Role, "tokens", removed.Tokens, "total_tokens", m.totalTokens)
	}
}

// GetPrompt returns the entire conversation history as a formatted string.
// Each message is formatted as "role: content\n".
// This operation is thread-safe.
//
// Returns:
//   - Formatted conversation history
func (m *Memory) GetPrompt() string {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	var prompt string
	for _, msg := range m.messages {
		prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}
	return prompt
}

// Clear removes all messages from memory and resets the token count.
// This operation is thread-safe.
func (m *Memory) Clear() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.messages = []MemoryMessage{}
	m.totalTokens = 0
	m.logger.Debug("Cleared memory")
}

// GetMessages returns a copy of all messages in memory.
// This operation is thread-safe.
//
// Returns:
//   - Slice of MemoryMessage containing the conversation history
func (m *Memory) GetMessages() []MemoryMessage {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return append([]MemoryMessage(nil), m.messages...)
}

// LLMWithMemory wraps an LLM instance with conversation memory capabilities.
// It maintains conversation history and provides context for each generation.
type LLMWithMemory struct {
	LLM            // Underlying LLM instance
	memory *Memory // Conversation memory manager
}

// NewLLMWithMemory creates a new LLM instance with conversation memory.
//
// Parameters:
//   - baseLLM: Base LLM instance to wrap
//   - maxTokens: Maximum number of tokens to keep in memory
//   - model: Name of the LLM model for token encoding
//   - logger: Logger for debugging and monitoring
//
// Returns:
//   - LLM instance with memory capabilities
//   - ErrorTypeProvider if memory initialization fails
func NewLLMWithMemory(baseLLM LLM, maxTokens int, model string, logger utils.Logger) (*LLMWithMemory, error) {
	memory, err := NewMemory(maxTokens, model, logger)
	if err != nil {
		return nil, err
	}
	return &LLMWithMemory{
		LLM:    baseLLM,
		memory: memory,
	}, nil
}

// Generate produces text based on the given prompt and conversation history.
// It automatically adds the prompt and response to memory.
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - prompt: Input prompt
//   - opts: Generation options
//
// Returns:
//   - Generated text response
//   - Error types as per the base LLM's Generate method
func (l *LLMWithMemory) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	l.memory.Add("user", prompt.Input)
	fullPrompt := l.memory.GetPrompt()

	// Create a new Prompt with the full memory context
	memoryPrompt := &Prompt{
		Input: fullPrompt,
		// Copy other fields from the original prompt if needed
	}

	response, err := l.LLM.Generate(ctx, memoryPrompt, opts...)
	if err != nil {
		return "", err
	}

	l.memory.Add("assistant", response)
	return response, nil
}

// ClearMemory removes all messages from the conversation history.
func (l *LLMWithMemory) ClearMemory() {
	l.memory.Clear()
}

// GetMemory returns a copy of all messages in the conversation history.
//
// Returns:
//   - Slice of MemoryMessage containing the conversation history
func (l *LLMWithMemory) GetMemory() []MemoryMessage {
	return l.memory.GetMessages()
}

// GenerateWithSchema generates text conforming to a schema, with conversation history.
// It automatically adds the prompt and response to memory.
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - prompt: Input prompt
//   - schema: JSON schema for response validation
//   - opts: Generation options
//
// Returns:
//   - Generated text response
//   - Error types as per the base LLM's GenerateWithSchema method
func (l *LLMWithMemory) GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error) {
	l.memory.Add("user", prompt.Input)
	fullPrompt := l.memory.GetPrompt()

	memoryPrompt := &Prompt{
		Input: fullPrompt,
		// Copy other fields from the original prompt if needed
	}

	response, err := l.LLM.GenerateWithSchema(ctx, memoryPrompt, schema, opts...)
	if err != nil {
		return "", err
	}

	l.memory.Add("assistant", response)
	return response, nil
}
