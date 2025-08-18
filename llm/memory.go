// Package llm provides a unified interface for interacting with various Language Learning Model providers.
package llm

import (
	"context"
	"fmt"
	"sync"

	"github.com/pkoukk/tiktoken-go"

	"github.com/weave-labs/gollm/internal/logging"
	"github.com/weave-labs/gollm/providers"
)

// Memory manages conversation history with token-based truncation.
// It provides thread-safe operations for adding, retrieving, and managing messages
// while ensuring the total token count stays within specified limits.
type Memory struct {
	logger      logging.Logger
	encoding    *tiktoken.Tiktoken
	messages    []MemoryMessage
	totalTokens int
	maxTokens   int
	mutex       sync.Mutex
}

// NewMemory creates a new Memory instance with the specified token limit and model.
// It initializes the token encoder based on the model and sets up logging.
func NewMemory(maxTokens int, model string, logger logging.Logger) (*Memory, error) {
	encoding, err := tiktoken.EncodingForModel(model)
	if err != nil {
		logger.Warn("Failed to get encoding for model, defaulting to gpt-4o", "model", model, "error", err)
		encoding, err = tiktoken.EncodingForModel("gpt-4o")
		if err != nil {
			return nil, fmt.Errorf("failed to get default encoding: %w", err)
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
func (m *Memory) Add(role, content string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	tokens := m.encoding.Encode(content, nil, nil)
	message := MemoryMessage{Role: role, Content: content, Tokens: len(tokens)}
	m.messages = append(m.messages, message)
	m.totalTokens += len(tokens)

	// Truncate if needed
	m.truncateIfNeeded()

	m.logger.Debug("Added message", "role", role, "tokens", len(tokens), "total_tokens", m.totalTokens)
}

// AddStructured adds a pre-constructed message to the conversation history.
// This allows adding messages with custom metadata like cache control.
// It automatically truncates older messages if the token limit is exceeded.
// This operation is thread-safe.
func (m *Memory) AddStructured(message MemoryMessage) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// If tokens aren't already calculated, calculate them
	if message.Tokens == 0 && message.Content != "" {
		tokens := m.encoding.Encode(message.Content, nil, nil)
		message.Tokens = len(tokens)
	}

	m.messages = append(m.messages, message)
	m.totalTokens += message.Tokens

	// Truncate if needed
	m.truncateIfNeeded()

	m.logger.Debug("Added structured message",
		"role", message.Role,
		"tokens", message.Tokens,
		"cache_control", message.CacheControl,
		"total_tokens", m.totalTokens)
}

// truncateIfNeeded truncates messages if the total token count exceeds the maxTokens.
// This is called automatically by Add when necessary.
func (m *Memory) truncateIfNeeded() {
	for m.totalTokens > m.maxTokens && len(m.messages) > 1 {
		removed := m.messages[0]
		m.messages = m.messages[1:]
		m.totalTokens -= removed.Tokens
		m.logger.Debug(
			"Removed message from memory",
			"role",
			removed.Role,
			"tokens",
			removed.Tokens,
			"total_tokens",
			m.totalTokens,
		)
	}
}

// GetPrompt returns the full conversation history as a formatted string.
// Messages are formatted as "role: content" with newlines between them.
// This operation is thread-safe.
//
// Returns:
//   - Formatted conversation history string
func (m *Memory) GetPrompt() string {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	var prompt string
	for _, msg := range m.messages {
		prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}
	return prompt
}

// GetMessages returns the conversation history as structured messages.
// This returns the actual message objects with their roles, content, and metadata.
// This operation is thread-safe.
//
// Returns:
//   - Slice of MemoryMessage objects representing the conversation
func (m *Memory) GetMessages() []MemoryMessage {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Return a copy to prevent external modifications
	messages := make([]MemoryMessage, len(m.messages))
	copy(messages, m.messages)
	return messages
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

// LLMWithMemory wraps an LLM instance with conversation memory capabilities.
// It maintains a conversation history, automatically adding user prompts and
// assistant responses to create context for future interactions.
//
//nolint:revive // LLMWithMemory clearly describes its purpose as an LLM with memory capabilities
type LLMWithMemory struct {
	LLM                   LLM     // The base LLM instance to use for generation
	memory                *Memory // Conversation memory manager
	useStructuredMessages bool    // Whether to use structured messages with the provider
}

// GetLogger returns the logger from the wrapped LLM instance.
func (l *LLMWithMemory) GetLogger() logging.Logger {
	return l.LLM.GetLogger()
}

// SetLogLevel adjusts the logging verbosity.
func (l *LLMWithMemory) SetLogLevel(level logging.LogLevel) {
	l.LLM.SetLogLevel(level)
}

// SetEndpoint updates the API endpoint (primarily for local models).
func (l *LLMWithMemory) SetEndpoint(endpoint string) {
	l.LLM.SetEndpoint(endpoint)
}

// SetOption configures a provider-specific option.
func (l *LLMWithMemory) SetOption(key string, value any) {
	l.LLM.SetOption(key, value)
}

// GenerateStream initiates a streaming response from the LLM.
func (l *LLMWithMemory) GenerateStream(
	ctx context.Context,
	prompt *Prompt,
	opts ...GenerateOption,
) (TokenStream, error) {
	stream, err := l.LLM.GenerateStream(ctx, prompt, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to start stream: %w", err)
	}

	return stream, nil
}

// NewLLMWithMemory creates a new LLM instance with memory.
// It initializes a memory store with the specified token limit and configures
// the conversation context for the wrapped LLM.
func NewLLMWithMemory(llm LLM, maxTokens int, model string) (*LLMWithMemory, error) {
	logger := llm.GetLogger()
	memory, err := NewMemory(maxTokens, model, logger)
	if err != nil {
		return nil, err
	}

	return &LLMWithMemory{
		LLM:                   llm,
		memory:                memory,
		useStructuredMessages: true, // Default to using structured messages
	}, nil
}

// Generate produces text based on the given prompt and conversation history.
// It automatically adds the prompt and response to memory for future context.
func (l *LLMWithMemory) Generate(
	ctx context.Context,
	prompt *Prompt,
	opts ...GenerateOption,
) (*providers.Response, error) {
	l.memory.Add("user", prompt.Input)

	var response *providers.Response
	var err error

	if l.useStructuredMessages {
		// Get structured messages from memory
		messages := l.memory.GetMessages()

		// Make a copy of the original prompt with empty input
		// (since content will be in structured messages)
		emptyPrompt := &Prompt{
			SystemPrompt: prompt.SystemPrompt,
			Tools:        prompt.Tools,
			ToolChoice:   prompt.ToolChoice,
			Input:        "", // Empty as content is in messages
		}

		// Add structured messages to the options
		withMessages := func(_ *GenerateConfig) {
			// We're simply passing this function to keep the original options
			// The structured messages will be added in the WithOption call
		}

		// Set structured messages option
		l.LLM.SetOption("structured_messages", messages)

		// Generate with structured messages
		response, err = l.LLM.Generate(ctx, emptyPrompt, append(opts, withMessages)...)

		// Remove the structured messages option after use
		l.LLM.SetOption("structured_messages", nil)
	} else {
		// Fallback to traditional flattened prompt approach
		fullPrompt := l.memory.GetPrompt()

		// Create a new Prompt with the full memory context
		memoryPrompt := &Prompt{
			SystemPrompt: prompt.SystemPrompt,
			Tools:        prompt.Tools,
			ToolChoice:   prompt.ToolChoice,
			Input:        fullPrompt,
		}

		response, err = l.LLM.Generate(ctx, memoryPrompt, opts...)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}

	// Add assistant response to memory
	l.memory.Add("assistant", response.AsText())
	return response, nil
}

// SetUseStructuredMessages configures whether to use structured messages.
// When enabled, messages are passed to the provider as structured objects.
// When disabled, messages are flattened into a single text prompt.
//
// Parameters:
//   - use: Whether to use structured messages
func (l *LLMWithMemory) SetUseStructuredMessages(use bool) {
	l.useStructuredMessages = use
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

// AddToMemory adds a message to memory with the default role format.
// This is a convenience method that wraps memory.Add.
func (l *LLMWithMemory) AddToMemory(role, content string) {
	l.memory.Add(role, content)
}

// AddStructuredMessage adds a message to memory with cache control options.
// This allows clients to specify caching behavior for individual messages.
//
// Parameters:
//   - role: Role of the message sender (e.g. "user", "assistant")
//   - content: Message content
//   - cacheControl: Caching strategy ("ephemeral", "persistent", etc.)
func (l *LLMWithMemory) AddStructuredMessage(role, content, cacheControl string) {
	message := MemoryMessage{
		Role:         role,
		Content:      content,
		CacheControl: cacheControl,
	}
	l.memory.AddStructured(message)
}

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
type MemoryMessage struct {
	Metadata     map[string]any
	Role         string
	Content      string
	CacheControl string
	Tokens       int
}
