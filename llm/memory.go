// File: llm/memory.go

package llm

import (
	"context"
	"fmt"
	"sync"

	"github.com/pkoukk/tiktoken-go"
	"github.com/teilomillet/gollm/utils"
)

type MemoryMessage struct {
	Role    string
	Content string
	Tokens  int
}

type Memory struct {
	messages    []MemoryMessage
	mutex       sync.Mutex
	totalTokens int
	maxTokens   int
	encoding    *tiktoken.Tiktoken
	logger      utils.Logger
}

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

func (m *Memory) truncate() {
	for m.totalTokens > m.maxTokens && len(m.messages) > 1 {
		removed := m.messages[0]
		m.messages = m.messages[1:]
		m.totalTokens -= removed.Tokens
		m.logger.Debug("Removed message from memory", "role", removed.Role, "tokens", removed.Tokens, "total_tokens", m.totalTokens)
	}
}

func (m *Memory) GetPrompt() string {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	var prompt string
	for _, msg := range m.messages {
		prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}
	return prompt
}

func (m *Memory) Clear() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.messages = []MemoryMessage{}
	m.totalTokens = 0
	m.logger.Debug("Cleared memory")
}

func (m *Memory) GetMessages() []MemoryMessage {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	return append([]MemoryMessage(nil), m.messages...)
}

type LLMWithMemory struct {
	LLM
	memory *Memory
}

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

// Update the Generate method to match the new interface
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

func (l *LLMWithMemory) ClearMemory() {
	l.memory.Clear()
}

func (l *LLMWithMemory) GetMemory() []MemoryMessage {
	return l.memory.GetMessages()
}

// If GenerateWithSchema is used, you might want to add a similar method here
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
