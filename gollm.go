package gollm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// LLM is the interface that wraps the basic LLM operations
type LLM interface {
	// Generate produces a response given a context, prompt, and optional generate options
	Generate(ctx context.Context, prompt *llm.Prompt, opts ...GenerateOption) (string, error)

	// SetOption sets an option for the LLM
	SetOption(key string, value interface{})

	// GetPromptJSONSchema returns the JSON schema for the prompt
	GetPromptJSONSchema(opts ...llm.SchemaOption) ([]byte, error)

	// GetProvider returns the provider of the LLM
	GetProvider() string

	// GetModel returns the model of the LLM
	GetModel() string

	// UpdateDebugLevel updates the debug level of the LLM
	UpdateDebugLevel(level LogLevel)

	// Debug logs a debug message with optional key-value pairs
	Debug(msg string, keysAndValues ...interface{})

	// GetDebugLevel returns the current debug level of the LLM
	GetDebugLevel() LogLevel

	SetOllamaEndpoint(endpoint string) error

	SetSystemPrompt(prompt string, cacheType llm.CacheType)
}

// llmImpl is the concrete implementation of the LLM interface
type llmImpl struct {
	llm.LLM
	logger   utils.Logger
	provider string
	model    string
	config   *Config
}

// ToolCall is the struct for the function calling
type ToolCall = llm.ToolCall

// GenerateOption is a function type for configuring generate options
type GenerateOption func(*generateConfig)

// generateConfig holds configuration options for the Generate method
type generateConfig struct {
	useJSONSchema bool
}

// SetSystemPrompt sets the system prompt for the LLM
func (l *llmImpl) SetSystemPrompt(prompt string, cacheType llm.CacheType) {
	newPrompt := llm.NewPrompt(prompt, llm.WithSystemPrompt(prompt, cacheType))
	l.SetOption("system_prompt", newPrompt)
}

// WithCaching enables or disables caching in the Config
func WithCaching(enable bool) ConfigOption {
	return func(c *Config) {
		c.EnableCaching = enable
	}
}

// WithJSONSchemaValidation returns a GenerateOption that enables JSON schema validation
func WithJSONSchemaValidation() GenerateOption {
	return func(c *generateConfig) {
		c.useJSONSchema = true
	}
}

// GetProvider returns the provider of the LLM
func (l *llmImpl) GetProvider() string {
	return l.provider
}

// GetModel returns the model of the LLM
func (l *llmImpl) GetModel() string {
	return l.model
}

// Debug logs a debug message with optional key-value pairs
func (l *llmImpl) Debug(msg string, keysAndValues ...interface{}) {
	l.logger.Debug(msg, keysAndValues...)
}

// GetDebugLevel returns the current debug level of the LLM
func (l *llmImpl) GetDebugLevel() LogLevel {
	return l.config.DebugLevel
}

// SetOption sets an option for the LLM with the given key and value
func (l *llmImpl) SetOption(key string, value interface{}) {
	l.logger.Debug("Setting option", "key", key, "value", value)
	l.LLM.SetOption(key, value)
	l.logger.Debug("Option set successfully")
}

func (l *llmImpl) SetOllamaEndpoint(endpoint string) error {
	if p, ok := l.LLM.(interface{ SetEndpoint(string) }); ok {
		p.SetEndpoint(endpoint)
		return nil
	}
	return fmt.Errorf("current provider does not support setting custom endpoint")
}

func (l *llmImpl) ClearMemory() {
	if llmWithMemory, ok := l.LLM.(*llm.LLMWithMemory); ok {
		llmWithMemory.ClearMemory()
	}
}

func (l *llmImpl) GetMemory() []llm.MemoryMessage {
	if llmWithMemory, ok := l.LLM.(*llm.LLMWithMemory); ok {
		return llmWithMemory.GetMemory()
	}
	return nil
}

// GetPromptJSONSchema generates and returns the JSON schema for the Prompt
// It accepts optional SchemaOptions to customize the schema generation
func (l *llmImpl) GetPromptJSONSchema(opts ...llm.SchemaOption) ([]byte, error) {
	p := &llm.Prompt{}
	return p.GenerateJSONSchema(opts...)
}

// UpdateDebugLevel updates the debug level for both the gollm package and the internal llm package
func (l *llmImpl) UpdateDebugLevel(level LogLevel) {
	l.logger.Debug("Updating debug level",
		"current_level", l.config.DebugLevel,
		"new_level", level)

	l.config.DebugLevel = level
	l.logger.SetLevel(utils.LogLevel(level))

	if internalLLM, ok := l.LLM.(interface{ SetDebugLevel(utils.LogLevel) }); ok {
		internalLLM.SetDebugLevel(utils.LogLevel(level))
		l.logger.Debug("Updated internal LLM debug level")
	} else {
		l.logger.Warn("Internal LLM does not support SetDebugLevel")
	}

	l.logger.Debug("Debug level updated successfully")
}

// CleanResponse removes markdown code block syntax and trims the JSON response
func CleanResponse(response string) string {
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")

	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		response = response[start : end+1]
	}

	return strings.TrimSpace(response)
}

// Generate produces a response given a context, prompt, and optional generate options
func (l *llmImpl) Generate(ctx context.Context, prompt *llm.Prompt, opts ...GenerateOption) (string, error) {
	l.logger.Debug("Starting Generate method", "prompt_length", len(prompt.String()), "context", ctx)

	if l == nil || l.LLM == nil {
		return "", fmt.Errorf("llmImpl or internal LLM is nil")
	}

	config := &generateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	if config.useJSONSchema {
		if err := prompt.Validate(); err != nil {
			return "", fmt.Errorf("invalid prompt: %w", err)
		}
	}

	// Generate the response using the internal LLM
	response, _, err := l.LLM.Generate(ctx, prompt.String())
	if err != nil {
		return "", fmt.Errorf("LLM.Generate error: %w", err)
	}

	l.logger.Debug("Raw response from LLM", "response", response)

	// Parse the response
	var parsedResponse struct {
		Choices []struct {
			Message struct {
				Content   string     `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls,omitempty"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal([]byte(response), &parsedResponse); err != nil {
		return "", fmt.Errorf("failed to parse LLM response: %w", err)
	}

	if len(parsedResponse.Choices) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	message := parsedResponse.Choices[0].Message

	if len(message.ToolCalls) > 0 {
		toolCallsJSON, err := json.Marshal(message.ToolCalls)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tool calls: %w", err)
		}
		return string(toolCallsJSON), nil
	}

	return message.Content, nil
}

func (l *llmImpl) HandleFunctionCallResponse(response string) (string, error) {
	var llmResponse struct {
		Choices []struct {
			Message struct {
				Content   string     `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls,omitempty"`
			} `json:"message"`
		} `json:"choices"`
	}

	err := json.Unmarshal([]byte(response), &llmResponse)
	if err != nil {
		l.logger.Debug("Received non-JSON response", "response", response)
		return response, nil
	}

	if len(llmResponse.Choices) == 0 {
		return "", fmt.Errorf("empty response from LLM")
	}

	message := llmResponse.Choices[0].Message

	if len(message.ToolCalls) > 0 {
		l.logger.Debug("Received function call response", "tool_calls", message.ToolCalls)
		toolCallsJSON, err := json.Marshal(message.ToolCalls)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tool calls: %w", err)
		}
		return string(toolCallsJSON), nil
	}

	l.logger.Debug("Received regular response", "content", message.Content)
	return message.Content, nil
}

func (l *llmImpl) GenerateFunctionCallFollowUp(ctx context.Context, originalPrompt *llm.Prompt, functionCallResponse string, functionResult string) (string, error) {
	var parsedResponse struct {
		Choices []struct {
			Message struct {
				ToolCalls []ToolCall `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(functionCallResponse), &parsedResponse); err != nil {
		return "", fmt.Errorf("failed to parse function call response: %w", err)
	}

	if len(parsedResponse.Choices) == 0 || len(parsedResponse.Choices[0].Message.ToolCalls) == 0 {
		return "", fmt.Errorf("invalid function call response")
	}

	toolCall := parsedResponse.Choices[0].Message.ToolCalls[0]

	newMessages := append(originalPrompt.Messages,
		llm.PromptMessage{
			Role:      "assistant",
			Content:   "",
			ToolCalls: parsedResponse.Choices[0].Message.ToolCalls,
		},
		llm.PromptMessage{
			Role:       "tool",
			Content:    functionResult,
			Name:       toolCall.Function.Name,
			ToolCallID: toolCall.ID,
		},
	)

	followUpPrompt := llm.NewPrompt(
		"",
		llm.WithMessages(newMessages),
	)

	return l.Generate(ctx, followUpPrompt)
}

// NewLLM creates a new LLM instance, potentially with memory if the option is set
func NewLLM(opts ...ConfigOption) (LLM, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	for _, opt := range opts {
		opt(config)
	}

	logger := utils.NewLogger(utils.LogLevel(config.DebugLevel))

	if config.Provider == "anthropic" && config.EnableCaching {
		if config.ExtraHeaders == nil {
			config.ExtraHeaders = make(map[string]string)
		}
		config.ExtraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	internalConfig := config.toInternalConfig()

	baseLLM, err := llm.NewLLM(internalConfig, logger, providers.NewProviderRegistry())
	if err != nil {
		logger.Error("Failed to create internal LLM", "error", err)
		return nil, fmt.Errorf("failed to create internal LLM: %w", err)
	}

	var llmInstance LLM

	if config.MemoryOption != nil {
		llmWithMemory, err := llm.NewLLMWithMemory(baseLLM, config.MemoryOption.MaxTokens, config.Model, logger)
		if err != nil {
			logger.Error("Failed to create LLM with memory", "error", err)
			return nil, fmt.Errorf("failed to create LLM with memory: %w", err)
		}
		llmInstance = &llmImpl{
			LLM:      llmWithMemory,
			logger:   logger,
			provider: config.Provider,
			model:    config.Model,
			config:   config,
		}
	} else {
		llmInstance = &llmImpl{
			LLM:      baseLLM,
			logger:   logger,
			provider: config.Provider,
			model:    config.Model,
			config:   config,
		}
	}

	return llmInstance, nil
}
