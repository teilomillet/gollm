package gollm

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// LLM is the interface that wraps the basic LLM operations
type LLM interface {
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error)
	SetOption(key string, value interface{})
	GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error)
	GetProvider() string
	GetModel() string
	UpdateDebugLevel(level LogLevel)
	Debug(msg string, keysAndValues ...interface{})
	GetDebugLevel() LogLevel
	SetOllamaEndpoint(endpoint string) error
	SetSystemPrompt(prompt string, cacheType CacheType)
}

// llmImpl is the concrete implementation of the LLM interface
type llmImpl struct {
	llm.LLM
	provider providers.Provider
	logger   utils.Logger
	model    string
	config   *Config
}

// SetSystemPrompt sets the system prompt for the LLM
func (l *llmImpl) SetSystemPrompt(prompt string, cacheType CacheType) {
	newPrompt := NewPrompt(prompt, WithSystemPrompt(prompt, cacheType))
	l.SetOption("system_prompt", newPrompt)
}

// GetProvider returns the provider of the LLM
func (l *llmImpl) GetProvider() string {
	return l.provider.Name()
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
	if p, ok := l.provider.(interface{ SetEndpoint(string) }); ok {
		p.SetEndpoint(endpoint)
		return nil
	}
	return fmt.Errorf("current provider does not support setting custom endpoint")
}

// GetPromptJSONSchema generates and returns the JSON schema for the Prompt
func (l *llmImpl) GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error) {
	p := &Prompt{}
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

// Generate produces a response given a context, prompt, and optional generate options
func (l *llmImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	l.logger.Debug("Starting Generate method", "prompt_length", len(prompt.String()), "context", ctx)

	if l == nil || l.LLM == nil {
		return "", fmt.Errorf("llmImpl or internal LLM is nil")
	}

	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	if config.UseJSONSchema {
		if err := prompt.Validate(); err != nil {
			return "", fmt.Errorf("invalid prompt: %w", err)
		}
	}

	response, _, err := l.LLM.Generate(ctx, prompt.String())
	if err != nil {
		return "", fmt.Errorf("LLM.Generate error: %w", err)
	}
	// Return the raw response, let the caller decide how to handle it
	return response, nil
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

	provider, err := providers.NewProviderRegistry().Get(config.Provider, config.APIKey, config.Model, config.ExtraHeaders)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	llmInstance := &llmImpl{
		LLM:      baseLLM,
		provider: provider,
		logger:   logger,
		model:    config.Model,
		config:   config,
	}

	if config.MemoryOption != nil {
		llmWithMemory, err := llm.NewLLMWithMemory(baseLLM, config.MemoryOption.MaxTokens, config.Model, logger)
		if err != nil {
			logger.Error("Failed to create LLM with memory", "error", err)
			return nil, fmt.Errorf("failed to create LLM with memory: %w", err)
		}
		llmInstance.LLM = llmWithMemory
	}

	return llmInstance, nil
}
