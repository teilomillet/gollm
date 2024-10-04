package gollm

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/llm"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// LLM is the interface that wraps the basic LLM operations
type LLM interface {
	llm.LLM // Embed the base interface
	// Additional methods specific to gollm
	GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error)
	GetProvider() string
	GetModel() string
	UpdateLogLevel(level LogLevel)
	Debug(msg string, keysAndValues ...interface{})
	GetLogLevel() LogLevel
	SetOllamaEndpoint(endpoint string) error
	SetSystemPrompt(prompt string, cacheType CacheType)
}

// llmImpl is the concrete implementation of the LLM interface
type llmImpl struct {
	llm.LLM
	provider providers.Provider
	logger   utils.Logger
	model    string
	config   *config.Config
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

// GetLogLevel returns the current log level of the LLM
func (l *llmImpl) GetLogLevel() LogLevel {
	return LogLevel(l.config.LogLevel)
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

// UpdateLogLevel updates the log level for both the gollm package and the internal llm package
func (l *llmImpl) UpdateLogLevel(level LogLevel) {
	l.config.LogLevel = utils.LogLevel(level)
	l.logger.SetLevel(utils.LogLevel(level))
	if internalLLM, ok := l.LLM.(interface{ SetLogLevel(utils.LogLevel) }); ok {
		internalLLM.SetLogLevel(utils.LogLevel(level))
	}
}

// Implement the base Generate method (if not already provided by embedded llm.LLM)
func (l *llmImpl) Generate(ctx context.Context, prompt *llm.Prompt, opts ...llm.GenerateOption) (string, error) {
	l.logger.Debug("Starting Generate method", "prompt_length", len(prompt.String()), "context", ctx)

	config := &llm.GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	if config.UseJSONSchema {
		if err := prompt.Validate(); err != nil {
			return "", fmt.Errorf("invalid prompt: %w", err)
		}
	}

	// Call the base LLM's Generate method
	response, err := l.LLM.Generate(ctx, prompt, opts...)
	if err != nil {
		return "", fmt.Errorf("LLM.Generate error: %w", err)
	}

	return response, nil
}

// NewLLM creates a new LLM instance, potentially with memory if the option is set
func NewLLM(opts ...ConfigOption) (LLM, error) {
	cfg, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	for _, opt := range opts {
		opt(cfg)
	}

	logger := utils.NewLogger(cfg.LogLevel)

	if cfg.Provider == "anthropic" && cfg.EnableCaching {
		if cfg.ExtraHeaders == nil {
			cfg.ExtraHeaders = make(map[string]string)
		}
		cfg.ExtraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	baseLLM, err := llm.NewLLM(cfg, logger, providers.NewProviderRegistry())
	if err != nil {
		logger.Error("Failed to create internal LLM", "error", err)
		return nil, fmt.Errorf("failed to create internal LLM: %w", err)
	}

	provider, err := providers.NewProviderRegistry().Get(cfg.Provider, cfg.APIKeys[cfg.Provider], cfg.Model, cfg.ExtraHeaders)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	llmInstance := &llmImpl{
		LLM:      baseLLM,
		provider: provider,
		logger:   logger,
		model:    cfg.Model,
		config:   cfg,
	}

	if cfg.MemoryOption != nil {
		llmWithMemory, err := llm.NewLLMWithMemory(baseLLM, cfg.MemoryOption.MaxTokens, cfg.Model, logger)
		if err != nil {
			logger.Error("Failed to create LLM with memory", "error", err)
			return nil, fmt.Errorf("failed to create LLM with memory: %w", err)
		}
		llmInstance.LLM = llmWithMemory
	}

	return llmInstance, nil
}
