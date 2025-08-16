// Package gollm provides a high-level interface for interacting with various Language Learning Models (LLMs).
// It supports multiple providers including OpenAI, Anthropic, Ollama, and others, with features
// like prompt optimization, caching, and structured output handling.
package gollm

import (
	"context"
	"errors"
	"fmt"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
	"github.com/weave-labs/gollm/llm"
	"github.com/weave-labs/gollm/providers"
)

// LLM is the interface that wraps the basic LLM operations.
// It extends the base llm.LLM interface with additional functionality specific to gollm,
// providing a comprehensive set of methods for LLM interaction, configuration, and management.
type LLM interface {
	llm.LLM // Embed the base interface
	// GetPromptJSONSchema returns the JSON schema for prompt validation in byte format.
	// The schema can be customized using SchemaOption parameters.
	GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error)
	// GetProvider returns the name of the current LLM provider (e.g., "openai", "anthropic").
	GetProvider() string
	// GetModel returns the name of the current model being used.
	GetModel() string
	// UpdateLogLevel changes the logging verbosity for both gollm and internal LLM operations.
	UpdateLogLevel(level LogLevel)
	// Debug logs a debug message with optional key-value pairs for detailed logging.
	Debug(msg string, keysAndValues ...any)
	// GetLogLevel returns the current logging verbosity level.
	GetLogLevel() LogLevel
	// SetOllamaEndpoint configures a custom endpoint for Ollama provider.
	// Returns an error if the current provider doesn't support endpoint configuration.
	SetOllamaEndpoint(endpoint string) error
	// SetSystemPrompt updates the system prompt with caching configuration.
	// The cacheType parameter determines how the prompt should be cached.
	SetSystemPrompt(prompt string, cacheType CacheType)
}

// LlmImpl is the concrete implementation of the LLM interface.
// It wraps the base LLM implementation and adds provider-specific functionality,
// logging capabilities, and configuration management.
type LlmImpl struct {
	llm.LLM
	provider providers.Provider
	logger   logging.Logger
	config   *config.Config
	model    string
}

// SetSystemPrompt sets the system prompt for the LLM.
func (l *LlmImpl) SetSystemPrompt(prompt string, cacheType CacheType) {
	newPrompt := NewPrompt(prompt, WithSystemPrompt(prompt, cacheType))
	l.SetOption("system_prompt", newPrompt)
}

// GetProvider returns the provider of the LLM.
func (l *LlmImpl) GetProvider() string {
	return l.provider.Name()
}

// GetModel returns the model of the LLM.
func (l *LlmImpl) GetModel() string {
	return l.model
}

// Debug logs a debug message with optional key-value pairs.
func (l *LlmImpl) Debug(msg string, keysAndValues ...any) {
	l.logger.Debug(msg, keysAndValues...)
}

// GetLogLevel returns the current log level of the LLM.
func (l *LlmImpl) GetLogLevel() LogLevel {
	return l.config.LogLevel
}

// SetOption sets an option for the LLM with the given key and value.
func (l *LlmImpl) SetOption(key string, value any) {
	l.logger.Debug("Setting option", "key", key, "value", value)
	l.LLM.SetOption(key, value)
	l.logger.Debug("Option set successfully")
}

// SetOllamaEndpoint sets the Ollama server endpoint URL
func (l *LlmImpl) SetOllamaEndpoint(endpoint string) error {
	if p, ok := l.provider.(interface{ SetEndpoint(endpoint string) }); ok {
		p.SetEndpoint(endpoint)
		return nil
	}
	return errors.New("current provider does not support setting custom endpoint")
}

// GetPromptJSONSchema generates and returns the JSON schema for the Prompt.
func (l *LlmImpl) GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error) {
	p := &Prompt{}
	schema, err := p.GenerateJSONSchema(opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to generate JSON schema: %w", err)
	}
	return schema, nil
}

// UpdateLogLevel updates the log level for both the gollm package and the internal llm package.
func (l *LlmImpl) UpdateLogLevel(level LogLevel) {
	l.config.LogLevel = level
	l.logger.SetLevel(level)
	if internalLLM, ok := l.LLM.(interface{ SetLogLevel(level logging.LogLevel) }); ok {
		internalLLM.SetLogLevel(level)
	}
}

// Generate Implement the base Generate method (if not already provided by embedded llm.LLM)
func (l *LlmImpl) Generate(
	ctx context.Context,
	prompt *llm.Prompt,
	opts ...llm.GenerateOption,
) (*providers.Response, error) {
	l.logger.Debug("Starting Generate method", "prompt_length", len(prompt.String()), "context", ctx)

	// Call the base LLM's Generate method
	response, err := l.LLM.Generate(ctx, prompt, opts...)
	if err != nil {
		return nil, fmt.Errorf("LLM.Generate error: %w", err)
	}

	return response, nil
}

// NewLLM creates a new LLM instance with the specified configuration options.
// It supports memory management, caching, and provider-specific optimizations.
// If memory options are provided, it creates an LLM instance with conversation memory.
//
// The function performs the following setup:
// 1. Loads and applies configuration from both default and provided options
// 2. Initializes logging system with appropriate verbosity
// 3. Sets up provider-specific optimizations (e.g., Anthropic caching headers)
// 4. Creates and configures the base LLM instance
// 5. Optionally enables conversation memory if specified in config
//
// Returns an error if:
// - Configuration loading fails
// - Provider initialization fails
// - Memory setup fails (if memory option is enabled)
// ensureOllamaAPIKey ensures Ollama has an API key set
func ensureOllamaAPIKey(cfg *Config) {
	if cfg.Provider == "ollama" {
		if cfg.APIKeys == nil {
			cfg.APIKeys = make(map[string]string)
		}
		if _, exists := cfg.APIKeys[cfg.Provider]; !exists || cfg.APIKeys[cfg.Provider] == "" {
			cfg.APIKeys[cfg.Provider] = "ollama-local"
		}
	}
}

// setupAnthropicCaching configures Anthropic caching headers
func setupAnthropicCaching(cfg *Config) {
	if cfg.Provider == "anthropic" && cfg.EnableCaching {
		if cfg.ExtraHeaders == nil {
			cfg.ExtraHeaders = make(map[string]string)
		}
		cfg.ExtraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}
}

// NewLLM creates a new LLM instance with the provided configuration options
func NewLLM(opts ...ConfigOption) (*LlmImpl, error) {
	cfg, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	for _, opt := range opts {
		opt(cfg)
	}

	ensureOllamaAPIKey(cfg)

	// Validate config
	if err := llm.Validate(cfg); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	logger := logging.NewLogger(cfg.LogLevel)
	setupAnthropicCaching(cfg)

	baseLLM, err := llm.NewLLM(cfg, logger, providers.NewProviderRegistry())
	if err != nil {
		logger.Error("Failed to create internal LLM", "error", err)
		return nil, fmt.Errorf("failed to create internal LLM: %w", err)
	}

	provider, err := providers.NewProviderRegistry().
		Get(cfg.Provider, cfg.APIKeys[cfg.Provider], cfg.Model, cfg.ExtraHeaders)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	llmInstance := &LlmImpl{
		LLM:      baseLLM,
		provider: provider,
		logger:   logger,
		model:    cfg.Model,
		config:   cfg,
	}

	if cfg.MemoryOption != nil {
		llmWithMemory, err := llm.NewLLMWithMemory(baseLLM, cfg.MemoryOption.MaxTokens, cfg.Model)
		if err != nil {
			logger.Error("Failed to create LLM with memory", "error", err)
			return nil, fmt.Errorf("failed to create LLM with memory: %w", err)
		}
		llmInstance.LLM = llmWithMemory
	}

	return llmInstance, nil
}
