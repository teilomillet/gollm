// File: goal/goal.go

package goal

import (
	"context"
	"time"

	"github.com/teilomillet/goal/internal/llm"
)

// LLM represents the main interface for interacting with language models
type LLM interface {
	Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
	SetOption(key string, value interface{})
}

type llmImpl struct {
	llm.LLM
}

// Config represents the configuration for creating a new LLM instance
type Config struct {
	Providers   []string
	Provider    string
	Model       string
	Temperature float64
	MaxTokens   int
	APIKey      string
	Timeout     time.Duration
	LogLevel    string
}

// ConfigBuilder helps in building the Config with default values
type ConfigBuilder struct {
	config Config
}

// NewConfigBuilder creates a new ConfigBuilder with default values
func NewConfigBuilder() *ConfigBuilder {
	return &ConfigBuilder{
		config: Config{
			Provider:    "anthropic",
			Model:       "claude-3-opus-20240229",
			Temperature: 0.7,
			MaxTokens:   100,
			Timeout:     30 * time.Second,
			LogLevel:    "warn",
		},
	}
}

// SetProviders sets the list of providers
func (cb *ConfigBuilder) SetProviders(providers ...string) *ConfigBuilder {
	cb.config.Providers = providers
	return cb
}

// SetProvider sets the primary provider
func (cb *ConfigBuilder) SetProvider(provider string) *ConfigBuilder {
	cb.config.Provider = provider
	return cb
}

// SetModel sets the model
func (cb *ConfigBuilder) SetModel(model string) *ConfigBuilder {
	cb.config.Model = model
	return cb
}

// SetTemperature sets the temperature
func (cb *ConfigBuilder) SetTemperature(temperature float64) *ConfigBuilder {
	cb.config.Temperature = temperature
	return cb
}

// SetMaxTokens sets the max tokens
func (cb *ConfigBuilder) SetMaxTokens(maxTokens int) *ConfigBuilder {
	cb.config.MaxTokens = maxTokens
	return cb
}

// SetAPIKey sets the API key
func (cb *ConfigBuilder) SetAPIKey(apiKey string) *ConfigBuilder {
	cb.config.APIKey = apiKey
	return cb
}

// SetTimeout sets the timeout
func (cb *ConfigBuilder) SetTimeout(timeout time.Duration) *ConfigBuilder {
	cb.config.Timeout = timeout
	return cb
}

// SetLogLevel sets the log level
func (cb *ConfigBuilder) SetLogLevel(logLevel string) *ConfigBuilder {
	cb.config.LogLevel = logLevel
	return cb
}

// Build creates the final Config
func (cb *ConfigBuilder) Build() Config {
	return cb.config
}

// NewLLM creates a new LLM instance from a Config
func NewLLM(cfg Config) (LLM, error) {
	registry := llm.NewProviderRegistry(cfg.Providers...)

	logger := llm.NewDefaultLogger(cfg.LogLevel)

	internalConfig := &llm.Config{
		Provider:    cfg.Provider,
		Model:       cfg.Model,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		APIKey:      cfg.APIKey,
		Timeout:     cfg.Timeout,
	}

	l, err := llm.NewLLM(internalConfig, logger, registry)
	if err != nil {
		return nil, err
	}
	return &llmImpl{l}, nil
}
