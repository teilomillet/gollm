// File: goal.go

package goal

import (
	"context"
	"fmt"
	"time"

	"github.com/teilomillet/goal/internal/llm"
)

type LLM interface {
	Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
	SetOption(key string, value interface{})
}

type llmImpl struct {
	llm.LLM
}

// ConfigOption is a function type that modifies a Config
type ConfigOption func(*llm.Config)

// SetProvider sets the provider in the Config
func SetProvider(provider string) ConfigOption {
	return func(c *llm.Config) {
		c.Provider = provider
	}
}

// SetModel sets the model in the Config
func SetModel(model string) ConfigOption {
	return func(c *llm.Config) {
		c.Model = model
	}
}

// SetTemperature sets the temperature in the Config
func SetTemperature(temperature float64) ConfigOption {
	return func(c *llm.Config) {
		c.Temperature = temperature
	}
}

// SetMaxTokens sets the max tokens in the Config
func SetMaxTokens(maxTokens int) ConfigOption {
	return func(c *llm.Config) {
		c.MaxTokens = maxTokens
	}
}

// SetTimeout sets the timeout in the Config
func SetTimeout(timeout time.Duration) ConfigOption {
	return func(c *llm.Config) {
		c.Timeout = timeout
	}
}

// SetAPIKey sets the API key for the current provider in the Config
func SetAPIKey(key string) ConfigOption {
	return func(c *llm.Config) {
		c.APIKeys[c.Provider] = key
	}
}

// NewLLM creates a new LLM instance with the provided config options
func NewLLM(opts ...ConfigOption) (LLM, error) {
	config, err := llm.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// log.Printf("Loaded config: Provider=%s, Model=%s, MaxTokens=%d", config.Provider, config.Model, config.MaxTokens)
	// log.Printf("API Keys loaded for providers: %v", getKeysWithoutValues(config.APIKeys))

	for _, opt := range opts {
		opt(config)
	}

	// log.Printf("Config after applying options: Provider=%s, Model=%s, MaxTokens=%d", config.Provider, config.Model, config.MaxTokens)
	// log.Printf("Final API Keys loaded for providers: %v", getKeysWithoutValues(config.APIKeys))

	if len(config.APIKeys) == 0 {
		return nil, fmt.Errorf("at least one API key is required")
	}

	logger := llm.NewDefaultLogger("warn")
	registry := llm.NewProviderRegistry()

	l, err := llm.NewLLM(config, logger, registry)
	if err != nil {
		return nil, err
	}

	return &llmImpl{l}, nil
}

func getKeysWithoutValues(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
