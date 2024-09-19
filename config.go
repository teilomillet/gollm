// File: config.go

package gollm

import (
	"time"

	"github.com/teilomillet/gollm/internal/llm"
)

// LogLevel represents the level of logging
type LogLevel int

type MemoryOption struct {
	MaxTokens int
}

const (
	LogLevelOff LogLevel = iota
	LogLevelError
	LogLevelWarn
	LogLevelInfo
	LogLevelDebug
)

// Config holds the configuration for the LLM
type Config struct {
	Provider         string
	Model            string
	OllamaEndpoint   string
	Temperature      float64
	MaxTokens        int
	TopP             float64
	FrequencyPenalty float64
	PresencePenalty  float64
	Timeout          time.Duration
	MaxRetries       int
	RetryDelay       time.Duration
	APIKey           string
	DebugLevel       LogLevel
	MemoryOption     *MemoryOption
	Seed             *int
	MinP             *float64
	RepeatPenalty    *float64
	RepeatLastN      *int
	Mirostat         *int
	MirostatEta      *float64
	MirostatTau      *float64
	TfsZ             *float64
}

// toInternalConfig converts Config to internal llm.Config
func (c *Config) toInternalConfig() *llm.Config {
	internalLevel := llm.LogLevel(c.DebugLevel)
	return &llm.Config{
		Provider:         c.Provider,
		Model:            c.Model,
		OllamaEndpoint:   c.OllamaEndpoint,
		Temperature:      c.Temperature,
		MaxTokens:        c.MaxTokens,
		TopP:             c.TopP,
		FrequencyPenalty: c.FrequencyPenalty,
		PresencePenalty:  c.PresencePenalty,
		Timeout:          c.Timeout,
		MaxRetries:       c.MaxRetries,
		RetryDelay:       c.RetryDelay,
		APIKeys:          map[string]string{c.Provider: c.APIKey},
		LogLevel:         internalLevel,
		Seed:             c.Seed,
		MinP:             c.MinP,
		RepeatPenalty:    c.RepeatPenalty,
		RepeatLastN:      c.RepeatLastN,
		Mirostat:         c.Mirostat,
		MirostatEta:      c.MirostatEta,
		MirostatTau:      c.MirostatTau,
		TfsZ:             c.TfsZ,
	}
}

// Convert llm.LogLevel to gollm.LogLevel
func convertLogLevel(level llm.LogLevel) LogLevel {
	switch level {
	case llm.LogLevelDebug:
		return LogLevelDebug
	case llm.LogLevelInfo:
		return LogLevelInfo
	case llm.LogLevelWarn:
		return LogLevelWarn
	case llm.LogLevelError:
		return LogLevelError
	default:
		return LogLevelWarn // Default to Warn if unknown
	}
}

// ConfigOption is a function type for modifying Config
type ConfigOption func(*Config)

// LoadConfig loads the configuration from environment variables
func LoadConfig() (*Config, error) {
	internalConfig, err := llm.LoadConfig()
	if err != nil {
		return nil, err
	}

	config := &Config{
		Provider:    internalConfig.Provider,
		Model:       internalConfig.Model,
		Temperature: internalConfig.Temperature,
		MaxTokens:   internalConfig.MaxTokens,
		Timeout:     internalConfig.Timeout,
		MaxRetries:  internalConfig.MaxRetries,
		RetryDelay:  internalConfig.RetryDelay,
		DebugLevel:  convertLogLevel(internalConfig.LogLevel),
	}

	// Set the API key for the default provider
	if apiKey, exists := internalConfig.APIKeys[internalConfig.Provider]; exists {
		config.APIKey = apiKey
	}

	return config, nil
}

// SetProvider sets the provider in the Config
func SetProvider(provider string) ConfigOption {
	return func(c *Config) {
		c.Provider = provider
	}
}

// SetModel sets the model in the Config
func SetModel(model string) ConfigOption {
	return func(c *Config) {
		c.Model = model
	}
}

// SetOllamaEndpoint sets the Ollama endpoint in the Config
func SetOllamaEndpoint(endpoint string) ConfigOption {
	return func(c *Config) {
		c.OllamaEndpoint = endpoint
	}
}

// SetTemperature sets the temperature in the Config
func SetTemperature(temperature float64) ConfigOption {
	return func(c *Config) {
		c.Temperature = temperature
	}
}

// SetMaxTokens sets the max tokens in the Config
func SetMaxTokens(maxTokens int) ConfigOption {
	return func(c *Config) {
		if maxTokens < 1 {
			maxTokens = 1
		}
		c.MaxTokens = maxTokens
	}
}

// SetTimeout sets the timeout in the Config
func SetTimeout(timeout time.Duration) ConfigOption {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

// SetAPIKey sets the API key for the current provider in the Config
func SetAPIKey(apiKey string) ConfigOption {
	return func(c *Config) {
		c.APIKey = apiKey
	}
}

// SetMaxRetries sets the maximum number of retries in the Config
func SetMaxRetries(maxRetries int) ConfigOption {
	return func(c *Config) {
		c.MaxRetries = maxRetries
	}
}

// SetRetryDelay sets the delay between retries in the Config
func SetRetryDelay(retryDelay time.Duration) ConfigOption {
	return func(c *Config) {
		c.RetryDelay = retryDelay
	}
}

// SetDebugLevel sets the debug level in the Config
func SetDebugLevel(level LogLevel) ConfigOption {
	return func(c *Config) {
		c.DebugLevel = level
	}
}

// SetMemory sets the memory option for the LLM
func SetMemory(maxTokens int) ConfigOption {
	return func(c *Config) {
		c.MemoryOption = &MemoryOption{
			MaxTokens: maxTokens,
		}
	}
}

// SetTopP sets the top-p value. A higher value (e.g., 0.95) will lead to more diverse text,
// while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
func SetTopP(topP float64) ConfigOption {
	return func(c *Config) {
		c.TopP = topP
	}
}

// SetFrequencyPenalty sets the frequency penalty. It penalizes new tokens based on their frequency in the text so far.
// Higher values decrease the model's likelihood to repeat the same line verbatim.
func SetFrequencyPenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.FrequencyPenalty = penalty
	}
}

// SetPresencePenalty sets the presence penalty. It penalizes new tokens based on whether they appear in the text so far.
// Higher values increase the model's likelihood to talk about new topics.
func SetPresencePenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.PresencePenalty = penalty
	}
}

// SetSeed sets the random number seed for generation. Setting this to a specific number
// will make the model generate the same text for the same prompt. (Default: 0)
func SetSeed(seed int) ConfigOption {
	return func(c *Config) {
		c.Seed = &seed
	}
}

// SetMinP sets the minimum probability for a token to be considered, relative to the probability
// of the most likely token. It aims to ensure a balance of quality and variety. (Default: 0.05)
func SetMinP(minP float64) ConfigOption {
	return func(c *Config) {
		c.MinP = &minP
	}
}

// SetRepeatPenalty sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
// will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
func SetRepeatPenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.RepeatPenalty = &penalty
	}
}

// SetRepeatLastN sets how far back the model should look to prevent repetition.
// (Default: 64, 0 = disabled, -1 = num_ctx)
func SetRepeatLastN(n int) ConfigOption {
	return func(c *Config) {
		c.RepeatLastN = &n
	}
}

// SetMirostat enables Mirostat sampling for controlling perplexity.
// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
func SetMirostat(mode int) ConfigOption {
	return func(c *Config) {
		c.Mirostat = &mode
	}
}

// SetMirostatEta influences how quickly the algorithm responds to feedback from the generated text.
// A lower learning rate will result in slower adjustments, while a higher learning rate will make
// the algorithm more responsive. (Default: 0.1)
func SetMirostatEta(eta float64) ConfigOption {
	return func(c *Config) {
		c.MirostatEta = &eta
	}
}

// SetMirostatTau controls the balance between coherence and diversity of the output.
// A lower value will result in more focused and coherent text. (Default: 5.0)
func SetMirostatTau(tau float64) ConfigOption {
	return func(c *Config) {
		c.MirostatTau = &tau
	}
}

// SetTfsZ sets the value for tail free sampling, which is used to reduce the impact of less probable tokens.
// A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (Default: 1)
func SetTfsZ(z float64) ConfigOption {
	return func(c *Config) {
		c.TfsZ = &z
	}
}
