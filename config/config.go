// File: config/config.go

package config

import (
	"os"
	"strings"
	"time"

	"github.com/caarlos0/env/v11"
	"github.com/teilomillet/gollm/utils"
)

type MemoryOption struct {
	MaxTokens int
}

type Config struct {
	Provider              string        `env:"LLM_PROVIDER" envDefault:"anthropic"`
	Model                 string        `env:"LLM_MODEL" envDefault:"claude-3-opus-20240229"`
	OllamaEndpoint        string        `env:"OLLAMA_ENDPOINT" envDefault:"http://localhost:11434"`
	Temperature           float64       `env:"LLM_TEMPERATURE" envDefault:"0.7"`
	MaxTokens             int           `env:"LLM_MAX_TOKENS" envDefault:"100"`
	TopP                  float64       `env:"LLM_TOP_P" envDefault:"0.9"`
	FrequencyPenalty      float64       `env:"LLM_FREQUENCY_PENALTY" envDefault:"0.0"`
	PresencePenalty       float64       `env:"LLM_PRESENCE_PENALTY" envDefault:"0.0"`
	Timeout               time.Duration `env:"LLM_TIMEOUT" envDefault:"30s"`
	MaxRetries            int           `env:"LLM_MAX_RETRIES" envDefault:"3"`
	RetryDelay            time.Duration `env:"LLM_RETRY_DELAY" envDefault:"2s"`
	APIKeys               map[string]string
	LogLevel              utils.LogLevel `env:"LLM_LOG_LEVEL" envDefault:"WARN"`
	Seed                  *int           `env:"LLM_SEED"`
	MinP                  *float64       `env:"LLM_MIN_P" envDefault:"0.05"`
	RepeatPenalty         *float64       `env:"LLM_REPEAT_PENALTY" envDefault:"1.1"`
	RepeatLastN           *int           `env:"LLM_REPEAT_LAST_N" envDefault:"64"`
	Mirostat              *int           `env:"LLM_MIROSTAT" envDefault:"0"`
	MirostatEta           *float64       `env:"LLM_MIROSTAT_ETA" envDefault:"0.1"`
	MirostatTau           *float64       `env:"LLM_MIROSTAT_TAU" envDefault:"5.0"`
	TfsZ                  *float64       `env:"LLM_TFS_Z" envDefault:"1"`
	SystemPrompt          string
	SystemPromptCacheType string
	ExtraHeaders          map[string]string
	EnableCaching         bool `env:"LLM_ENABLE_CACHING" envDefault:"false"`
	MemoryOption          *MemoryOption
}

func LoadConfig() (*Config, error) {
	cfg := &Config{
		APIKeys: make(map[string]string),
	}
	if err := env.Parse(cfg); err != nil {
		return nil, err
	}

	loadAPIKeys(cfg)
	return cfg, nil
}

func loadAPIKeys(cfg *Config) {
	for _, envVar := range os.Environ() {
		key, value, found := strings.Cut(envVar, "=")
		if found && strings.HasSuffix(strings.ToUpper(key), "_API_KEY") {
			provider := strings.TrimSuffix(strings.ToUpper(key), "_API_KEY")
			cfg.APIKeys[strings.ToLower(provider)] = value
		}
	}

	// Ensure the default provider has an API key
	if apiKey, exists := cfg.APIKeys[strings.ToUpper(cfg.Provider)]; exists {
		cfg.APIKeys[cfg.Provider] = apiKey
	}
}

type ConfigOption func(*Config)

func NewConfig() *Config {
	return &Config{
		Provider:     "openai",
		Model:        "gpt-4o-mini",
		Temperature:  0.7,
		MaxTokens:    300,
		Timeout:      30 * time.Second,
		MaxRetries:   3,
		RetryDelay:   2 * time.Second,
		APIKeys:      make(map[string]string),
		LogLevel:     utils.LogLevelWarn,
		ExtraHeaders: make(map[string]string),
	}
}

func SetEnableCaching(enableCaching bool) ConfigOption {
	return func(c *Config) {
		c.EnableCaching = enableCaching
	}
}

func SetProvider(provider string) ConfigOption {
	return func(c *Config) {
		c.Provider = provider
	}
}

func SetModel(model string) ConfigOption {
	return func(c *Config) {
		c.Model = model
	}
}

func SetOllamaEndpoint(endpoint string) ConfigOption {
	return func(c *Config) {
		c.OllamaEndpoint = endpoint
	}
}

func SetTemperature(temperature float64) ConfigOption {
	return func(c *Config) {
		c.Temperature = temperature
	}
}

func SetMaxTokens(maxTokens int) ConfigOption {
	return func(c *Config) {
		if maxTokens < 1 {
			maxTokens = 1
		}
		c.MaxTokens = maxTokens
	}
}

func SetTimeout(timeout time.Duration) ConfigOption {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

func SetAPIKey(apiKey string) ConfigOption {
	return func(c *Config) {
		if c.APIKeys == nil {
			c.APIKeys = make(map[string]string)
		}
		c.APIKeys[c.Provider] = apiKey
	}
}

func SetMaxRetries(maxRetries int) ConfigOption {
	return func(c *Config) {
		c.MaxRetries = maxRetries
	}
}

func SetRetryDelay(retryDelay time.Duration) ConfigOption {
	return func(c *Config) {
		c.RetryDelay = retryDelay
	}
}

func SetLogLevel(level utils.LogLevel) ConfigOption {
	return func(c *Config) {
		c.LogLevel = level
	}
}

func SetMemory(maxTokens int) ConfigOption {
	return func(c *Config) {
		c.MemoryOption = &MemoryOption{
			MaxTokens: maxTokens,
		}
	}
}

func SetExtraHeaders(headers map[string]string) ConfigOption {
	return func(c *Config) {
		if c.ExtraHeaders == nil {
			c.ExtraHeaders = make(map[string]string)
		}
		for k, v := range headers {
			c.ExtraHeaders[k] = v
		}
	}
}

func SetTopP(topP float64) ConfigOption {
	return func(c *Config) {
		c.TopP = topP
	}
}

func SetFrequencyPenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.FrequencyPenalty = penalty
	}
}

func SetPresencePenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.PresencePenalty = penalty
	}
}

func SetSeed(seed int) ConfigOption {
	return func(c *Config) {
		c.Seed = &seed
	}
}

func SetMinP(minP float64) ConfigOption {
	return func(c *Config) {
		c.MinP = &minP
	}
}

func SetRepeatPenalty(penalty float64) ConfigOption {
	return func(c *Config) {
		c.RepeatPenalty = &penalty
	}
}

func SetRepeatLastN(n int) ConfigOption {
	return func(c *Config) {
		c.RepeatLastN = &n
	}
}

func SetMirostat(mode int) ConfigOption {
	return func(c *Config) {
		c.Mirostat = &mode
	}
}

func SetMirostatEta(eta float64) ConfigOption {
	return func(c *Config) {
		c.MirostatEta = &eta
	}
}

func SetMirostatTau(tau float64) ConfigOption {
	return func(c *Config) {
		c.MirostatTau = &tau
	}
}

func SetTfsZ(z float64) ConfigOption {
	return func(c *Config) {
		c.TfsZ = &z
	}
}

func ApplyOptions(cfg *Config, options ...ConfigOption) {
	for _, option := range options {
		option(cfg)
	}
}
