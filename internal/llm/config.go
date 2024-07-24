// File: internal/llm/config.go

package llm

import (
	"os"
	"strings"
	"time"

	"github.com/caarlos0/env/v11"
)

type Config struct {
	Provider    string        `env:"LLM_PROVIDER" envDefault:"anthropic"`
	Model       string        `env:"LLM_MODEL" envDefault:"claude-3-opus-20240229"`
	Temperature float64       `env:"LLM_TEMPERATURE" envDefault:"0.7"`
	MaxTokens   int           `env:"LLM_MAX_TOKENS" envDefault:"100"`
	Timeout     time.Duration `env:"LLM_TIMEOUT" envDefault:"30s"`
	MaxRetries  int           `env:"LLM_MAX_RETRIES" envDefault:"3"`
	RetryDelay  time.Duration `env:"LLM_RETRY_DELAY" envDefault:"2s"`
	APIKeys     map[string]string
	LogLevel    LogLevel `env:"LLM_LOG_LEVEL" envDefault:"WARN"`
}

func LoadConfig() (*Config, error) {
	cfg := &Config{
		APIKeys: make(map[string]string),
	}
	if err := env.Parse(cfg); err != nil {
		return nil, err
	}

	// log.Printf("Config after env.Parse: %+v", cfg)

	// Manually parse API keys from environment variables
	for _, envVar := range os.Environ() {
		parts := strings.SplitN(envVar, "=", 2)
		if len(parts) == 2 {
			key, value := parts[0], parts[1]
			if strings.HasSuffix(strings.ToUpper(key), "_API_KEY") {
				provider := strings.TrimSuffix(strings.ToUpper(key), "_API_KEY")
				cfg.APIKeys[provider] = value
				// log.Printf("Found API key for provider: %s", provider)
			}
		}
	}

	// Ensure the default provider has an API key
	if apiKey, exists := cfg.APIKeys[strings.ToUpper(cfg.Provider)]; exists {
		cfg.APIKeys[cfg.Provider] = apiKey
	}

	// log.Printf("Final config: %+v", cfg)

	return cfg, nil
}
