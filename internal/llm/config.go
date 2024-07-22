// File: internal/llm/config.go

package llm

import (
	"time"

	"github.com/caarlos0/env/v11"
)

type Config struct {
	Provider    string        `env:"LLM_PROVIDER" envDefault:"anthropic"`
	Model       string        `env:"LLM_MODEL" envDefault:"claude-3-opus-20240229"`
	Temperature float64       `env:"LLM_TEMPERATURE" envDefault:"0.7"`
	MaxTokens   int           `env:"LLM_MAX_TOKENS" envDefault:"100"`
	APIKey      string        `env:"LLM_API_KEY,required"`
	Timeout     time.Duration `env:"LLM_TIMEOUT" envDefault:"30s"`
}

func LoadConfig() (*Config, error) {
	cfg := &Config{}
	if err := env.Parse(cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}
