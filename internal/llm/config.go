// File: internal/llm/config.go

package llm

import (
	"os"
	"strings"
	"time"

	"github.com/caarlos0/env/v11"
)

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
	LogLevel              LogLevel `env:"LLM_LOG_LEVEL" envDefault:"WARN"`
	Seed                  *int     `env:"LLM_SEED"`
	MinP                  *float64 `env:"LLM_MIN_P" envDefault:"0.05"`
	RepeatPenalty         *float64 `env:"LLM_REPEAT_PENALTY" envDefault:"1.1"`
	RepeatLastN           *int     `env:"LLM_REPEAT_LAST_N" envDefault:"64"`
	Mirostat              *int     `env:"LLM_MIROSTAT" envDefault:"0"`
	MirostatEta           *float64 `env:"LLM_MIROSTAT_ETA" envDefault:"0.1"`
	MirostatTau           *float64 `env:"LLM_MIROSTAT_TAU" envDefault:"5.0"`
	TfsZ                  *float64 `env:"LLM_TFS_Z" envDefault:"1"`
	SystemPrompt          string
	SystemPromptCacheType string
	ExtraHeaders          map[string]string
	EnableCaching         bool `env:"LLM_ENABLE_CACHING" envDefault:"false"`
}

func LoadConfig() (*Config, error) {
	cfg := &Config{
		APIKeys: make(map[string]string),
	}
	if err := env.Parse(cfg); err != nil {
		return nil, err
	}

	// Manually parse API keys from environment variables
	for _, envVar := range os.Environ() {
		parts := strings.SplitN(envVar, "=", 2)
		if len(parts) == 2 {
			key, value := parts[0], parts[1]
			if strings.HasSuffix(strings.ToUpper(key), "_API_KEY") {
				provider := strings.TrimSuffix(strings.ToUpper(key), "_API_KEY")
				cfg.APIKeys[provider] = value
			}
		}
	}

	// Ensure the default provider has an API key
	if apiKey, exists := cfg.APIKeys[strings.ToUpper(cfg.Provider)]; exists {
		cfg.APIKeys[cfg.Provider] = apiKey
	}

	return cfg, nil
}
