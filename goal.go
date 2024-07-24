// File: goal.go

package goal

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"

	"github.com/teilomillet/goal/internal/llm"
)

var validate *validator.Validate

func init() {
	validate = validator.New()
}

// Validate checks if the given struct is valid according to its validation rules
func Validate(s interface{}) error {
	return validate.Struct(s)
}

type LLM interface {
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error)
	SetOption(key string, value interface{})
	GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error)
	GetProvider() string
	GetModel() string
}

type llmImpl struct {
	llm.LLM
	logger   llm.Logger
	provider string
	model    string
}

type GenerateOption func(*generateConfig)

type generateConfig struct {
	useJSONSchema bool
}

func WithJSONSchemaValidation() GenerateOption {
	return func(c *generateConfig) {
		c.useJSONSchema = true
	}
}

func (l *llmImpl) GetProvider() string {
	return l.provider
}

func (l *llmImpl) GetModel() string {
	return l.model
}

// CleanResponse removes markdown code block syntax and trims the JSON response
func CleanResponse(response string) string {
	// Remove markdown code block syntax if present
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")

	// Remove any text before the first '{' and after the last '}'
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start != -1 && end != -1 && end > start {
		response = response[start : end+1]
	}

	return strings.TrimSpace(response)
}

func (l *llmImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	if l == nil {
		return "", fmt.Errorf("llmImpl is nil")
	}
	if l.LLM == nil {
		return "", fmt.Errorf("internal LLM is nil")
	}
	if l.logger == nil {
		l.logger = llm.NewLogger(llm.LogLevel(LogLevelWarn)) // Default to warn level if logger is not set
	}

	config := &generateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	if config.useJSONSchema {
		if err := prompt.Validate(); err != nil {
			return "", fmt.Errorf("invalid prompt: %w", err)
		}
	}

	l.logger.Debug("Sending prompt to LLM", "prompt", prompt.String())
	response, _, err := l.LLM.Generate(ctx, prompt.String())
	if err != nil {
		l.logger.Error("Error from LLM.Generate", "error", err)
		return "", err
	}

	// Clean the response
	cleanedResponse := CleanResponse(response)
	l.logger.Debug("Cleaned response", "response", cleanedResponse)

	return cleanedResponse, nil
}

func (l *llmImpl) SetOption(key string, value interface{}) {
	l.LLM.SetOption(key, value)
}

func (l *llmImpl) GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error) {
	p := &Prompt{}
	return p.GenerateJSONSchema(opts...)
}

func NewLLM(opts ...ConfigOption) (LLM, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	for _, opt := range opts {
		opt(config)
	}

	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	logger := llm.NewLogger(llm.LogLevel(config.DebugLevel))
	logger.Debug("Creating LLM", "config", fmt.Sprintf("%+v", config))

	internalConfig := config.toInternalConfig()

	l, err := llm.NewLLM(internalConfig, logger, llm.NewProviderRegistry())
	if err != nil {
		return nil, fmt.Errorf("failed to create internal LLM: %w", err)
	}

	if l == nil {
		return nil, fmt.Errorf("internal LLM is nil after creation")
	}

	return &llmImpl{
		LLM:      l,
		logger:   logger,
		provider: config.Provider,
		model:    config.Model,
	}, nil
}
