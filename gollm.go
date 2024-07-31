// File: gollm.go

package gollm

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"

	"github.com/teilomillet/gollm/internal/llm"
)

// validate is a global validator instance used for struct validation
var validate *validator.Validate

// init initializes the global validator instance
func init() {
	validate = validator.New()
}

// Validate checks if the given struct is valid according to its validation rules
func Validate(s interface{}) error {
	return validate.Struct(s)
}

// LLM is the interface that wraps the basic LLM operations
type LLM interface {
	// Generate produces a response given a context, prompt, and optional generate options
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error)

	// SetOption sets an option for the LLM
	SetOption(key string, value interface{})

	// GetPromptJSONSchema returns the JSON schema for the prompt
	GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error)

	// GetProvider returns the provider of the LLM
	GetProvider() string

	// GetModel returns the model of the LLM
	GetModel() string

	// UpdateDebugLevel updates the debug level of the LLM
	UpdateDebugLevel(level LogLevel)

	// Debug logs a debug message with optional key-value pairs
	Debug(msg string, keysAndValues ...interface{})

	// GetDebugLevel returns the current debug level of the LLM
	GetDebugLevel() LogLevel
}

// llmImpl is the concrete implementation of the LLM interface
type llmImpl struct {
	llm.LLM  // Embedded LLM interface from internal package
	logger   llm.Logger
	provider string
	model    string
	config   *Config
}

// GenerateOption is a function type for configuring generate options
type GenerateOption func(*generateConfig)

// generateConfig holds configuration options for the Generate method
type generateConfig struct {
	useJSONSchema bool
}

// WithJSONSchemaValidation returns a GenerateOption that enables JSON schema validation
func WithJSONSchemaValidation() GenerateOption {
	return func(c *generateConfig) {
		c.useJSONSchema = true
	}
}

// GetProvider returns the provider of the LLM
func (l *llmImpl) GetProvider() string {
	return l.provider
}

// GetModel returns the model of the LLM
func (l *llmImpl) GetModel() string {
	return l.model
}

// Debug logs a debug message with optional key-value pairs
func (l *llmImpl) Debug(msg string, keysAndValues ...interface{}) {
	l.logger.Debug(msg, keysAndValues...)
}

// GetDebugLevel returns the current debug level of the LLM
func (l *llmImpl) GetDebugLevel() LogLevel {
	return l.config.DebugLevel
}

// SetOption sets an option for the LLM with the given key and value
func (l *llmImpl) SetOption(key string, value interface{}) {
	// Log the attempt to set an option
	l.logger.Debug("Setting option", "key", key, "value", value)

	// Call the SetOption method of the embedded LLM
	l.LLM.SetOption(key, value)

	// Log the successful setting of the option
	l.logger.Debug("Option set successfully")
}

// GetPromptJSONSchema generates and returns the JSON schema for the Prompt
// It accepts optional SchemaOptions to customize the schema generation
func (l *llmImpl) GetPromptJSONSchema(opts ...SchemaOption) ([]byte, error) {
	// Create a new Prompt instance
	p := &Prompt{}

	// Generate and return the JSON schema for the Prompt
	// Pass along any provided SchemaOptions
	return p.GenerateJSONSchema(opts...)
}

// UpdateDebugLevel updates the debug level for both the gollm package and the internal llm package
func (l *llmImpl) UpdateDebugLevel(level LogLevel) {
	l.logger.Debug("Updating debug level",
		"current_level", l.config.DebugLevel,
		"new_level", level)

	l.config.DebugLevel = level
	l.logger.SetLevel(llm.LogLevel(level))

	if internalLLM, ok := l.LLM.(interface{ SetDebugLevel(llm.LogLevel) }); ok {
		internalLLM.SetDebugLevel(llm.LogLevel(level))
		l.logger.Debug("Updated internal LLM debug level")
	} else {
		l.logger.Warn("Internal LLM does not support SetDebugLevel")
	}

	l.logger.Debug("Debug level updated successfully")
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
	// Log the start of the Generate method
	l.logger.Debug("Starting Generate method",
		"prompt_length", len(prompt.String()),
		"context", ctx)

	// Check if llmImpl is nil
	if l == nil {
		l.logger.Error("llmImpl is nil")
		return "", fmt.Errorf("llmImpl is nil")
	}

	// Check if internal LLM is nil
	if l.LLM == nil {
		l.logger.Error("internal LLM is nil")
		return "", fmt.Errorf("internal LLM is nil")
	}

	// Ensure logger is initialized
	if l.logger == nil {
		l.logger = llm.NewLogger(llm.LogLevel(LogLevelWarn))
		l.logger.Warn("Logger was nil, created new logger with WARN level")
	}

	// Apply generate options
	config := &generateConfig{}
	for _, opt := range opts {
		opt(config)
	}
	l.logger.Debug("Generate options applied",
		"useJSONSchema", config.useJSONSchema)

	// Validate prompt with JSON schema if enabled
	if config.useJSONSchema {
		l.logger.Debug("Validating prompt with JSON schema")
		if err := prompt.Validate(); err != nil {
			l.logger.Error("Prompt validation failed", "error", err)
			return "", fmt.Errorf("invalid prompt: %w", err)
		}
		l.logger.Debug("Prompt validation successful")
	}

	// Send prompt to LLM
	l.logger.Debug("Sending prompt to LLM",
		"prompt", prompt.String(),
		"provider", l.GetProvider(),
		"model", l.GetModel())
	response, fullPrompt, err := l.LLM.Generate(ctx, prompt.String())
	if err != nil {
		l.logger.Error("Error from LLM.Generate",
			"error", err,
			"fullPrompt", fullPrompt)
		return "", err
	}
	l.logger.Debug("Received response from LLM",
		"response_length", len(response))

	// Clean the response
	cleanedResponse := CleanResponse(response)
	l.logger.Debug("Response cleaned",
		"original_length", len(response),
		"cleaned_length", len(cleanedResponse))

	return cleanedResponse, nil
}

// NewLLM creates a new LLM instance
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
	logger.Debug("Creating new LLM",
		"provider", config.Provider,
		"model", config.Model,
		"debug_level", config.DebugLevel)

	internalConfig := config.toInternalConfig()
	logger.Debug("Internal config created",
		"max_tokens", internalConfig.MaxTokens,
		"temperature", internalConfig.Temperature)

	l, err := llm.NewLLM(internalConfig, logger, llm.NewProviderRegistry())
	if err != nil {
		logger.Error("Failed to create internal LLM", "error", err)
		return nil, fmt.Errorf("failed to create internal LLM: %w", err)
	}

	if l == nil {
		logger.Error("Internal LLM is nil after creation")
		return nil, fmt.Errorf("internal LLM is nil after creation")
	}

	logger.Debug("Internal LLM created successfully")

	return &llmImpl{
		LLM:      l,
		logger:   logger,
		provider: config.Provider,
		model:    config.Model,
		config:   config,
	}, nil
}
