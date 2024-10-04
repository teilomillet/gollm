// llm/llm.go

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// LLM interface defines the methods that our internal language model should implement
type LLM interface {
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (response string, err error)
	GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error)
	SetOption(key string, value interface{})
	SetLogLevel(level utils.LogLevel)
	SetEndpoint(endpoint string)
	NewPrompt(input string) *Prompt
	GetLogger() utils.Logger
	SupportsJSONSchema() bool
}

// LLMImpl is our implementation of the internal LLM interface
type LLMImpl struct {
	Provider   providers.Provider
	Options    map[string]interface{}
	client     *http.Client
	logger     utils.Logger
	config     *config.Config
	MaxRetries int
	RetryDelay time.Duration
}

type GenerateOption func(*GenerateConfig)

type GenerateConfig struct {
	UseJSONSchema bool
}

func NewLLM(cfg *config.Config, logger utils.Logger, registry *providers.ProviderRegistry) (LLM, error) {
	extraHeaders := make(map[string]string)
	if cfg.Provider == "anthropic" && cfg.EnableCaching {
		extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	provider, err := registry.Get(cfg.Provider, cfg.APIKeys[cfg.Provider], cfg.Model, extraHeaders)
	if err != nil {
		return nil, err
	}

	provider.SetDefaultOptions(cfg)

	llmClient := &LLMImpl{
		Provider:   provider,
		client:     &http.Client{Timeout: cfg.Timeout},
		logger:     logger,
		config:     cfg,
		MaxRetries: cfg.MaxRetries,
		RetryDelay: cfg.RetryDelay,
		Options:    make(map[string]interface{}),
	}

	return llmClient, nil
}

func (l *LLMImpl) SetOption(key string, value interface{}) {
	l.Options[key] = value
	l.logger.Debug("Option set", key, value)
}

func (l *LLMImpl) SetEndpoint(endpoint string) {
	// This is a no-op for non-Ollama providers
	l.logger.Debug("SetEndpoint called on non-Ollama provider", "endpoint", endpoint)
}

// SetDebugLevel updates the debug level for the internal LLM
func (l *LLMImpl) SetLogLevel(level utils.LogLevel) {
	l.logger.Debug("Setting internal LLM log level", "new_level", level)
	l.logger.SetLevel(level)
}

func (l *LLMImpl) GetLogger() utils.Logger {
	return l.logger
}

func (l *LLMImpl) NewPrompt(prompt string) *Prompt {
	return &Prompt{Input: prompt}
}

func (l *LLMImpl) SupportsJSONSchema() bool {
	return l.Provider.SupportsJSONSchema()
}

func (l *LLMImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}
	// Set the system prompt in the LLM's options
	if prompt.SystemPrompt != "" {
		l.SetOption("system_prompt", prompt.SystemPrompt)
	}
	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text", "provider", l.Provider.Name(), "prompt", prompt.Input, "system_prompt", prompt.SystemPrompt, "attempt", attempt+1)
		// Pass the entire Prompt struct to attemptGenerate
		result, err := l.attemptGenerate(ctx, prompt)
		if err == nil {
			return result, nil
		}
		l.logger.Warn("Generation attempt failed", "error", err, "attempt", attempt+1)
		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			if err := l.wait(ctx); err != nil {
				return "", err
			}
		}
	}
	return "", fmt.Errorf("failed to generate after %d attempts", l.MaxRetries+1)
}

func (l *LLMImpl) wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(l.RetryDelay):
		return nil
	}
}

func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt *Prompt) (string, error) {
	// Prepare the request with both the user prompt and the options (which include the system prompt)
	reqBody, err := l.Provider.PrepareRequest(prompt.Input, l.Options)
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}
	l.logger.Debug("Full request body", "body", string(reqBody))
	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	l.logger.Debug("Full API request", "method", req.Method, "url", req.URL.String(), "headers", req.Header, "body", string(reqBody))
	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
		l.logger.Debug("Request header", "provider", l.Provider.Name(), "key", k, "value", v)
	}
	resp, err := l.client.Do(req)
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	// Log the full API response
	l.logger.Debug("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return "", NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	// Extract and log caching information
	var fullResponse map[string]interface{}
	if err := json.Unmarshal(body, &fullResponse); err == nil {
		if usage, ok := fullResponse["usage"].(map[string]interface{}); ok {
			l.logger.Debug("Usage information", "usage", usage)
			cacheInfo := map[string]interface{}{
				"cache_creation_input_tokens": usage["cache_creation_input_tokens"],
				"cache_read_input_tokens":     usage["cache_read_input_tokens"],
			}
			l.logger.Info("Cache information", "info", cacheInfo)
		} else {
			l.logger.Info("Cache information not available in the response")
		}
	} else {
		l.logger.Warn("Failed to parse response for cache information", "error", err)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return "", NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}
	l.logger.Debug("Text generated successfully", "result", result)
	return result, nil
}

func (l *LLMImpl) GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	var result string
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text with schema", "provider", l.Provider.Name(), "prompt", prompt.Input, "attempt", attempt+1)

		result, _, lastErr = l.attemptGenerateWithSchema(ctx, prompt.Input, schema)
		if lastErr == nil {
			return result, nil
		}

		l.logger.Warn("Generation attempt with schema failed", "error", lastErr, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(l.RetryDelay):
				// Continue to next attempt
			}
		}
	}

	return "", fmt.Errorf("failed to generate with schema after %d attempts: %w", l.MaxRetries+1, lastErr)
}

func (l *LLMImpl) attemptGenerateWithSchema(ctx context.Context, prompt string, schema interface{}) (string, string, error) {
	var reqBody []byte
	var err error
	var fullPrompt string

	if l.SupportsJSONSchema() {
		reqBody, err = l.Provider.PrepareRequestWithSchema(prompt, l.Options, schema)
		fullPrompt = prompt
	} else {
		fullPrompt = l.preparePromptWithSchema(prompt, schema)
		reqBody, err = l.Provider.PrepareRequest(fullPrompt, l.Options)
	}

	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return "", fullPrompt, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	// Validate the result against the schema
	if err := ValidateAgainstSchema(result, schema); err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "response does not match schema", err)
	}

	l.logger.Debug("Text generated successfully", "result", result)
	return result, fullPrompt, nil
}

func (l *LLMImpl) preparePromptWithSchema(prompt string, schema interface{}) string {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		l.logger.Warn("Failed to marshal schema", "error", err)
		return prompt
	}

	return fmt.Sprintf("%s\n\nPlease provide your response in JSON format according to this schema:\n%s", prompt, string(schemaJSON))
}
