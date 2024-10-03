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
	Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
	GenerateWithSchema(ctx context.Context, prompt string, schema interface{}) (string, string, error)
	SetOption(key string, value interface{})
	SetDebugLevel(level utils.LogLevel)
	SetEndpoint(endpoint string)
	NewPrompt(prompt string) *Prompt
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

func NewLLM(config *config.Config, logger utils.Logger, registry *providers.ProviderRegistry) (LLM, error) {
	extraHeaders := make(map[string]string)
	if config.Provider == "anthropic" && config.EnableCaching {
		extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	provider, err := registry.Get(config.Provider, config.APIKeys[config.Provider], config.Model, extraHeaders)
	if err != nil {
		return nil, err
	}

	llmClient := &LLMImpl{
		Provider:   provider,
		Options:    make(map[string]interface{}),
		client:     &http.Client{Timeout: config.Timeout},
		logger:     logger,
		config:     config,
		MaxRetries: config.MaxRetries,
		RetryDelay: config.RetryDelay,
	}

	// Set common options for all providers
	llmClient.SetOption("temperature", config.Temperature)
	llmClient.SetOption("max_tokens", config.MaxTokens)

	if config.Seed != nil {
		llmClient.SetOption("seed", *config.Seed)
	}

	// Special handling for Ollama provider
	if config.Provider == "ollama" {
		if config.OllamaEndpoint != "" {
			llmClient.SetEndpoint(config.OllamaEndpoint)
		}
		// Set Ollama-specific options
		llmClient.SetOption("top_p", config.TopP)
		llmClient.SetOption("min_p", config.MinP)
		llmClient.SetOption("repeat_penalty", config.RepeatPenalty)
		llmClient.SetOption("repeat_last_n", config.RepeatLastN)
		llmClient.SetOption("mirostat", config.Mirostat)
		llmClient.SetOption("mirostat_eta", config.MirostatEta)
		llmClient.SetOption("mirostat_tau", config.MirostatTau)
		llmClient.SetOption("tfs_z", config.TfsZ)
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
func (l *LLMImpl) SetDebugLevel(level utils.LogLevel) {
	l.logger.Debug("Setting internal LLM debug level", "new_level", level)
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

func (l *LLMImpl) Generate(ctx context.Context, prompt string) (string, string, error) {
	var result string
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text", "provider", l.Provider.Name(), "prompt", prompt, "attempt", attempt+1)

		result, lastErr = l.attemptGenerate(ctx, prompt)
		if lastErr == nil {
			return result, prompt, nil
		}

		l.logger.Warn("Generation attempt failed", "error", lastErr, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			select {
			case <-ctx.Done():
				return "", prompt, ctx.Err()
			case <-time.After(l.RetryDelay):
				// Continue to next attempt
			}
		}
	}

	return "", prompt, fmt.Errorf("failed to generate after %d attempts: %w", l.MaxRetries+1, lastErr)
}

func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt string) (string, error) {
	reqBody, err := l.Provider.PrepareRequest(prompt, l.Options)
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}
	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

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

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return "", NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return "", NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	l.logger.Debug("Text generated successfully", "result", result)
	return result, nil
}

func (l *LLMImpl) GenerateWithSchema(ctx context.Context, prompt string, schema interface{}) (string, string, error) {
	var result string
	var fullPrompt string
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text with schema", "provider", l.Provider.Name(), "prompt", prompt, "attempt", attempt+1)

		result, fullPrompt, lastErr = l.attemptGenerateWithSchema(ctx, prompt, schema)
		if lastErr == nil {
			return result, fullPrompt, nil
		}

		l.logger.Warn("Generation attempt with schema failed", "error", lastErr, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			select {
			case <-ctx.Done():
				return "", fullPrompt, ctx.Err()
			case <-time.After(l.RetryDelay):
				// Continue to next attempt
			}
		}
	}

	return "", fullPrompt, fmt.Errorf("failed to generate with schema after %d attempts: %w", l.MaxRetries+1, lastErr)
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
