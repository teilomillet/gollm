// File: llm/llm.go

package llm

import (
	"bytes"
	"context"
	"io"
	"net/http"
)

type LLM interface {
	Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
	SetOption(key string, value interface{})
}

type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
}

type LLMImpl struct {
	Provider Provider
	Options  map[string]interface{}
	client   *http.Client
	logger   Logger
}

func NewLLM(config *Config, logger Logger, registry *ProviderRegistry) (LLM, error) {
	provider, err := registry.Get(config.Provider, config.APIKey, config.Model)
	if err != nil {
		return nil, err
	}

	llmClient := &LLMImpl{
		Provider: provider,
		Options:  make(map[string]interface{}),
		client:   &http.Client{Timeout: config.Timeout},
		logger:   logger,
	}
	llmClient.SetOption("temperature", config.Temperature)
	llmClient.SetOption("max_tokens", config.MaxTokens)

	return llmClient, nil
}

func (l *LLMImpl) SetOption(key string, value interface{}) {
	l.Options[key] = value
	l.logger.Debug("Option set", key, value)
}

func (l *LLMImpl) Generate(ctx context.Context, prompt string) (string, string, error) {
	l.logger.Info("Generating text", "provider", l.Provider.Name(), "prompt", prompt)

	reqBody, err := l.Provider.PrepareRequest(prompt, l.Options)
	if err != nil {
		l.logger.Error("Failed to prepare request", "error", err)
		return "", prompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		l.logger.Error("Failed to create request", "error", err)
		return "", prompt, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		l.logger.Error("Failed to send request", "error", err)
		return "", prompt, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "status_code", resp.StatusCode)
		return "", prompt, NewLLMError(ErrorTypeAPI, "API error", nil)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		l.logger.Error("Failed to read response body", "error", err)
		return "", prompt, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		l.logger.Error("Failed to parse response", "error", err)
		return "", prompt, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	l.logger.Info("Text generated successfully", "result", result)
	return result, prompt, nil
}
