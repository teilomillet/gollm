// File: llm/llm.go

package llm

import (
	"bytes"
	"context"
	"go.uber.org/zap"
	"io"
	"net/http"
)

// LLM defines the common interface for all LLM providers
type LLM interface {
	Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
	SetOption(key string, value interface{})
}

// Provider defines the interface for different LLM providers
type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
}

// LLMImpl represents a generic language model instance
type LLMImpl struct {
	Provider Provider
	Options  map[string]interface{}
	client   *http.Client
}

// NewLLM creates a new LLM instance
func NewLLM(provider Provider) LLM {
	return &LLMImpl{
		Provider: provider,
		Options:  make(map[string]interface{}),
		client:   &http.Client{},
	}
}

// SetOption sets an option for the LLM
func (l *LLMImpl) SetOption(key string, value interface{}) {
	l.Options[key] = value
	Logger.Info("Option set", zap.String("key", key), zap.Any("value", value))
}

// Generate generates text based on the given prompt
func (l *LLMImpl) Generate(ctx context.Context, prompt string) (string, string, error) {
	Logger.Info("Generating text", zap.String("provider", l.Provider.Name()), zap.String("prompt", prompt))

	reqBody, err := l.Provider.PrepareRequest(prompt, l.Options)
	if err != nil {
		Logger.Error("Failed to prepare request", zap.Error(err))
		return "", prompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		Logger.Error("Failed to create request", zap.Error(err))
		return "", prompt, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		Logger.Error("Failed to send request", zap.Error(err))
		return "", prompt, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		Logger.Error("API error", zap.Int("status_code", resp.StatusCode))
		return "", prompt, NewLLMError(ErrorTypeAPI, "API error", nil)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		Logger.Error("Failed to read response body", zap.Error(err))
		return "", prompt, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		Logger.Error("Failed to parse response", zap.Error(err))
		return "", prompt, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	Logger.Info("Text generated successfully", zap.String("result", result))
	return result, prompt, nil
}

