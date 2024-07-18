package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
)

// LLM defines the common interface for all LLM providers
type LLM interface {
	Generate(ctx context.Context, prompt string) (string, error)
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
}

// Generate generates text based on the given prompt
func (l *LLMImpl) Generate(ctx context.Context, prompt string) (string, error) {
	reqBody, err := l.Provider.PrepareRequest(prompt, l.Options)
	if err != nil {
		return "", fmt.Errorf("error preparing request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response body: %w", err)
	}

	return l.Provider.ParseResponse(body)
}

