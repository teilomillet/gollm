// Package llm provides a unified interface for interacting with various Language Learning Model providers.
// It abstracts away provider-specific implementations and provides a consistent API for text generation,
// prompt management, and error handling.
package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
	"github.com/weave-labs/gollm/providers"
)

const (
	DefaultBufferSize      = 100
	DefaultRetryMultiplier = 10
)

// LLM interface defines the methods that our internal language model should implement.
// It provides a unified way to interact with different LLM providers while abstracting
// away provider-specific details.
type LLM interface {
	// Generate produces text based on the given prompt and options.
	// Returns ErrorTypeRequest for request preparation failures,
	// ErrorTypeAPI for provider API errors, or ErrorTypeResponse for response processing issues.
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (response *providers.Response, err error)
	// GenerateStream initiates a streaming response from the LLM.
	// Returns ErrorTypeUnsupported if the provider doesn't support streaming.
	GenerateStream(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (TokenStream, error)
	// SetOption configures a provider-specific option.
	// Returns ErrorTypeInvalidInput if the option is not supported.
	SetOption(key string, value any)
	// SetLogLevel adjusts the logging verbosity.
	SetLogLevel(level logging.LogLevel)
	// SetEndpoint updates the API endpoint (primarily for local models).
	// Returns ErrorTypeProvider if the provider doesn't support endpoint configuration.
	SetEndpoint(endpoint string)
	// GetLogger returns the current logger instance.
	GetLogger() logging.Logger
}

// LLMImpl implements the LLM interface and manages interactions with specific providers.
// It handles provider communication, error management, and logging.
//
//nolint:revive // LLMImpl clearly indicates this is the implementation of the LLM interface
type LLMImpl struct {
	Provider     providers.Provider
	logger       logging.Logger
	Options      map[string]any
	client       *http.Client
	config       *config.Config
	MaxRetries   int
	RetryDelay   time.Duration
	optionsMutex sync.RWMutex
}

// NewLLM creates a new LLM instance with the specified configuration.
// It initializes the appropriate provider and sets up logging and HTTP clients.
func NewLLM(cfg *config.Config, logger logging.Logger, registry *providers.ProviderRegistry) (*LLMImpl, error) {
	extraHeaders := make(map[string]string)

	apiKey := cfg.APIKeys[cfg.Provider]
	if apiKey == "" {
		return nil, NewLLMError(ErrorTypeAuthentication, "empty API key", nil)
	}

	provider, err := registry.Get(cfg.Provider, apiKey, cfg.Model, extraHeaders)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider %s: %w", cfg.Provider, err)
	}

	provider.SetDefaultOptions(cfg)

	llmClient := &LLMImpl{
		Provider:   provider,
		client:     &http.Client{Timeout: cfg.Timeout},
		logger:     logger,
		config:     cfg,
		MaxRetries: cfg.MaxRetries,
		RetryDelay: cfg.RetryDelay,
		Options:    make(map[string]any),
	}

	return llmClient, nil
}

// SetOption sets a provider-specific option with the given key and value.
func (l *LLMImpl) SetOption(key string, value any) {
	l.optionsMutex.Lock()
	defer l.optionsMutex.Unlock()

	l.Options[key] = value
	l.logger.Debug("Option set", key, value)
}

// SetEndpoint updates the API endpoint for the provider.
// This is primarily used for local models like Ollama.
func (l *LLMImpl) SetEndpoint(endpoint string) {
	// This is a no-op for non-Ollama providers
	l.logger.Debug("SetEndpoint called on non-Ollama provider", "endpoint", endpoint)
}

// SetLogLevel updates the logging verbosity level.
func (l *LLMImpl) SetLogLevel(level logging.LogLevel) {
	l.logger.Debug("Setting internal LLM log level", "new_level", level)
	l.logger.SetLevel(level)
}

// GetLogger returns the current logger instance.
func (l *LLMImpl) GetLogger() logging.Logger {
	return l.logger
}

// SupportsStreaming checks if the provider supports streaming responses.
func (l *LLMImpl) SupportsStreaming() bool {
	return l.Provider.SupportsStreaming()
}

// Generate produces text based on the given prompt and options.
func (l *LLMImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (*providers.Response, error) {
	generateConfig := &GenerateConfig{}
	for _, opt := range opts {
		opt(generateConfig)
	}

	if prompt.SystemPrompt != "" {
		l.SetOption("system_prompt", prompt.SystemPrompt)
	}

	return l.generateWithRetries(ctx, prompt, generateConfig.StructuredResponseSchema)
}

// GenerateStream initiates a streaming response from the LLM.
func (l *LLMImpl) GenerateStream(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (TokenStream, error) {
	if !l.SupportsStreaming() {
		return nil, NewLLMError(ErrorTypeUnsupported, "streaming not supported by provider", nil)
	}

	generateConfig := &GenerateConfig{
		StreamBufferSize: DefaultBufferSize,
		RetryStrategy: &DefaultRetryStrategy{
			MaxRetries:  l.MaxRetries,
			InitialWait: l.RetryDelay,
			MaxWait:     l.RetryDelay * DefaultRetryMultiplier,
		},
	}
	for _, opt := range opts {
		opt(generateConfig)
	}

	options := make(map[string]any)
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()
	options["stream"] = true

	builder := providers.NewRequestBuilder().WithPrompt(prompt.Input)

	if prompt.SystemPrompt != "" {
		builder.WithSystemPrompt(prompt.SystemPrompt)
	}

	if generateConfig.StructuredResponseSchema != nil {
		builder.WithResponseSchema(generateConfig.StructuredResponseSchema)
	}

	providerReq := builder.Build()
	body, err := l.Provider.PrepareStreamRequest(providerReq, options)
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to prepare stream request", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, l.Provider.Endpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to create stream request", err)
	}

	for k, v := range l.Provider.Headers() {
		httpReq.Header.Set(k, v)
	}

	resp, err := l.client.Do(httpReq)
	if err != nil {
		return nil, NewLLMError(ErrorTypeAPI, "failed to make stream request", err)
	}

	if resp.StatusCode != http.StatusOK {
		if err := resp.Body.Close(); err != nil {
			l.logger.Error("Failed to close response body", "error", err)
		}
		return nil, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	return newProviderStream(resp.Body, l.Provider, generateConfig), nil
}

// generateWithRetries handles standard generation with retry logic
func (l *LLMImpl) generateWithRetries(ctx context.Context, prompt *Prompt, schema any) (*providers.Response, error) {
	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		result, err := l.attemptGenerate(ctx, prompt, schema)
		if err == nil {
			return result, nil
		}

		l.logger.Warn("Generation attempt failed", "error", err, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			if err := l.wait(ctx); err != nil {
				return nil, err
			}
		}
	}

	return nil, fmt.Errorf("failed to generate after %d attempts", l.MaxRetries+1)
}

// wait implements a cancellable delay between retry attempts.
// Returns context.Canceled if the context is cancelled during the wait.
func (l *LLMImpl) wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return fmt.Errorf("context cancelled during retry wait: %w", ctx.Err())
	case <-time.After(l.RetryDelay):
		return nil
	}
}

// attemptGenerate makes a single attempt to generate text using the provider.
// It handles request preparation, API communication, and response processing.
func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt *Prompt, schema any) (*providers.Response, error) {
	response := &providers.Response{}

	reqBody, err := l.prepareRequestBody(prompt, schema)
	if err != nil {
		return response, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	response, err = l.executeRequest(ctx, reqBody)
	if err != nil {
		return response, err
	}

	if schema != nil {
		textContent, ok := response.Content.(providers.Text)
		if !ok {
			return nil, NewLLMError(ErrorTypeResponse, "response content is not text", nil)
		}

		if err := ValidateAgainstSchema(textContent.Value, schema); err != nil {
			return nil, NewLLMError(ErrorTypeResponse, "response does not match schema", err)
		}
	}

	return response, nil
}

// prepareOptions creates the options map for the request
func (l *LLMImpl) prepareOptions(prompt *Prompt) map[string]any {
	options := make(map[string]any)

	// Safely read from the Options map
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if len(prompt.ToolChoice) > 0 {
		options["tool_choice"] = prompt.ToolChoice
	}

	return options
}

// prepareRequestBody prepares the request body using the new provider architecture
func (l *LLMImpl) prepareRequestBody(prompt *Prompt, schema any) ([]byte, error) {
	options := l.prepareOptions(prompt)

	builder := providers.NewRequestBuilder()
	builder.WithSystemPrompt(prompt.SystemPrompt).
		WithMessages(ToMessages(prompt.Messages)).
		WithPrompt(prompt.Input)

	if schema != nil {
		builder.WithResponseSchema(schema)
	}

	if prompt.SystemPrompt != "" {
		builder.WithSystemPrompt(prompt.SystemPrompt)
	}

	req := builder.Build()

	reqBody, err := l.Provider.PrepareRequest(req, options)
	if err != nil {
		return nil, fmt.Errorf("provider PrepareRequest failed: %w", err)
	}

	return reqBody, nil
}

// executeRequest sends the HTTP request and processes the response
func (l *LLMImpl) executeRequest(ctx context.Context, reqBody []byte) (*providers.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}

	defer func() {
		if err := resp.Body.Close(); err != nil {
			l.logger.Error("Failed to close response body", "error", err)
		}
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return nil, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	response, err := l.Provider.ParseResponse(body)
	if err != nil {
		return nil, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	return response, nil
}

// providerStream implements TokenStream for a specific provider
type providerStream struct {
	provider      providers.Provider
	retryStrategy RetryStrategy
	decoder       *SSEDecoder
	config        *GenerateConfig
	buffer        []byte
	currentIndex  int
}

func newProviderStream(reader io.ReadCloser, provider providers.Provider, cfg *GenerateConfig) *providerStream {
	return &providerStream{
		decoder:       NewSSEDecoder(reader),
		provider:      provider,
		config:        cfg,
		buffer:        make([]byte, 0, DefaultStreamBufferSize),
		currentIndex:  0,
		retryStrategy: cfg.RetryStrategy,
	}
}

func (s *providerStream) Next(ctx context.Context) (*StreamToken, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context canceled: %w", ctx.Err())
		default:
			token, shouldContinue, err := s.processNextEvent()
			if err != nil {
				return nil, err
			}
			if shouldContinue {
				continue
			}
			return token, nil
		}
	}
}

// processNextEvent handles the next event from the decoder
func (s *providerStream) processNextEvent() (*StreamToken, bool, error) {
	if !s.decoder.Next() {
		return s.handleDecoderEnd()
	}

	event := s.decoder.Event()
	if len(event.Data) == 0 {
		return nil, true, nil // continue
	}

	return s.processEventData(event)
}

// handleDecoderEnd handles the case when decoder has no more events
func (s *providerStream) handleDecoderEnd() (*StreamToken, bool, error) {
	if err := s.decoder.Err(); err != nil {
		if s.retryStrategy.ShouldRetry(err) {
			time.Sleep(s.retryStrategy.NextDelay())
			return nil, true, nil // continue
		}
		return nil, false, err
	}
	return nil, false, io.EOF
}

// processEventData processes the event data and creates a stream token
func (s *providerStream) processEventData(event Event) (*StreamToken, bool, error) {
	resp, err := s.provider.ParseStreamResponse(event.Data)
	if err != nil {
		if err.Error() == "skip resp" {
			return nil, true, nil // continue
		}
		if errors.Is(err, io.EOF) {
			return nil, false, io.EOF
		}
		return nil, true, nil // continue - Not enough data or malformed
	}

	return s.createStreamToken(event, resp), false, nil
}

// createStreamToken creates a stream token from the response
func (s *providerStream) createStreamToken(event Event, resp *providers.Response) *StreamToken {
	streamToken := &StreamToken{
		Text:  "",
		Type:  event.Type,
		Index: s.currentIndex,
	}

	if resp == nil {
		return streamToken
	}

	if resp.Content != nil {
		streamToken.Text = resp.AsText()
	}

	if resp.Usage != nil {
		streamToken.InputTokens = resp.Usage.InputTokens
		streamToken.OutputTokens = resp.Usage.OutputTokens
	}

	return streamToken
}

func (s *providerStream) Close() error {
	return nil
}
