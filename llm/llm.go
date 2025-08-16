// Package llm provides a unified interface for interacting with various Language Learning Model providers.
// It abstracts away provider-specific implementations and provides a consistent API for text generation,
// prompt management, and error handling.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/providers"
	"github.com/weave-labs/gollm/types"
	"github.com/weave-labs/gollm/utils"
)

// Constants for configuration defaults
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
	// SupportsStreaming checks if the provider supports streaming responses.
	SupportsStreaming() bool
	// SetOption configures a provider-specific option.
	// Returns ErrorTypeInvalidInput if the option is not supported.
	SetOption(key string, value any)
	// SetLogLevel adjusts the logging verbosity.
	SetLogLevel(level utils.LogLevel)
	// SetEndpoint updates the API endpoint (primarily for local models).
	// Returns ErrorTypeProvider if the provider doesn't support endpoint configuration.
	SetEndpoint(endpoint string)
	// NewPrompt creates a new prompt instance.
	NewPrompt(input string) *Prompt
	// GetLogger returns the current logger instance.
	GetLogger() utils.Logger
	// SupportsStructuredResponse checks if the provider supports JSON schema validation.
	SupportsStructuredResponse() bool
}

// LLMImpl implements the LLM interface and manages interactions with specific providers.
// It handles provider communication, error management, and logging.
//
//nolint:revive // LLMImpl clearly indicates this is the implementation of the LLM interface
type LLMImpl struct {
	Provider     providers.Provider
	logger       utils.Logger
	Options      map[string]any
	client       *http.Client
	config       *config.Config
	MaxRetries   int
	RetryDelay   time.Duration
	optionsMutex sync.RWMutex
}

// NewLLM creates a new LLM instance with the specified configuration.
// It initializes the appropriate provider and sets up logging and HTTP clients.
//
// Returns:
//   - Configured LLM instance
//   - ErrorTypeProvider if provider initialization fails
//   - ErrorTypeAuthentication if API key validation fails
func NewLLM(cfg *config.Config, logger utils.Logger, registry *providers.ProviderRegistry) (*LLMImpl, error) {
	extraHeaders := make(map[string]string)
	if cfg.Provider == "anthropic" && cfg.EnableCaching {
		extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	// Check if API key is empty
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
// The option is logged at debug level for troubleshooting.
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
func (l *LLMImpl) SetLogLevel(level utils.LogLevel) {
	l.logger.Debug("Setting internal LLM log level", "new_level", level)
	l.logger.SetLevel(level)
}

// GetLogger returns the current logger instance.
func (l *LLMImpl) GetLogger() utils.Logger {
	return l.logger
}

// NewPrompt creates a new prompt instance with the given input text.
func (l *LLMImpl) NewPrompt(prompt string) *Prompt {
	return &Prompt{Input: prompt}
}

// SupportsStructuredResponse checks if the current provider supports JSON schema validation.
func (l *LLMImpl) SupportsStructuredResponse() bool {
	return l.Provider.SupportsStructuredResponse()
}

// Generate produces text based on the given prompt and options.
// It handles retries, logging, and error management.
//
// Returns:
//   - Generated text response
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
func (l *LLMImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (*providers.Response, error) {
	generateConfig := &GenerateConfig{}
	for _, opt := range opts {
		opt(generateConfig)
	}

	// Set the system prompt in the LLM's options
	if prompt.SystemPrompt != "" {
		l.SetOption("system_prompt", prompt.SystemPrompt)
	}

	// If a Structured Response schema is provided, use the schema-aware generation path
	if generateConfig.StructuredResponseSchema != nil {
		return l.generateWithSchema(ctx, prompt, generateConfig.StructuredResponseSchema)
	}

	return l.generateWithRetries(ctx, prompt)
}

// generateWithSchema handles generation with structured response schema
func (l *LLMImpl) generateWithSchema(ctx context.Context, prompt *Prompt, schema any) (*providers.Response, error) {
	var result *providers.Response
	var lastErr error
	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug(
			"Generating text (schema)",
			"provider", l.Provider.Name(),
			"prompt", prompt.String(),
			"attempt", attempt+1,
		)
		result, _, lastErr = l.attemptGenerateWithSchema(ctx, prompt.String(), schema)
		if lastErr == nil {
			return result, nil
		}
		l.logger.Warn("Generation attempt with schema failed", "error", lastErr, "attempt", attempt+1)
		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			if err := l.wait(ctx); err != nil {
				return nil, err
			}
		}
	}
	return nil, fmt.Errorf("failed to generate with schema after %d attempts: %w", l.MaxRetries+1, lastErr)
}

// generateWithRetries handles standard generation with retry logic
func (l *LLMImpl) generateWithRetries(ctx context.Context, prompt *Prompt) (*providers.Response, error) {
	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug(
			"Generating text",
			"provider", l.Provider.Name(),
			"prompt", prompt.String(),
			"system_prompt", prompt.SystemPrompt,
			"attempt", attempt+1,
		)
		result, err := l.attemptGenerate(ctx, prompt)
		if err == nil {
			return result, nil
		}
		l.logger.Warn("Generation attempt failed", "error", err, "attempt", attempt+1)
		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
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
//
// Returns:
//   - Generated text response
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt *Prompt) (*providers.Response, error) {
	response := &providers.Response{}

	// Prepare options and request body
	options := l.prepareOptions(prompt)
	reqBody, err := l.prepareRequestBody(prompt, options)
	if err != nil {
		return response, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	// Execute the request and get response
	return l.executeRequest(ctx, reqBody)
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

	// Add Tools and ToolChoice to options
	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if len(prompt.ToolChoice) > 0 {
		options["tool_choice"] = prompt.ToolChoice
	}

	return options
}

// prepareRequestBody prepares the request body based on message type
func (l *LLMImpl) prepareRequestBody(prompt *Prompt, options map[string]any) ([]byte, error) {
	// Check if we have structured messages
	l.optionsMutex.RLock()
	structuredMessages, hasStructuredMessages := l.Options["structured_messages"]
	l.optionsMutex.RUnlock()

	if !hasStructuredMessages {
		reqBody, err := l.Provider.PrepareRequest(prompt.String(), options)
		if err != nil {
			return nil, fmt.Errorf("provider PrepareRequest failed: %w", err)
		}
		return reqBody, nil
	}

	// Use the structured messages API if the provider supports it
	if prepareWithMessages, ok := l.Provider.(interface {
		PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]any) ([]byte, error)
	}); ok {
		// Convert to the expected type
		messages, ok := structuredMessages.([]types.MemoryMessage)
		if ok {
			l.logger.Debug("Using structured messages API", "message_count", len(messages))
			reqBody, err := prepareWithMessages.PrepareRequestWithMessages(messages, options)
			if err != nil {
				return nil, fmt.Errorf("provider PrepareRequestWithMessages failed: %w", err)
			}
			return reqBody, nil
		}
		l.logger.Warn("Invalid structured_messages format", "type", fmt.Sprintf("%T", structuredMessages))
	}

	l.logger.Debug("Provider does not support structured messages API", "provider", l.Provider.Name())
	reqBody, err := l.Provider.PrepareRequest(prompt.String(), options)
	if err != nil {
		return nil, fmt.Errorf("provider PrepareRequest failed: %w", err)
	}
	return reqBody, nil
}

// executeRequest sends the HTTP request and processes the response
func (l *LLMImpl) executeRequest(ctx context.Context, reqBody []byte) (*providers.Response, error) {
	response := &providers.Response{}

	l.logger.Debug("Full request body", "body", string(reqBody))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return response, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	l.logger.Debug(
		"Full API request",
		"method",
		req.Method,
		"url",
		req.URL.String(),
		"headers",
		req.Header,
		"body",
		string(reqBody),
	)
	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
		l.logger.Debug("Request header", "provider", l.Provider.Name(), "key", k, "value", v)
	}
	resp, err := l.client.Do(req)
	if err != nil {
		return response, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			l.logger.Error("Failed to close response body", "error", err)
		}
	}()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return response, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	// Log the full API response
	l.logger.Debug("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return response, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return response, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	l.logger.Debug("Text generated successfully", "result", result)
	return result, nil
}

// attemptGenerateWithSchema makes a single attempt to generate text using the provider and a JSON schema.
// It handles request preparation, API communication, and response processing.
//
// Returns:
//   - Generated text response
//   - Full prompt used for generation
//   - ErrorTypeInvalidInput for schema validation failures
//   - Other error types as per attemptGenerate
func (l *LLMImpl) attemptGenerateWithSchema(
	ctx context.Context,
	prompt string,
	schema any,
) (*providers.Response, string, error) {
	// Prepare request with schema
	reqBody, fullPrompt, err := l.prepareSchemaRequest(prompt, schema)
	if err != nil {
		return nil, fullPrompt, err
	}

	// Execute the request
	response, err := l.executeSchemaRequest(ctx, reqBody)
	if err != nil {
		return nil, fullPrompt, err
	}

	// Validate the result against the schema
	textContent, ok := response.Content.(providers.Text)
	if !ok {
		return nil, fullPrompt, NewLLMError(ErrorTypeResponse, "response content is not text", nil)
	}
	if err := ValidateAgainstSchema(textContent.Value, schema); err != nil {
		return nil, fullPrompt, NewLLMError(ErrorTypeResponse, "response does not match schema", err)
	}

	return response, fullPrompt, nil
}

// prepareSchemaRequest prepares a request with JSON schema
func (l *LLMImpl) prepareSchemaRequest(prompt string, schema any) ([]byte, string, error) {
	l.optionsMutex.RLock()
	options := make(map[string]any)
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	var reqBody []byte
	var err error
	var fullPrompt string

	if l.SupportsStructuredResponse() {
		reqBody, err = l.Provider.PrepareRequestWithSchema(prompt, options, schema)
		fullPrompt = prompt
	} else {
		fullPrompt = l.preparePromptWithSchema(prompt, schema)
		reqBody, err = l.Provider.PrepareRequest(fullPrompt, options)
	}

	if err != nil {
		return nil, fullPrompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))
	return reqBody, fullPrompt, nil
}

// executeSchemaRequest executes a request and validates against schema
func (l *LLMImpl) executeSchemaRequest(ctx context.Context, reqBody []byte) (*providers.Response, error) {
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
		return nil, NewLLMError(
			ErrorTypeAPI,
			fmt.Sprintf("API error: status code %d", resp.StatusCode),
			nil,
		)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return nil, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	l.logger.Debug("Text generated successfully", "result", result)
	return result, nil
}

// preparePromptWithSchema prepares a prompt with a JSON schema for providers that do not support JSON schema
// validation.
// Returns the original prompt if schema marshaling fails (with a warning log).
func (l *LLMImpl) preparePromptWithSchema(prompt string, schema any) string {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		l.logger.Warn("Failed to marshal schema", "error", err)
		return prompt
	}

	return fmt.Sprintf(
		"%s\n\nPlease provide your response in JSON format according to this schema:\n%s",
		prompt,
		string(schemaJSON),
	)
}

// GenerateStream initiates a streaming response from the LLM.
func (l *LLMImpl) GenerateStream(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (TokenStream, error) {
	if !l.SupportsStreaming() {
		return nil, NewLLMError(ErrorTypeUnsupported, "streaming not supported by provider", nil)
	}

	// Apply stream options
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

	body, err := l.Provider.PrepareStreamRequest(prompt.String(), options)
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to prepare stream request", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.Provider.Endpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to create stream request", err)
	}

	// Add headers
	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	// Make request
	resp, err := l.client.Do(req)
	if err != nil {
		return nil, NewLLMError(ErrorTypeAPI, "failed to make stream request", err)
	}

	if resp.StatusCode != http.StatusOK {
		if err := resp.Body.Close(); err != nil {
			l.logger.Error("Failed to close response body", "error", err)
		}
		return nil, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	// Create and return stream
	return newProviderStream(resp.Body, l.Provider, generateConfig), nil
}

// SupportsStreaming checks if the provider supports streaming responses.
func (l *LLMImpl) SupportsStreaming() bool {
	return l.Provider.SupportsStreaming()
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

func newProviderStream(reader io.ReadCloser, provider providers.Provider, config *GenerateConfig) *providerStream {
	return &providerStream{
		decoder:       NewSSEDecoder(reader),
		provider:      provider,
		config:        streamConfig,
		buffer:        make([]byte, 0, DefaultStreamBufferSize),
		currentIndex:  0,
		retryStrategy: streamConfig.RetryStrategy,
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
