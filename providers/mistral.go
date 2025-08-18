// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/weave-labs/gollm/config"
	"github.com/weave-labs/gollm/internal/logging"
)

// Common parameter keys
const (
	mistralKeyMaxTokens      = "max_tokens"
	mistralKeyStream         = "stream"
	mistralKeyModel          = "model"
	mistralKeyMessages       = "messages"
	mistralKeySystemPrompt   = "system_prompt"
	mistralKeyTools          = "tools"
	mistralKeyToolChoice     = "tool_choice"
	mistralKeyResponseFormat = "response_format"
	mistralKeyStrict         = "strict"
	mistralKeyTemperature    = "temperature"
	mistralKeySeed           = "seed"
)

// MistralProvider implements the Provider interface for Mistral AI's API.
// It supports Mistral's language models and provides access to their capabilities,
// including chat completion and structured output.
type MistralProvider struct {
	logger       logging.Logger
	extraHeaders map[string]string
	options      map[string]any
	apiKey       string
	model        string
}

// NewMistralProvider creates a new Mistral provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Mistral API key for authentication
//   - model: The model to use (e.g., "mistral-large", "mistral-medium")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Mistral Provider instance
func NewMistralProvider(apiKey, model string, extraHeaders map[string]string) *MistralProvider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &MistralProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       logging.NewLogger(logging.LogLevelInfo),
	}
}

// SetLogger configures the logger for the Mistral provider.
// This is used for debugging and monitoring API interactions.
func (p *MistralProvider) SetLogger(logger logging.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the Mistral provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - random_seed: Random seed for deterministic sampling
func (p *MistralProvider) SetOption(key string, value any) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *MistralProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption(mistralKeyTemperature, cfg.Temperature)
	p.SetOption(mistralKeyMaxTokens, cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption(mistralKeySeed, *cfg.Seed)
	}
}

// Name returns "mistral" as the provider identifier.
func (p *MistralProvider) Name() string {
	return "mistral"
}

// Endpoint returns the Mistral API endpoint URL.
// This is "https://api.mistral.ai/v1/chat/completions".
func (p *MistralProvider) Endpoint() string {
	return "https://api.mistral.ai/v1/chat/completions"
}

// SupportsStructuredResponse indicates that Mistral supports structured output
// through its system prompts and response formatting capabilities.
// All models support structured output except codestral-mamba.
func (p *MistralProvider) SupportsStructuredResponse() bool {
	return p.model != "codestral-mamba"
}

// SupportsStreaming returns whether the provider supports streaming responses.
// All Mistral models support streaming.
func (p *MistralProvider) SupportsStreaming() bool {
	return true
}

// SupportsFunctionCalling indicates if the provider supports function calling.
// Only specific models support function calling.
func (p *MistralProvider) SupportsFunctionCalling() bool {
	supportedModels := map[string]bool{
		"mistral-large-latest":  true,
		"mistral-medium-latest": true,
		"mistral-small-latest":  true,
		"devstral-small-latest": true,
		"codestral-latest":      true,
		"ministral-8b-latest":   true,
		"ministral-3b-latest":   true,
		"pixtral-12b-latest":    true,
		"pixtral-large-latest":  true,
		"open-mistral-nemo":     true,
	}
	return supportedModels[p.model]
}

// Headers returns the required HTTP headers for Mistral API requests.
// This includes:
//   - Authorization: Bearer token using the API key
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *MistralProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

// PrepareRequest creates the request body for a Mistral API call using the new Request structure.
func (p *MistralProvider) PrepareRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()

	// Add system prompt if present
	systemPrompt := p.extractSystemPromptFromRequest(req, options)
	if systemPrompt != "" {
		p.addSystemPromptToRequestBody(requestBody, systemPrompt)
	}

	// Add messages
	p.addMessagesToRequestBody(requestBody, req.Messages)

	// Add structured response if supported
	if req.ResponseSchema != nil && p.SupportsStructuredResponse() {
		p.addStructuredResponseToRequest(requestBody, req.ResponseSchema)
	}

	// Add remaining options
	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseResponse extracts the generated text from the Mistral API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *MistralProvider) ParseResponse(body []byte) (*Response, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
		return nil, errors.New("empty response from API")
	}

	// Combine content and tool calls
	var finalResponse strings.Builder
	finalResponse.WriteString(response.Choices[0].Message.Content)

	// Process tool calls if present
	for _, toolCall := range response.Choices[0].Message.ToolCalls {
		// Parse arguments as raw JSON to preserve the exact format
		var args any
		if err := json.Unmarshal(toolCall.Function.Arguments, &args); err != nil {
			return nil, fmt.Errorf("error parsing function arguments: %w", err)
		}

		functionCall, err := FormatFunctionCall(toolCall.Function.Name, args)
		if err != nil {
			return nil, fmt.Errorf("error formatting function call: %w", err)
		}
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(functionCall)
	}

	return &Response{Content: Text{Value: finalResponse.String()}}, nil
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *MistralProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *MistralProvider) PrepareStreamRequest(req *Request, options map[string]any) ([]byte, error) {
	requestBody := p.initializeRequestBody()
	requestBody[mistralKeyStream] = true

	// Add system prompt if present
	systemPrompt := p.extractSystemPromptFromRequest(req, options)
	if systemPrompt != "" {
		p.addSystemPromptToRequestBody(requestBody, systemPrompt)
	}

	// Add messages
	p.addMessagesToRequestBody(requestBody, req.Messages)

	// Add structured response if supported
	if req.ResponseSchema != nil && p.SupportsStructuredResponse() {
		p.addStructuredResponseToRequest(requestBody, req.ResponseSchema)
	}

	// Add remaining options
	p.addRemainingOptions(requestBody, options)

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}
	return data, nil
}

// ParseStreamResponse parses a single chunk from a streaming response
func (p *MistralProvider) ParseStreamResponse(chunk []byte) (*Response, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return nil, errors.New("empty chunk")
	}
	// [DONE] guard
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return nil, io.EOF
	}

	var response struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(chunk, &response); err != nil {
		return nil, fmt.Errorf("malformed response: %w", err)
	}

	if len(response.Choices) == 0 || response.Choices[0].Delta.Content == "" {
		return nil, errors.New("skip token")
	}

	return &Response{Content: Text{Value: response.Choices[0].Delta.Content}}, nil
}

// initializeRequestBody creates the base request structure
func (p *MistralProvider) initializeRequestBody() map[string]any {
	return map[string]any{
		mistralKeyModel:     p.model,
		mistralKeyMaxTokens: p.options[mistralKeyMaxTokens],
		mistralKeyMessages:  []map[string]any{},
	}
}

// extractSystemPromptFromRequest gets system prompt from request or options
func (p *MistralProvider) extractSystemPromptFromRequest(req *Request, options map[string]any) string {
	// Priority: Request.SystemPrompt > options["system_prompt"]
	if req.SystemPrompt != "" {
		return req.SystemPrompt
	}
	if sp, ok := options[mistralKeySystemPrompt].(string); ok && sp != "" {
		return sp
	}
	return ""
}

// addSystemPromptToRequestBody adds the system prompt to the request
func (p *MistralProvider) addSystemPromptToRequestBody(requestBody map[string]any, systemPrompt string) {
	if systemPrompt == "" {
		return
	}

	if messagesArray, ok := requestBody[mistralKeyMessages].([]map[string]any); ok {
		systemMessage := map[string]any{
			"role":    "system",
			"content": systemPrompt,
		}
		requestBody[mistralKeyMessages] = append(messagesArray, systemMessage)
	}
}

// addMessagesToRequestBody converts Request messages to Mistral format
func (p *MistralProvider) addMessagesToRequestBody(requestBody map[string]any, messages []Message) {
	if messagesArray, ok := requestBody[mistralKeyMessages].([]map[string]any); ok {
		for _, msg := range messages {
			mistralMessage := map[string]any{
				"role":    msg.Role,
				"content": msg.Content,
			}
			if msg.Name != "" {
				mistralMessage["name"] = msg.Name
			}
			if len(msg.ToolCalls) > 0 {
				mistralMessage["tool_calls"] = msg.ToolCalls
			}
			if msg.ToolCallID != "" {
				mistralMessage["tool_call_id"] = msg.ToolCallID
			}
			messagesArray = append(messagesArray, mistralMessage)
		}
		requestBody[mistralKeyMessages] = messagesArray
	}
}

// addStructuredResponseToRequest adds structured response schema to the request
func (p *MistralProvider) addStructuredResponseToRequest(requestBody map[string]any, schema any) {
	requestBody[mistralKeyResponseFormat] = map[string]any{
		"type":   "json_schema",
		"schema": schema,
	}
}

// addRemainingOptions adds provider options and additional options to the request
func (p *MistralProvider) addRemainingOptions(requestBody map[string]any, options map[string]any) {
	// Add provider options first
	for k, v := range p.options {
		if k != mistralKeyMaxTokens { // Already added in initialize
			requestBody[k] = v
		}
	}

	// Add additional options (may override provider options)
	for k, v := range options {
		if k != mistralKeySystemPrompt { // Already handled
			requestBody[k] = v
		}
	}
}
